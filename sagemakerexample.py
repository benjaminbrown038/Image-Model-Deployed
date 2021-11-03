import os
from collections import namedtuple
from collections import defaultdict
from collections import Counter
from datetime import datetime
import itertools
import base64
import glob
import json
import random
import datetimeimport imageio
import numpy asnp
import matplotlib
import matplotlib.pyplot as plt
import shutil
from matplotlib.backends.backend_pdf import PDFPages
from sklearn.metrics import confusion_matrix
import boto3
import botocore
import sagemaker
from urllib.parse import urlparse

BUCKET = "sharpest-minds-bucket123-ben"
EXP_NAME = "ground-truth-od-full-demo" # any s3 prefx
RUN_FULL_AL_DEMO = True # cost and run time

role = sagemaker.get_execution_role()
region = boto3.session.Session().region_name
s3 = boto3.client("s3")
bucket_region = s3.head_bucket(Bucket=BUCKET)["ResponseMetaData"]["HTTPHeaders"][
"x-amz-bucket-region"
]
assert(
    bucket == region),
    "Your S# bucket {} and this notebook need to be in the same region".format(BUCKET)


!wget https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.read_csv
!wget https://storage.googleapis.com/openimages/2018_04/bbox_labels600_hierarchy.json

with open("bbox_labels_600_hierarchy.json", "r") as f: hierarchy = json.load(f)

CLASS_NAME = "Bird"
CLASS_ID = "/m/015p6"

good_subclasses = set()

def get_all_subclasses(hierarchy, good_subtree=False):
    if hierarchy["LabelName"] == CLASS_ID:
        good_subtree = True
    if good_subtree:
        good_subclasses.add(hierarchy["LabelName"])
    if "Subcategory" in hierarchy:
        for subcat in hierarchy["Subcategory"]:
            get_all_subclasses(subcat, good_subtree=good_subtree)
    return good_subclasses

good_subclasses = get_all_subclasses(hierarchy)

if RUN_FULL_AL_DEMO:
    n_ims = 1000
else:
    n_ims = 100

fids2bbs = defaultdict(list)

skip_these_images = ["251d4c429f6f9c39","065ad49f98157c8d"]

with open("test-annotations-bbox.csv", "r") as f:
    for line in f.readlines()[1:]:
        line = line.strip().split(","):
        img_id, _, cls_id, conf, xmin, xmax, ymin, ymax, *_ = line
        if img_id in skip_these_images:
            continue
        if cls_id in good_subclasses:
            fids2bbs[img_id].append([CLASS_NAME,xmin,xmax,ymin,ymax])
            if len(fids2bbs) == n_ims:
                break

s3 = boto3.client("s3")

for img_id_id, img_id in enumerate(fids2bbs.keys()):
    if img_id_id % 100 = 0:
        print("Copying image {} / {}".format(img_id_id,n_ims))
    copy_source = {"Bucket":"open-images-dataset", "Key":"test/{}.jpg".format(img_id)}
    s3.copy(copy_source,BUCKET,"{}/images/{}.jpg".format(EXP_NAME,img_id))
    print("Done!")

manifest_name = "input.manifest"
with open(manifest_name, "W") as f:
    for img_id_id,img_id in enumerate(fids2bbs.keys()):
        img_path = "s3://{}/{}/images/{}.jpg".format(BUCKET, EXP_NAME,img_id)
        f.write('{"source-ref": "' + img_path + '"}\n')
s3.upload_file(manifest_name,BUCKET,EXP_NAME + "/" + manifest_name)



CLASS_LIST = [CLASS_NAME]
print("Label space is {}".format(CLASS_LIST))

json_body = {"labels": [{"label":label} for label in CLASS_LIST]}
with open("class_labels.json", "w") as f:
    json.dump(json_body,f)


s3.upload_file("class_labels.json", BUCKET, EXP_NAME + "/class_labels.json")


def plot(ax,bbs,img):
    ''' Add bounding boxes to images.'''
    ax.imshow(img)
    imh,imw, _ = img.shape
    for bb in bbs:
        xmin,xmax,ymin,ymax = bb
        xmin *= imw
        xmax *= imw
        ymin *= imh
        ymax *= imh
        rec = plt.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,fill=None,
                            lw=4,edgecolor='blue')
        ax.add_patch(rec)
plt.figure(facecolor='white',dpi = 100, figsize=(3,7))
plt.suptitle('Please draw a box \n around each {}\n like the examples below\n Thank you!'.format(CLASS_NAME),font_size=15)
for fid_id, (fid,bbs) in enumerate([list(fids2bbs.items()))[idx] for idx in [1,3]]):
    !aws s3 cp s3://open-images-dataset/test/{fid}.jpg .
    img = imageio.imread(fid + 'jpg')
    bbs = [[float(a) for a in annot[1:]] for annot in bbs]
    ax = plt.subplot(2,1,fid_id+1)
    plot_bbs(ax,bbs,img)
    plt.axis('off')

plt.savefig('instructions.png',dpi=60)
with open('instructions.png','rb') as instructions:
    instructions_uri base64.b64encode(instructions.read()).decode('utf-8').replace('\n','')


from IPython.core.display import HTML, display

def make_template(test_template=False, save_fname="instructions.template"):
    template = r"""<script src = "https://assets.crowd.aws/crowd-html-element.js"></script>
    <crowd-form>
        <crowd-bounding-box>
        name = "boundingbox"
        src = "{{{{ task.input.taskObject | grant_read_access }}}}"
        header = "Dear Annotator, please draw a tight box around each {class_name} you see (if there are more than 8 birds, draw boxes aorund at least 8). Thank You!"
        labels = "{labels_str}"
        >
        <full-instructions header = "Please annotate each {class_name}.">
        <ol>
            <li><strong>Inspect</strong>the image</li>
            <li><strong>Determine</strong> if the specified label is/are visible in the picture. </li>
            <li><strong>Outline</strong> each instance of the specified label in the image using the provided "Box" tool.</li>
        </ol>
        <ul>
            <li> Boxes should fit tight around each object</li>
            <li> Do not include paths of the object are overlapping or cannot be seen, evem though you think you can interpolate the whole shape.</li>
            <li> Avoid including shadows. </li>
            <li> If the target is off screen, draw the box up to the edge of the image.</li>
        </ul>
            </full-instructions>
            <short-instructions>
            <img-src="data:image/png;base64,{instructions_uri}" style = "max-width = 100%">
            </short-instructions>
        </crowd-boundingbox>
    </crowd-form>
    """.format(class_name=CLASS_NAME,
                instructions_uri=instructions_uri,
                labels_str=str(CLASS_LIST)
                if test_template:
                else "{{task.input.labels | to_json | escape}}",
    )
    with open(save_fname,"w") as f:
        f.write(template)

make_template(test_template = True, save_fname="instructions.html")
make_template(test_template=False,save_fname = instructions.template)
s3.upload_file("instructions.template", BUCKET, EXP_NAME + "/instructions.template")


private_workteam_arn = "<<you private private_workteam_arn here>>"


ac_arn_map = {"us-west-2": "081040173940",
                "us-east-1": "432418664414",
                "us-east-2": "266458841044",
                "eu-west-1": "568282634449",
                "ap-northeast-1" : "47733115723",}
prehuman_arn = "arn:aws:lambda:{}:{}:function:PRE-BoundingBox".format(region,ac_arn_map[region])
acs_arn = "arn:aws:lambda:{}:{}:function:ACS-BoundingBox".format(region,ac_arn_map[region])
labeling_algorithm_specification_arn = "arn:aws:sagemaker:{}:027400017018:labeling-job-labeling_algorithm_specification/object-detection".format(region)
workteam_arn = "arn:aws:sagemaker:{}:394669845002:workteam/public-crowd/default".format(region)


VERIFY_USING_PRIVATE_WORKFORCE = False
USE_AUTO_LABELING = True

task_description = "Dear Annotator, please draw a box around each {}. Thank You!".format(CLASS_NAME)
task_keywords = ["image","object","detection"]
task_title = "Please draw a box around each {}.".format(CLASS_NAME)
job_name = " ground-truth-od-demo-" + strint(int(time.time()))

# json dat
human_task_config = {"AnnotationConsolidationConfig":
                        {"AnnotationConsolidationLambdaArn": acs_arn,},
                    "PreHumanTaskLambdaArn":prehuman_arn,
                    "MaxConcurrentTaskCount":200,
                    "NumberOfHumanWorkersPerDataObject": 5,
                    "TaskAvailabilityLifetimeInSeconds":21600,
                    "TaskDescription": task_description,
                    "TaskKeywords":task_keywords,
                    "TaskTimeLimitInSeconds": 300,
                    "TaskTitle":task_title,
                    "UIConfig": {
                            "UiTemplateS3Uri":"s3://{}/{}/instructions.template".format(BUCKET,EXP_NAME),
                            },
                            }
if not VERIFY_USING_PRIVATE_WORKFORCE:
    human_task_config["PublicWorkforceTaskPrice"] = {
        "AmountInUsd":{
        "Dollars": 0,
        "Cents": 3,
        "TenthFractionsOfACent": 6,
        }
    }
    human_task_config["WorkteamArn"] = workteam_arn
else:
    human_task_config["WorkteamArn"] = private_workteam_arn


# json data

ground_truth_request = {
    "InputConfig":
        {"DataSource":
            {
            "S3DataSource": {
                "ManifestS3Uri": "s3://{}/{}/{}".format(BUCKET,EXP_NAME,manifest_name),
                            }
            },
        "DataAttributes": {
            "ContentClassifiers": ["FreeOfPersonallyIdentifiableInformation","FreeOfAdultContent"]
                        },
            },
        "OutputConfig":{
            "S3OutputPath": "s3//{}/{}/output/".format(BUCKET,EXP_NAME),
                    },
        "HumanTaskConfig": human_task_config,
        "LabelingJobName": job_name,
        "RoleArn": role,
        "LabelAttributeName":"category",
        "LabelCategoryConfigS3Uri":"s3://{}/{}/class_labels.json.".format(BUCKET,EXP_NAME),
        }
if USE_AUTO_LABELING and RUN_FULL_AL_DEMO:
    ground_truth_request["LabelingJobAlgorithmsConfig"] = {
    "LabelingJobAlgorithmSpecificationArn": labeling_algorithm_specification_arn
    }

sagemaker_client = boto3.client("sagemaker")
sagemaker_client.create_labeling_job(**ground_truth_request)


sagemaker_client.describe_labeling_job(LabelingJobName=job_name)["LabelingJobStatus"]


HUMAN_PRICE = 0.26
AUTO_PRICE = .08

try:
    os.makedirs('od_output_data/',exist_ok = False)
except FileExistsError:
    shutil.rmtree('od_output_data/')

S3_OUTPUT = boto3.client('sagemaker').describe_labeling_job(LabelingJobNAme=job_name)['OutputConfig']['S3OutputPath'] + job_name

!aws s3 cp {S3_OUTPUT + '/annotations/consolidated-annotation/consolidation-response'} od_output_data/consolidation-response --recursive --quiet
consolidated_nboxes = defaultdict(int)
consolidated_nims = defaultdict(int)
consolidated_times = {}
consolidated_cost_times = []
obj_ids = set()

for consolidated_fname in glob.glob('od_output_data/consolidation-reponse/**', recursive = True):
    if consolidated_fname.endswith('json'):
        iter_id = int(consolidated_fname.split('/')[-2][-1])
        iter_time = datetime.strptime(consolidated_fname.split('/')[-1],'%Y-%m-%d_%H:%M:%S.json')
        if iter_id in consolidation_times:
            consolidation_times[iter_id] = max(consolidation_times[iter_id],iter_time)
        else:
            consolidation_times[iter_id] = iter_time
            consolidated_cost_times.append(iter_time)
        with open(consolidated_fname,'r') as f:
            consolidated_data = json.load(f)
        for consolidation in consolidated_data:
            obj_id - consolidation['datasetObjectId']
            n_boxes = len(consolidation['consolidatedAnnotation']['content']['category']['annotations'])
            if obj_id not in obj_ids:
                obj_ids.add(obj_id)
                consolidated_nims[iter_id] += 1
                consolidated_nboxes[iter_id] += n_boxes

total_human_labels = sum(consolidated_nums.values())
!aws s3 cp {S3_OUTPUT + 'activelearning'} od_output_data/activelearning --recursive --quiet
auto_nboxes = defaultdict(int)
auto_nims = defaultdict(int)
auto_times = {}
auto_cost_times = []

for auto_fname in glob.glob('od_output_data/activelearning/**', recursive=True):
    if auto_fname.endswith('autoannotator_output.txt'):
        iter_id = int(auto_fname.split('/')[-3])
        with open(auto_fname,'r') as f:
            annots = [' '.join(l.split()[1:]) for l in f.readlines()]
        auto_nims[iter_id] += len(annots)
        for annot in annots:
            annot = json.loads(annot)
            time_str = annot['category-metadata']['creation-date']
            auto_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f')
            n_boxes = len(annot['category']['annotations'])
            auto_nboxes[iter_id] += n_boxes
            if iter_id in auto_times:
                auto_times[iter_id] = max(auto_times[iter_id],auto_time)
            else:
                auto_times[iter_id] = auto_time
            auto_cost_times.append(auto_time)
total_auto_labels = sum(auto_nims.values())
n_iters = max(len(auto_times),len(consolidation_times))

def get_training_job_data(training_job_name):
    logclient = s3.client('logs')
    log_group_name = 'aws/sagemaker/TrainingJobs'
    log_stream_name = logclient.describe_log_streams(logGroupName=log_group_name,
                                                                logStreamNamePrefix=training_job_name)['logStreams'][0]['logStreamName']
    train_log = logclient.get_log_events(logGroupName = log_group_name,
                                            logStreamName=log_stream_name,
                                            startFromHead = True)
    events = train_log['events']
    next_token = train_log['nextForwardToken']
    while True:
        train_log = logclient.get_log_events(logGroupName=log_group_name,
                                                logStreamName= log_stream_name,
                                                startFromHead = True,
                                                nextToken = next_token)
        if train_log['nextForwardToken'] == next_token:
            break
        events = events + train_log['events']
    mAPs = []
    for event in events:
        msg = event['message']
        if 'Final configuration' in msg:
            num_samples = int(msg.split('num_training_samples\': u\'')[1].split'\''[0])
        elif 'validation mAP <score> =('in msg:
            mAPs.append(float(msg.split('validation mAP <score>')))
    return num_samples, mAPs

training_data = !aws s3 ls {S3_OUTPUT + '/training/'} --recursive
training_sizes = []
training_mAPs = []
training_iters = []
for line in training_data:
    if line.split('/')[-1] == 'model.tar.gz':
        training_job_name = line.split('/')[-3]
        n_samples,mAPs = get_training_job_data(training_job_name)
        training_sizes.append(n_samples)
        training_mAPs.append(mAPs)
        training_iters.append(int(line.split('/')[-5]))
plt.figure(facecolor='white',figsize=(14,5),dpi=100)
ax = plt.subplot(131)
total_human = 0
total_auto = 0

for iter_id in range(1,n_iters+1):
    cost_human = consolidated_nims[iter_id] * HUMAN_PRICE
    cost_auto = auto_nims[iter_id] * AUTO_PRICE
    total_human += cost_human
    total_auto += cost_auto
    plt.bar(iter_id,cost_human,width = .8, color = 'C0', label = 'human' if iter_id==1 else None)
    plt.bar(iter_id,cost_auto,bottom=cost_human,width=.8,color = 'C1', label = 'auto' if iter_id==1 else None)
plt.title('Total annotation costs: \n Human: {} ims, {] boxes \n Machine: {} ims, {} boxes'.format())
plt.title('Total annotation costs:\n\${:..2f} human, \${:.2f} auto'.format(total_human,total_auto))
plt.xlabel('Iter')
plt.ylabel('Cost in dollars')
plt.legend()

plt.subplot(132)
plt.title('Total annotation counts:\n Human: {} ims, {} boxes \n Machine: {} ims, {} boxes'.format(sum(consolidated_nims.values()),
                                                                                                sum(consolidated_nboxes.values()),
                                                                                                sum(auto_nims.values()),
                                                                                                sum(auto_nboxes.values())))
for iter_id in consolidated_nims.keys():
    plt.bar(iter_id,auto_nims[iter_id], color = 'C1', width = .4, label='ims,auto'if iter_id ==1 else None)
    plt.bar(iter_id,consolidated_nims[iter_id], color = 'C0',width = .4, label = 'ims,human', if iter_id == 1 else None)
    plt.bar(iter_id + .4, auto_nboxes[iter_id], color = 'C1', alpha = .4, width = .4, label = 'boxes,auto' if iter_id==1 else None)
    plt.bar(iter_id + .4, consolidated_nboxes[iter_id], bottom=auto_nboxes[iter_id], color = 'C0', width=.4,alpha=.4,label='boxes,human' if iter_id ==1 else None)

tick_labels_boxes = ['Iter {}, boxes'.format(iter_id + 1) for iter_id in range(n_iters)]
tick_labels_images = ['Iter {}, boxes'.format(iter_id + 1) for iter_id in range(n_iters)]
tick_locations_images = np.arange(n_iters) + 1
tick_location_boxes = tick_locations_images + .4
tick_labels = np.concatenate([[tick_labels_boxes[idx],tick_labels_images[idx]] for idx in range(n_iters)])
tick_locations = np.concatenate([[tick_location_boxes[idx],tick_locations_images[idx]] for idx in range(n_iters)])
plt.xticks(tick_locations,tick_labels,rotation=90)
plt.legend()
plt.ylabel('Count')

if len(training_sizes) > 0:
    plt.subplot(133)
    plt.title('Active learning training curve')
    plt.grid(True)

    cmap = plt.get_map('coolwarm')
    n_all = len(training_sizes)
    for iter_id_id, (iter_id,size,mAPs) in enumerate(zip(training_iters,training_sizes,training_mAPs)):
        plt.plot(mAPs,label = 'Iter {}, auto'.format(iter_id + 1), color = cmap(iter_id_id/max(1,n_all-1)))
        plt.legend()
    plt.xlabel('Training epoch')
    plt.ylabel('Validation mAP')
plt.tight_layout()

sagemaker_client.describe_labeling_job(LabelingJobName=job_name)["LabelingJobStatus"]
OUTPUT_MANIFEST = "s3://{}/{}/output/{}/manifests/output/output.manifest".format(BUCKET,EXP_NAME,job_name)
!aws s3 cp {OUTPUT_MANIFEST} 'output.manifest'

with open("output.manifest","r") as f:
    output = [json.loads(line.strip()) for line in f.readlines()]

!aws s3 cp {S3_OUTPUT + '/annotations/worker-response'} od_output_data/worker-response --recursive --quiet

worker_file_names = glob.glob("od_output_data/worker-response/**/*.json", recursive=True)

from ground_truth_od import BoundingBox, WorkerBoundingBox, GroundTruthBox, BoxedImage

confidences = np.zeros(len(output))

keys = list(output[0].keys())
metakey = key[np.where([("-metadata" in k) for k in keys])[0][0]]
jobname = metakey[:-9]
output_images = []
consolidated_boxes = []

for datum_id, datum in enumerate(output):
    image_size = date["category"]["image_size"][0]
    box_annotations = datum["category"]["annotations"]
    uri = datum["source-ref"]
    box_confidences = datum[metakey]["objects"]
    human = int(datum[metakey]["human-annotated"] == "yes")

    image = BoxedImage(id=datum_id,size = image_size,uri=uri)
    boxes = []
    for i, annotation in enumerate(box_annotations):
        box = BoundingBox(image_id=datum_id,boxdata=annotation)
        box.confidence = box_confidences[i]["confidence"]
        box.image = image
        box.human = human
        boxes.append(box)
        consolidated_boxes.append(box)
    image.consolidated_boxes = boxes

    image.human = human
    oid_boxes_data = fids2bbs[image.oid_oi]
    gt_boxes = []
    for data in oid_boxes_data:
        gt_box = GroundTruthBox(image_id=datum_id,oiddata=data, image = image)
        gt_boxes.append(gt_box)
    image.gt_boxes = gt_boxes

    output_images.append(image)

for wfn in worker_file_names:
    image_id = int(wfn.split("/")[-2])
    image = output_images[image_id]
    with open(wfn,"r") as worker_file:
        annotation = json.load(worker_file)
        answers = annotation["answers"]
        for answer in answers:
            wid=answer["workerId"]
            wdboxes_data = answer["answerContent"]["boundingBox"]["boundingBoxes"]
            for boxdata in wboxes_data or []:
                box = WorkerBoundingBox(image_id=image_id,worker_id=wid,boxdata = boxdata)
                box.image = image
                image.worker_boxes.append(box)
human_labeled = [img for img in output_images if img.human]
auto_labeled = [img for img in output_images if not img.human]


LOCAL_IMG_DIR = '<<choose local directoryname to download the images to>>'
assert LOCAL_IMG_DIR != '<< choose a local directoryname to download the images to >>', 'Please provide a local directory name'
DATASET_SIZE = len(output_images)

image_subset = np.random.choice(output_images,DATASET_SIZE, replace = False)

for img in image_subset:
    target_fname = os.path.join(
        LOCAL_IMG_DIR, img.uri.split('/')[-1])
        if not os.path.isfile(target_fname):
            !aws s3 cp {img.uri} {target_fname}


N_SHOW = 5

human_labeled_subset = [img for img in image_subset if img.human]
auto_labeled_subset = [img for img in image_subset if not img.human]

fig, axes = plt.subplots(N_SHOW,3,figsize=(9,2*N_SHOW),facecolor = "white", dpi=100)
fig.suptitle("Human-labeled examples",fontsize=24)
axes[0,0].set_title("Worker labels". fontsize=14)
axes[0,1].set_title("consolidated label". fontsize=14)
axes[0,2].set_title("True label". fontsize=14)
for row,img in enumerate(np.random.choice(human_labeled_subset,size=N_SHOW)):
    img.download(LOCAL_IMG_DIR)
    img.plot_worker_bbs(axes[row,0])
    img.plot_consolidated_bbs(axes[row,1])
    img.plot_gt_bbs(axes[row,2])
if auto_labeled_subset:
    fig,axes = plt.subplots(N_SHOW,2,figsize(6,2*N_SHOW),facecolor= "white",dpi=100)
    fig.suptitle("Auto-labeled examples",fontsize=24)
    axes[0,0].set_title("Auto-label",fontsize=14)
    axes[0,1].set_title("True-label",fontsize=14)
    for row,img in enumerate(np.random.choice(auto_labeled_subset,size=N_SHOW)):
    img.download(LOCAL_IMG_DIR)
    img.plot_consolidated_bbs(axes[row,0])
    img.plot_gt_bbs(axes[row,1])
else:
    print("No images were auto-labeled")


N_SHOW = 10

h_img_mious = [(img,img.compute_iou_bb()) for img in human_labeled]
a_img_mious = [(img,img.compute_iou_bb() for img in autolabeled)]
h_img_mious.sort(key = lambda x: x[1], reverse =True)
a_img_mious.sort(key = lambda x: x[1], reverse =True)

h_img_confs = [(img,img.compute_img_confidence()) for img in human_labeled]
a_img_confs = [(img,img.compute_img_confidence()) for img in human_labeled]
h_img_confs.sort(key = lambda x: x[1], reverse = True)
a_img_confs.sort(key = lambda x: x[1], reverse = True)

rows_per_page = 5
column_per_page = 5
n_per_page = rows_per_page * columns_per_page

def title_page(title):
    plt.figure(figsize=(10,10),facecolor-"white",dpi=100)
    plt.text(.1,.5,s=title,fontsize=20)
    plt.axis("off")
    pdf.savefig()
    plt.close()


def page_loop(miou,axes,worker=False):
    for i,row in enumerate(axes):
        for j,ax in enumerate(row):
            img_idx = n_per_page * page + rows_per_page * i + j

            if img_idx >= min(N_SHOW,len(mious)):
                return

            img,miou = mious[img_idx]
            img.download(LOCAL_IMG_DIR)
            if worker:
                img.plot_worker_bbs(ax, img_kwargs={"aspect": "auto"},box_kwargs={"lw":0.5})
            else:
                img.plot_gt_bbs(ax,img_kwargs={"aspect":"auto"},box_kwargs={"edgecolor":"C2","lw":0.5})
                img.plot_consolidated_bbs(ax,img_kwargs = {"aspect":"auto"}, box_kwargs={"edgecolor":"C1","lw",:0.5})

mode_metrics = (("mIoU", (("Worker", h_img_mious),("Consolidated human",h_img_mious),("Auto",a_img_mious))),
            "confidence",
            (("Worker",h_img_confs), ("Consolidated human",h_img_confs),("Auto",a_img_confs)),
            )

for mode,labels_metrics in mode_metrics:
    pdfname = f"ground-truth-od-{mode}.pdf"
    with PdfPages(pdfname) as pdf:
        title_page("Images labeled by Sagemaker Ground Truth \n" f"and sorted by {mode}")
        print()
