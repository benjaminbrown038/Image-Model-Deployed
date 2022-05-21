import s3
import sagemaker
import os 
import json 
import boto3
import sagemaker 
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role,Session

with open("code/config.json") as f:
    CONFIG = json.load(f)
    
sess=Session()

role = get_execution_role()

%store -r pt_mnist_model_data

try:
    pt_mnist_model_data
except NameError:
    import json
    
    s3 = boto3.client(s3)
    bucket = CONFIG["public_bucket"]
    key = "datasets/image/MNIST/model/pytorch-training-2020-22-21-22-02-56-203/model.tar.gz"
    s3.download_file(bucket,key,"model.tar.gz")
    pt_mnist_model_data = sess.upload_data(path = "model.tar.gz",bucket = sess.default_bucket(),
                                           key_prefix="model/pytorch")
    

    
    model = PyTorchModel(entry_point="inference.py",
                         source_dir = "code",
                         role=role
                         model_data=pt_mnist_model_data,
                         framework_version="1.5.0",
                         py_version="py3")
    
    def model_fn(model_dir):
        model = Net()
        with open(os.path.join(model_dir,"model.pth"),"rb") as f:
            model.load_state_dict(torch.load(f))
        model.to(device).eval()
        return model 
      
    def input_fn(request_body,request_content_type):
        assert request_content_type == 'application/json'
        data = json.loads(request_body)['inputs']
        data = torch.tensor(data,dtype=torch.float32,device=device)
        return data 
      
    def predict_fn(input_object,model):
        with torch.no_grad():
            prediction = model(input_object)
        return prediction
      
    def output_fn(predictions,content_type):
        assert content_type == 'application/json'
        res = predictions.cpu().numpy().tolist()
        return json.dumps(res)
      
      
  from sagemaker.serializers import JSONSerializer 
  from sagemaker.deserializers import JSONDeserializer
  
  local_mode = False 
  
  if local_mode:
      instance_type = "local"
  else:
    instance_type = "ml.c4.xlarge"
    
 predictor = model.deploy(initial_instance_count = 1, 
                          instance_type = instance_type, 
                          serializer = JSONSerializer(),
                          deserializer = JSONDeserializer())
import random
import numpy as np
dummy_data = {"inputs":np.random.rand(16,1,28,28).tolist()}
res = predictor.predict(dummy_data)

print("Predictions:",res)
dummy_data = [random.random() for _ in range(784)]

from utils.mnist import mnist_to_numpy, normalize
import random
import matplotlib.pyplot as plt
%matplotlib inline

data_dir = "tmp/data"
X,Y = mnist_to_numpy(data_dir, train = False)
mask = random.sample(range(X.shape[0]),16)
sample = X[mask]
labels = Y[mask]

fig,axs = plt.subplot(nrows=1,ncols=16,figsize=(16,1))

for i, splt in enumerate(axs):
    splt.imshow(samples[i])
  
print(samples.shape,samples.dtype)

samples = normalize(samples.astype(np.float32), axis(1,2))

res = predictor.predict({"inputs": np.expand_dims(samples,axis = 1).tolist()})

predictions = np.argmax(np.array(res,dytpe=np.float32),axis=1).tolist()
print("Predicted digits: ", predicitons)

!pygmentize code/test_inference.py

import os 
if not local_mode:
    predictor.delete_endpoint()
else:
    os.system("docker container ls | grep 8080 | awk '{print $1}' | xargs docker container rm -f")
    
