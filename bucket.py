import boto3
import os
import pandas as pd

'''
    Using Amazon's API via sagemaker and boto3 to create a bucket and add content
parameters: search_name: <string> class name of image
returns: end point for model and data
'''

def __init__(self,search_name):
    # run all file from same directory
    # there for directory where images will be will  be same
    directory = "/Images/Data/train/" + search_name + "/"
    #Creating Session With Boto3
    secret = pd.read_csv('rootkey.csv')
    aws_access_key_id = secret['AWSAccessKeyId']
    aws_secret_access_key = secret['AWSSecretKey']

'''
Creating Session for bucket:
    Access Key
    Secret Key
'''

    session = self.boto3.Session(aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key)

'''
Adding files to bucket:
    Bucket was created on AWS main page
    Open image in file
    Convert image into bytes
    Add data into bucket (in byte format)
'''

    for filename in os.listdir(directory):
        object = s3.Object('sharpest-minds-bucket123-ben', filename)
        #  open image as image,
        img = Image.open(filename)
        # open jpeg image as bytes
        f = io.BytesIO()
        img.save(f,format = "jpeg")
        # put takes in bytes
        image = img.get_value()
        result = object.put(image)
'''
creating client for
'''
# client

runtime = boto3.client('runtime.sagemaker')
response = runtime.invoke_endpoint(EndpointName=linear_endpoint,# LAMBDA EVENT
                                   ContentType='text/csv',
                                   Body=payload)


















import os
import boto3
import re
from sagemaker import get_execution_role
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import json
import sagemaker.amazon.common as smac
from sagemaker.amazon.amazon_estimator import get_image_uri


role = get_execution_role()
bucket_name = 'Your_S3_Bucket'
data_key = 'Your_dataset.csv'
data_location = 's3://{}/{}/'.format(bucket,data_key)

data = pd.read_csv(data_location)
# to csv
data.to_csv("data.csv",sep = ',', index = False)
print(data.shape)
display(data.head())
# .describe()
display(data.describe())
# .rand
rand_split = np.random.rand(len(data))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
test_list = rand_split >= 0.9

data_train = data[train_list]
data_val = data[val_list]
data_test = data[test_list]
# iloc
train_y = ((data_train.iloc[:,6])).as_matrix()
train_x = ((data_train.iloc[:,1:6])).as_matrix()
train_x = train_x.astype('float32')
train_y = train_y.astype('float32')
val_x = ((data_val.iloc[:,6])).as_matrix()
val_y = ((data_val.iloc[:,1:6])).as_matrix()
val_x = val_x.astype('float32')
val_y = val_y.astype('float32')
test_y = ((data_test.iloc[:6])).as_matrix()
test_x = data_test.iloc[:,1:6].as_matrix()
test_x = test_x.astype('float32')
test_y = test_y.astype('float32')
# smac or sm
smac.write_numpy_to_dense_tensor(f,train_x,train_y)
smac.write_numpy_to_dense_tensor(f,val_x,val_y)
smac.write_numpy_to_dense_tensor(f,test_x,test_y)
train_file = 'linear.data'
validation_file = 'validation.data'

f = io.BytesIO()
f.seek(0)
# .Object()
train_obj = os.path.join(prefix,'train',train_file)
val_obj = os.path.join(prefix,'validation',validation_file)
# lets do it
sess = boto3.Session()
res = sess.resource('s3')
buck = res.Bucket(bucket_name)
obje = buck.Object(obj)
region = sess.region_name
# get image uri
container = get_image_uri(region,'linear-learner')




































)
