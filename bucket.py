import os
import numpy as np
import pandas as pd
import io
import json
import matplotlib.pyplot as plt
import time

import boto3
from sagemaker import get_execution_role
import sagemaker.amazon.common as smac
import re
from sagemaker.amazon.amazon_estimator import get_image_uri


role = get_execution_role()

'''

Credentials for access to AWS through python

'''

bucket_name = 'S3 bucket name'
data_key = 'Your_dataset.csv'
# dATA IN s3 bucket as csv file
data_location = 's3://{}/{}'.format(bucket,data_key)
# read data from s3 bucket turn to pandas dataframe
data = pd.read_csv(data_location)
data.to_csv("data.csv",sep=',',index=False)
print(data.shape)
# checking data of data frame
display(data.head())
# statistics of data
display(data.describe())
# using numpy to
rand_split = np.random.rand(len(data))
train_list = rand_split < 0.8

val_list = (rand_split >= 0.8) & (rand_split < 0.9)
test_list = rand_split >= 0.9

data_train = data[train_list]
data_val = data[val_list]
data_test = data[test_list]

train_y = ((data_train.iloc[:,6])).as_matrix()
train_x = (data_train.iloc[:,1:6]).as_matrix()
train_x = train_x.astype('float32')
train_y = train_y.astype('float32')

val_x = (data_val.iloc[:,6]).as_matrix()
val_y = (data_val.iloc[:,1:6]).as_matrix()
val_x = val_x.astype('float32')
val_y = val_y.astype('float32')

test_y = (data_test.iloc[:,6]).as_matrix()
test_x = (data_test.iloc[:,1:6]).as_matrix()
test_x = test_x.astype('float32')
test_y = test_y.astype('float32')

f = BytesIO()
f.seek(0)

sesh = boto3.Session()
res = sesh.resource('s3')
buck = res.Bucket(bucket_name)
validation_file = os.path.join(prefix,'validation', validation_file)
train_file = os.path.join(prefix,'train',train_file)
val_obj = (buck.Object(validation_file)).upload_fileobj(f)
train_obj = (buck.Object(train_file)).upload_fileobj(f)
