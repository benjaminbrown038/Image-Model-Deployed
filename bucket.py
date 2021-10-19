import boto3

os.makedirs('/content/Images', exist_ok=True)
os.makedirs('/content/Images/training', exist_ok=True)
#Creating Session With Boto3.
session = boto3.Session(aws_access_key_id='AKIAW3L5PY3OAOBXAHOY',
aws_secret_access_key='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')

import os
directory = '/content/Images/training'
for filename in os.listdir(directory):
  object = s3.Object('sharpest-minds-bucket123-ben', filename)
  result = object.put(Body=txt_data)