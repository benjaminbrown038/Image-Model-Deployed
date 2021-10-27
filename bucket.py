import boto3
import os
import pandas as pd
'''
    Using Amazon's API via sagemaker and boto3 to create a bucket and add content
parameters:
    search_name: <string> class name of image
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
creating Session for bucket
'''
    session = self.boto3.Session(aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key)
'''
Adding files to bucket
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
