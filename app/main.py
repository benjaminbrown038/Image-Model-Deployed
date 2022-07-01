from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile 

# Can fill in here or migrate other files

# Check out the tutorial on FastAPI for details: https://fastapi.tiangolo.com/tutorial/

# Ideally, we want 1 route in our API (maybe call it /predict/ or something)
# We also want to create a class that defines our input and the data types that come with that class. 
# The rest is really up to you. But these are good practices to have when creating an API

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

try: 
    # passing file 
except:
    # unless not jpeg

@app.post("/uploadfile/")
def create_upload_file(file: UploadFile):
    return {"filename": file.filename}    



'''
class User(BaseModel):
    id: int
    name = 'John Doe'
    signup_ts: Optional[datetime] = None
    friends: List[int] = []

# dictionairy item with user credentials 

external_data = 

{
    'id': '123',
    'signup_ts': '2019-06-01 12:22',
    'friends': [1, 2, '3'],
}

# Defining user object using dictionairy data

user = User(**external_data)


# print user data 

print(user.dict())

"""
{
    'id': 123,
    'signup_ts': datetime.datetime(2019, 6, 1, 12, 22),
    'friends': [1, 2, 3],
    'name': 'John Doe',
}
"""
'''
