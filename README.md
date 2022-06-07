# Sharpest Minds Project

Goal:

The ```main.py``` file:

1. Scrape images of cats
2. Scrape images of dogs
3. Extract faces from Images
4. Save images in folder specific to ```cat``` or ```dog``` inside ```Images``` folder

The ```data.py``` file:

1. Create augmented images
2. save 80% of augmented images to train folder
3. save 20% of augmented images to test folder

Set up:

1. If not already installed:
https://www.anaconda.com/products/individual
2. Create the virtual environment.
Run: ```conda create -n envname python=3.8 anaconda ```
3. Clone
Run: ``` git clone https://www.github.com/benjaminbrown038/Sharpest-Minds-Project.git```
4. Activate:
```conda activate envname```

Notes:

  - Must install chrome driver in order to run selenium with chrome webdriver.
  - Chromedrivers can be found at https://chromedriver.chromium.org/downloads.

References:

  - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html.

  https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
  - MaxPool2D application tutorial: https://www.tutorialspoint.com/how-to-apply-a-2d-max-pooling-in-pytorch
  - Fitting a model in PyTorch: https://andrewpwheeler.com/2021/05/24/fitting-a-pytorch-model/
https://www.analyticsvidhya.com/blog/2021/09/convolutional-neural-network-pytorch-implementation-on-cifar10-dataset/

Future work:

- Integrating files to depend on each other
- Deploying model and data to cloud using AWS and lambda function (Find a couple of reliable tutorials)
- Running application from terminal using ```python data.py``` while passing classes as parameters
- Project requires downloading AWS access and secret key in a .csv file and saved in the project directory


## Deployment (under construction)

Objectives for the deployment of the model are as follow:
- Make the API interface with FastAPI
   - Input: an image file the user puts in manually
   - Output: a JSON with classification scores for each class

- Write a predictor class that does the following:
   1. Downloads the model and interacts with the config.yaml file to do this (so you will have to write PATHES!)
   2. Takes the image that is uploaded and does some array building
   3. Classifies the image
- Create a main.py or app.py for the main API
- The main or app script for the API runs on `univcorn main:app --reload`
- EXTRA: be able to send a cURL POST request to an endpoint and receive an answer

- Docker: create a Dockerfile that has instructions about creating the environment. This should include:
  - Python runtime
  - Copies the main script with your model
  - Copies the app config.yaml
  - Runs a update for git 
  - Runs an install for libglib2.0-0
  - Runs a pip install for the github repo to be on the Docker image
  - Exposes the port for Docker
  - Define the entrypoint for the host and the port to the FastAPI

  You should be able to run:
  `docker build -t <NAME OF IMAGE> .` and get your model response when you map to
  `docker run -p <PORT NUMBER>:<PORT NUMBER> <NAME OF IMAGE> .`