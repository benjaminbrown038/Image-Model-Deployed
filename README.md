# Sharpest Minds Project

This is a personal project that I have been working on to integrate some data collection other than traditional datasets provided by machine learning libraries.

The ```data.py``` file will gather images from Google of the user's choosing. Next, removing unnecessary information from the string retreived from Google and decoding the string retreived from Google search. It will then create training and testing folders and store the final result (.jpeg) in the folder. Must install chrome driver in order to run selenium with chrome webdriver.

Chromedrivers can be found at https://chromedriver.chromium.org/downloads. Pick the version that supports your chrome browser version.

The steps to know your specific browser version:
https://help.zenplanner.com/hc/en-us/articles/204253654-How-to-Find-Your-Internet-Browser-Version-Number-Google-Chrome.

The ```augment.py``` file uses keras library and ```ImageDataGenerator``` function to create an object that will have augmented techniques. This object can be passed to another important function ```flow_from_directory``` which grabs all files from the designated folder. The training and testing data is then resaved as augmented images.

This is a power tutorial for this augmentation technique by Francois Chollet: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html.
This is the tutorial I used for relevant information and steps for my application.


The ```model.py``` folder holds the model architecture for the CNN. Keras is used for the Sequential model along with imported functions Conv2d, MaxPool2D, and Flatten layers to create the model for prediction.

The ```bucket.py``` folder requires the user to navigate to the aws website and create a login. Then using the login, create an access key and a secret key. The website will prompt a .csv to pop up with these values stored. Open the file and change format of values to be in rows and column. Please see this tutorial:https://realpython.com/python-csv/. So the values are stored using commas as:
column1name(access_key_variable),column2name(secret_key_variable)
row1name(access_key_value),row2name(secret_key_value)


Future work:
- Deploying model and data to cloud using AWS and lambda function
- Running application from terminal using ```python data.py``` while passing classes as parameters
- Directions for implementing virtual environment for this application
- Project requires downloading AWS access and secret key in a .csv file and saved in the project directory
