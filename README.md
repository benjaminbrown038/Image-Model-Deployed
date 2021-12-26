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

  - Install anaconda using this link: https://www.anaconda.com/products/individual


2. Create the virtual environment.

  - ```conda create -n envname python=3.8 anaconda ```


3. ``` git clone https://www.github.com/benjaminbrown038/Sharpest-Minds-Project.git```


4. Activate it using ```conda activate envname```


Notes:


  - Must install chrome driver in order to run selenium with chrome webdriver.

  - Chromedrivers can be found at https://chromedriver.chromium.org/downloads.


References:

  - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html.

Future work:
- Integrating files to depend on each other
- Deploying model and data to cloud using AWS and lambda function (Find a couple of reliable tutorials)
- Running application from terminal using ```python data.py``` while passing classes as parameters
- Project requires downloading AWS access and secret key in a .csv file and saved in the project directory
