# Sharpest Minds Project

Goal:
```main.py```
1. Collect images of cats and dogs
2. extract faces from Images
```data.py```
1. Create augmented images
2. save 80% of augmented images to train folder
3. save 20% of augmented images to test folder


Set up:
1. Install anaconda using this link: https://www.anaconda.com/products/individual.
2. Create the virtual environment with anaconda and the version of python that was downloaded. The ```conda create -n envname python=3.8 anaconda ``` command will accomplish this, with 'envname' being anything of the user's choosing.
3. Use ``` git clone``` command to clone the project locally.
4. Activate it using ```conda activate envname```

```main.py```
This file will gather images from Google of the user's choosing.
Next, removing unnecessary information from the string retrieved from Google and decoding the string retrieved from Google search.
It will then create training and testing folders and store the final result (.jpeg) in the folder.

Must install chrome driver in order to run selenium with chrome webdriver.
Chromedrivers can be found at https://chromedriver.chromium.org/downloads. Pick the version that supports your chrome browser version.
The steps to know your specific browser version: https://help.zenplanner.com/hc/en-us/articles/204253654-How-to-Find-Your-Internet-Browser-Version-Number-Google-Chrome.

```data.py```
This file uses ```ImageDataGenerator``` function from keras for augmentation techniques.

```model.py```
Holds the architecture for the cnn.
The model is trained on the gathered data, cleaned, and augmented data within the folders.


References:
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html.

Future work:
- Integrating files to depend on each other
- Deploying model and data to cloud using AWS and lambda function (Find a couple of reliable tutorials)
- Running application from terminal using ```python data.py``` while passing classes as parameters
- Project requires downloading AWS access and secret key in a .csv file and saved in the project directory
