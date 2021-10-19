# Sharpest Minds Project

This is a personal project that I have been working on to integrate some data collection other than traditional datasets provided by machine learning libraries. The ```data.py``` file will gather images from Google of the user's choosing. Next, removing unnecessary information from the string retreived from Google and decoding the string retreived from Google search. It will then create training and testing folders and store the final result (.jpeg) in the folder. Must install chrome driver in order to run selenium with chrome webdriver. Chromedrivers can be found at https://chromedriver.chromium.org/downloads. Pick the version that supports your chrome browser version. The steps to know your specific browser version: https://help.zenplanner.com/hc/en-us/articles/204253654-How-to-Find-Your-Internet-Browser-Version-Number-Google-Chrome.


- Data file contains code that accesses the web to collect along with saving these images

- Augment file contains code that will access this file to augment, with keras, the files in the folder

- Model file will get these images and train my model with them


Future work: 

- Deploying model to cloud using AWS and lambda function
