'''
Importing libraries that are needed for this project:
        time
        selenium
        requests
        io
        os
        PIL
        base64
        webdriver_manager
        numpy
        keras
        matplotlib
        splitfolders
'''
import time
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
import os
import PIL
from PIL import Image
import io
import base64
import webdriver_manager
from webdriver_manager import chrome
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import splitfolders
import matplotlib.pyplot as plt
'''
Collecting image data of cats and dogs from google
    for image classification
'''

class Data:

'''
~ Comments are for why the code is there~
Using Selenium within this script to collect image data for image classification model.
    parameters:
        search_name: <string> search_name will be search query
        length: <integer> (out of 100) what percentage of the results to scrape (1 will be 40 images, 100 will be __ images)
'''
    def __init__(self,search_name, length):
        self.wd = webdriver.Chrome(ChromeDriverManager.install())
        search_url = "https://www.google.com/search?q={}&sxsrf=ALeKk02zAb9RaNNb-qSenTEJh1i2XX480w:1613489053802&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjChJqP2-7uAhVyTTABHdX0CPoQ_AUoAXoECAcQAw&biw=767&bih=841"
        self.wd.get(search_url.format(search_name))

    def scrape_and_save(self,search_name):
        body = self.wd.find_element(By.TAG_NAME,"body")
        for i in range(length):
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(.75)
            main = wd.find_elements(By.CLASS_NAME,"rg_i.Q4LuWd")
            links = [main[i].get_attribute('src') for i in range(len(main))]
            self.wd.quit()
            images=[]
            for image in links:
'''
Data conditions:    1. string
                        - If it has 'data' in front
                            - remove noise
                            - If it has byte format at end '=='
                                - If image is in RGB mode
                            - If it isn't in byte format
                        - If it has 'http' in front
                            - convert to rgb
'''
                if type(image) == str:
                    if image[0:4] == 'data':
                        new_image = image.replace("data:image.jpeg;base64,","")
                        if new_image[-2:] != '==':
                            new_image = new_image + '=='
                            new_image = (Image.open(io.BytesIO(base64.b64decode(new_image)))).resize(150,150)
                            if new_image.mode != 'RGB':
                                new_image = new_image.convert('RGB')
                                images.append(new_images)
                        else:
                            new_image = (Image.open(base64.b64decode(io.BytesIO(new_image)))).resize(150,150)
                            if new_image.mode != 'RGB':
                                new_image = new_image.convert('RGB')
                                images.append(new_images)
                    if image[0:4]  == 'http':
                        new_image = requests.get(new_image)
                        new_image = (Image.open(io.BytesIO(new_image.content))).resize(150,150)
                        if new_image.mode != 'RGB':
                            new_image = new_image.convert('RGB')
                            images.append(new_images)
            index = 0
            os.makedirs('Images', exists_ok= True)
            os.makedirs('Images/training', exists_ok= True)
            os.makedirs('Images/training/' + search_name, exists_ok = True)
            for image in images:
                image.save('Images/training' + search_name + '/' + str(index) + '.jpeg')
                index += 1
            split_folders.ratio('Images', output="Augmented_Images", seed=1337, ratio=((0.8, 0.2)))
            training_file = 'Augmented_Images/train/'
            validation_file = 'Augmented_Images/val/'
            files = [training_file,validation_file]

class Augment:

    def __init__(self):

        self.training = ImageDataGenerator(rotation_range = 40,
                                            width_shift_range = 0.2,
                                            height_shift_range = 0.2,
                                            rescale = 1./255,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip = True,
                                            fill_mode = 'nearest')

        self.validation = ImageDataGenerator(rescale = 1./ 255)

        obj = [self.training,self.validation]
        return data

    def augmentation(self,obj,files):

        training = obj[0]
        validation = obj[1]

        training_folder = files[0]
        validation_folder = files[1]

        training_folder_size = len(os.listdir(training_folder))
        validation_folder_size = len(os.listdir(validation_folder))

        training_data = training.flow_from_directory(training_folder,
                                    batch_size = training_folder_size,
                                    target_size = (150,150),
                                    shuffle = True,
                                    class_mode = 'binary')

        validation_data = validation.flow_from_directory(validation_folder,
                                        batch_size = validation_folder_size,
                                        target_size = (150,150),
                                        shuffle = True,
                                        class_mode = 'binary')

        data = [training_data,validation_data]
        return data

    def model():
