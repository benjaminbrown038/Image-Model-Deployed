# pause after scrolling results page 
import time 
# for accessing web 
import selenium 
# for chromedriver
from selenium import webdriver
# to navigate results page 
from selenium.webdriver.common.keys import Keys
# to retreive http data
import requests
# checking and making directory
import os
# open image after decoded
import PIL
# open image after decoded
from PIL import Image
# decoding 
import io
# decoding 
import base64 

# create class called Data
class Data():    
    
    # initializer for class with input search name 
    def __init__(self,search_name):
        # specific to user Desktop where chrome driver is downloaded
        DRIVER_PATH = '/home/trey/Sharpest-Minds-Project/chromedriver'
        self.wd = webdriver.Chrome(DRIVER_PATH)
        # specific for requesting images
        search_url = "https://www.google.com/search?q={q}&sxsrf=ALeKk02zAb9RaNNb-qSenTEJh1i2XX480w:1613489053802&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjChJqP2-7uAhVyTTABHdX0CPoQ_AUoAXoECAcQAw&biw=767&bih=841"
        self.wd.get(search_url.format(q=search_name))
    
    # get images links, remove noise, decode cleaned data, size and open image, save image, create folder, save images from search to folder 
    def scrape_and_save(self,search_name):
        
        body = self.wd.find_element_by_tag_name("body")
        # scrolling search page results
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(2.5)
        # list of classes
        main = self.wd.find_elements_by_class_name("rg_i.Q4LuWd")
        # getting image links (ASCII data communication) stored as base64 and http urls from img (html) tag with src holding the path (url)
        links = [main[i].get_attribute('src') for i in range(len(main))]
        # shut down web page 
        self.wd.quit()
        images = []
        #
        for image in links:
            # going through image links which stored as strings
            if type(image) == str:
                # checking base64 text 
                if image[0:4] == 'data':
                    # remove noise (cleaning data)
                    new = image.replace("data:image/jpeg;base64,","")
                    # adding equals at the end for decoding
                    if new[-2:] != '==':
                        new_edit = new + '=='
                        # image becomes Image object
                        new_image = (Image.open(io.BytesIO(base64.b64decode(new_edit)))).resize((150,150))
                        # append image to list
                        images.append(new_image)
                    else:
                        # open image as Image object
                        new_image = (Image.open(io.BytesIO(base64.b64decode(new)))).resize((150,150))
                        # append image to list 
                        images.append(new_image)
                if image[0:4] == 'http':
                    # http url from web results 
                    new = requests.get(image)
                    # decode and create image as Image object
                    new_image = Image.open(io.BytesIO(new.content))
                    # save to list 
                    images.append(new_image)
        # creating directories for training and testing 
        os.makedirs('Images/', exist_ok=True)
        os.makedirs('Images/training', exist_ok=True)
        os.makedirs('Images/testing', exist_ok=True)
        index = 0 
        # iterating through list where images are saved and saving images as jpeg to just created directories
        for i in images:
            i.save('Images/training'+'/'+ search_name + str(index) +'.jpeg')
            index += 1 
