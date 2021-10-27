'''
how to document imports
'''
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
import webdriver_manager
from webdriver_manager import chrome
from webdriver_manager.chrome import ChromeDriverManager
# create class called Data
class Data():
'''
    Using Selenium chromedriver in local directory will fetch google images based on parameter search_name
parameters:
    search_name: <string> will be the search query for google images
returns:
'''
    # initializer for class with input search name
    def __init__(self,search_name):
        # specific to user Desktop where chrome driver is downloaded
        self.wd = webdriver.Chrome(ChromeDriverManager().install())
        # specific for requesting images
        search_url = "https://www.google.com/search?q={q}&sxsrf=ALeKk02zAb9RaNNb-qSenTEJh1i2XX480w:1613489053802&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjChJqP2-7uAhVyTTABHdX0CPoQ_AUoAXoECAcQAw&biw=767&bih=841"
        self.wd.get(search_url.format(q=search_name))

'''
    Using webpage results of google images based on search_name, the webpage is scraped for the image data and saved locally
parameters:
    search_name: <string> will be used for the creation of folders and naming and saving of files within that folder
returns:
'''

    # get images links, remove noise, decode cleaned data, size and open image, save image, create folder, save images from search to folder
    def scrape_and_save(self,search_name):

        body = wd.find_element(By.TAG_NAME,"body")
        # scrolling search page results
        for i in range(60):
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(.3)
        # list of classes
        main = wd.find_elements(By.CLASS_NAME,"rg_i.Q4LuWd")
        # getting image links (ASCII data communication) stored as base64 and http urls from img (html) tag with src holding the path (url)
        #print(type(main))
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
        # data is saved to the folder that is created according to each instance of Data, exist_ok True for multiple instances of Data
        os.makedirs('Images/', exist_ok=True)
        # exists_ok True if same search_name is ran more than once, the folder will stay
        os.makedirs('Images/'+ search_name, exist_ok=True)
        index = 0
        # iterating through list where images are saved and saving images as jpeg to just created directories
        #split_folders.ratio('Images', output="output", seed=1337, ratio=((.8, 0.2)))
        index = 0
        for i in images:
            # saving images according to search_name in search_name directory
            i.save('Images/'+ search_name + '/' + str(index) + '.jpeg')
            index += 1

    if __name__ == "__main__":
    
