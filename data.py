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

class Data():    
    
    def __init__(self,search_name):
        DRIVER_PATH = '/home/trey/Sharpest-Minds-Project/chromedriver'
        self.wd = webdriver.Chrome(DRIVER_PATH)
        search_url = "https://www.google.com/search?q={q}&sxsrf=ALeKk02zAb9RaNNb-qSenTEJh1i2XX480w:1613489053802&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjChJqP2-7uAhVyTTABHdX0CPoQ_AUoAXoECAcQAw&biw=767&bih=841"
        self.wd.get(search_url.format(q=search_name))
  
    def scrape_and_save(self,search_name):
        body = self.wd.find_element_by_tag_name("body")
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(2.5)
        main = self.wd.find_elements_by_class_name("rg_i.Q4LuWd")
        links = [main[i].get_attribute('src') for i in range(len(main))]
        self.wd.quit()
        images = []
        for image in links:
            if type(image) == str:
                # checking conditional text so Image library can open based on based64 conversion
                if image[0:4] == 'data':
                    # replace text so we can open it
                    new = image.replace("data:image/jpeg;base64,","")
                    # adding equals at the end for decoding
                    if new[-2:] != '==':
                        new_edit = new + '=='
                        # image becomes Image object
                        new_image = (Image.open(io.BytesIO(base64.b64decode(new_edit)))).resize((150,150))
                        images.append(new_image)
                    else:
                        new_image = (Image.open(io.BytesIO(base64.b64decode(new)))).resize((150,150))
                        images.append(new_image)
                if image[0:4] == 'http':
                    new = requests.get(image)
                    new_image = Image.open(io.BytesIO(new.content))
                    images.append(new_image)
        # creating directories, saving images, and creating names for images as jpeg files
        os.makedirs('Images/', exist_ok=True)
        os.makedirs('Images/training', exist_ok=True)
        os.makedirs('Images/testing', exist_ok=True)
        index = 0 
        for i in images:
            i.save('Images/training'+'/'+ search_name + str(index) +'.jpeg')
            index += 1 
