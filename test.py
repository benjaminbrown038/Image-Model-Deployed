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
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from tensorflow.keras.image.preprocess import ImageDataGenerator
import numpy as np
import splitfolers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD, RMSProp, Optimizer, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy

class Data:

    def __init__(self,search_name):
        self.wd = webdriver.Chrome(ChromeDriverManager.install())
        search_url = "https://www.google.com/search?q={q}&sxsrf=ALeKk02zAb9RaNNb-qSenTEJh1i2XX480w:1613489053802&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjChJqP2-7uAhVyTTABHdX0CPoQ_AUoAXoECAcQAw&biw=767&bih=841"
        self.wd.get(search_url.format(q=search_name))

    def scrape_and_save(self,search_name):
        body = self.wd.find_element(By.TAG_NAME,"body")
        for i in range(10):
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(.75)
            main = self.wd.find_elements(By.CLASS_NAME,"rg_i.Q4LuWd")
            links = [main[i].get_attribute('src') for i in range(len(main))]
        self.wd.quit()
        images=[]
        for image in links:
            if type(image) == str:
                if image[0:4] == 'data':
                    new_image = image.replace("data:image/jpeg;base64,","")
                    if new_image[-2:] != '==':
                        new_image = new_image + '=='
                        new_image = (Image.open(io.BytesIO(base64.b64decode(new_image)))).resize(150,150)
                        if new_image.mode != 'RGB':
                            new_image = new_image.convert('RGB')
                else:
                    new_image = (Image.open(io.Bytes(base64.b64decode(new_image)))).resize(150,150)
                    if new_image.mode != 'RGB':
                        new_image = new_image.convert('RGB')
                    images.append(new_image)

        os.makedirs('Images/',exist_ok=True)
        os.makedirs('Images/'+search_name,exist_ok=True)
        index = 0
        for image in images:
            image.save('Images/' + search_name + '/' + str(index) + '.jpeg')
            index += 1
        if __name__ == __main__:
            # do something (run functions)
class Augment:

    def __init__(self,search_name):

        splitfolers.ratio('Images',output='Augmented_Images',seed=1337,ratio = ((0.8 , 0.2)))
        self.training = ImageDataGenerator(rotation_range=40,
                                            width_shift_range = 0.2,
                                            height_shift_range = 0.2,
                                            rescale = 1./255,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip = True,
                                            fill_mode = 'nearest')

        self.testing = ImageDataGenerator(scale_width = 1/255)

    # check folder names
    def training_exploration(self,search_name):
        # training_dir = 'DATA/train'
        normal_images = [fn for fn in os.listdir(f'{train_dir}/Images/' + search_name) if fn.endswith('.jpeg')]
        aug_images = [fn for fn in os.listdir(f'{train_dir}/Augmented_Images/' + search_name) if fn.endswith('.jpeg'))]
        select_norm = np.random.choice(normal_images, 3 , replace = False)
        select_aug = np.random.choice(aug_images,3,replace = False)

        fig = plt.figure(figsize=(8,6))
        for i in range(6):
            if i < 3:
                fp = f'{train_dir}/Images/{select_norm[i]}'
                label = 'normal'
            else:
                fp = f'{train_dir}/Augment_Images/{select_aug[i - 3]}'
                label = 'augmented'
                ax = fig.add_subplot(2,3,i+1)


        fn = image.load_img(fp,target_size = (100,100), color_mode = 'grayscale')
        plt.imshow(fn,cmap = 'Greys_r')
        plt.title(label)
        plt.axis('off')
        plt.show()

        search_name = 'dogs'

    # eventual extend this to multiple classes functionality
    def binary_class_augment(*classes)
        training_directory = 'Data/train/dogs'
        training_batch_size = len(os.listdir(training_dir))
        training_data = flow_from_directory('Data/train/dogs',
                                            target_size = (150,150),
                                            batch_size = training_batch_size,
                                            shuffle = False,
                                            class_mode = 'binary')
        x_train = training_data[0][0]
        x_train /= 255
        x_train = np.rollaxis(x_train,3,1)
        y_train = training_data[0][1]

        x_test = testing_data[0][0]
        x_test /= 255
        x_test = np.rollaxis(x_test,3,1)
        y_test = testing_data[0][1]
        training = [x_train,y_train]
        testing = [x_test,y_test]
        data = [training,testing]


    def model(self, data):

        model = Sequential()
        model.add(Conv2D())
        model.add(MaxPooling2D())
        model.add(Conv2D())
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense())
        model.add(Dense())
        opt = SGD(learning_rate=.01)
        loss = BinaryCrossentropy()
        ac = Accuracy()
        model.compile(optimizer = opt, loss = loss, accuracy = ac)
        model.fit(x_train = data[0][0], y_train = data[0][1],
                    x_test = data[1][0], y_test = data[1][1],
                    epochs = 25,
                    verbose = 2)

        # .fit()
        # https://keras.rstudio.com/reference/fit.html
