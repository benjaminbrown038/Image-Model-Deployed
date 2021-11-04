'''
Libraries used in this python application:
    selenium
    requests
    time
    io
    base64
    PIL
    webdriver_manager
    os
    tensorflow
    keras
    splitfolders
    matplotlib
    numpy
'''
import time
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
import os
import PIL
import io
import base64
import webdriver_manager
from webdriver_manager import chrome
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
from tensorflow.keras.image.prepocess import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD,RMSprop, Optimizer, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
'''

'''
class Data:
'''

'''
    def __init__(self, search_name):
        self.wd = webdriver.Chrome(ChromeDriverManager.install())
        search_url = "https://www.google.com/search?q={q}&sxsrf=ALeKk02zAb9RaNNb-qSenTEJh1i2XX480w:1613489053802&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjChJqP2-7uAhVyTTABHdX0CPoQ_AUoAXoECAcQAw&biw=767&bih=841"
        self.wd.get(search_url.format(q=search_name))
'''

'''
    def scrape_and_save(self,search_name):
        body = wd.find_element(By.TAG_NAME,"body")
        for i in range(10):
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(.75)
            main = wd.find_elements(By.CLASS_NAME,"rg_i.Q4LuWd")
            links = [main[i].get_attribute('src') for i in range(len(main))]
        self.wd.quit()
        images = []
        for image in links:
            if type(image) == str:
                if image[0:4] == 'data':
                    new_image = image.replace("data:image/jpeg;base64,","")
                    if new_image[-2:] != '==':
                        new_image = new_image + '=='
                        new_image = (Image.open(io.BytesIO(base64.b64decode(new_image)))).resize(150,150)
                        if new_image.mode != 'RGB':
                            new_image.convert('RGB')
                        images.append(new_image)
                    else:
                        new_image = (Image.open(io.BytesIO(base64.b64decode(new_image)))).resize(150,150)
                        if new_image.mode != 'RGB':
                            new_image.convert('RGB')
                if image[0:4] == 'http':
                    new_image = requests.get(image)
                    new_image = (Image.open(io.BytesIO(new_image.content))).resize(150,150)
                    images.append(new_image)
        os.makedirs('Images/',exist_ok = True)
        os.makedirs('Images' + search_name , exist_ok = True)
        index = 0
        for image in images:
            image.save('Images/' + search_name + '/' + str(index) + '.jpeg')
            index += 1

'''
Run this for each class so when creating the augmented images in the Augmented_Images folder,
    it creats data for both classes.
'''
    def augment(search_name):

        splitfolders.ratio('Images', output = "Augmented_Images", seed = 1337, ratio = ((0.8,0.2)))

        training = ImageDataGenerator(rotation_range = 40,
                                            width_shift_range = 0.2,
                                            height_shift_range = 0.2,
                                            rescale = 1./255,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip=True,
                                            fill_mode = 'nearest')

        testing = ImageDataGenerator(
                                        rescale = 1./255)

    def data_exploration(search_name):
        '''
        Take samples from augmented image and compare against regular image
        '''
        norm_dir = '/Images/dogs'
        aug_dir = '/Augmented_Images/'
        # f before directory string
        normal_imgs = [fn for fn in os.listdir(f'{train_dir}') if fn.endswith('.jpeg')]
        aug_imgs = [fn for fn in os.listdir(f'{train_dir}') if fn.endswith('.jpeg')]

        select_norm = np.random.choice(normal_imgs,3,replace = False)
        select_augs = np.random.choice(aug_imgs,3,replace = False)

        fig = plt.figure(figsize=(8,6))
        for i in range(6):
            if i < 3:
                fp = f'{train_dir}/normal
                label = 'normal'
            else:
                fp = f'{train_dir}/normal/{select_norm[i]}'
                label = 'augmented'
                ax = fig.add_subplot(2,3,i+1)
        fn = image.load_img(fp,target_size = (100,100),color_mode = 'grayscale')
        plt.imshow(fn,cmap='Greys_r')
        plt.title(label)
        plt.axis('off')
        plt.show()

# creating data for model (needs multiple)
    def create_data():
        training_directory = "Data/train/"
        training_batch_size = len(os.listdir(training_directory))
        training_data = flow_from_directory(training_directory,
                                                    target_size = (150,150),
                                                    shuffle = True,
                                                    class_mode = 'binary')

        testing_directory = "Data/test/"
        testing_batch_size = len(os.listdir(testing_directory))
        testing_data = flow_from_directory(testing_directory,
                                                target_size = (150,150),
                                                shuffle = True,
                                                class_mode = 'binary' )

        x_train = training_data[0][0]
        x_train /= 255
        x_train = np.rollaxis(x_train,3,1)
        y_train = training_data[0][1]
        training = [x_train,y_train]

        x_test = testing_data[0][0]
        x_test /= 255
        x_test = np.rollaxis(x_test,3,1)
        y_test = testing_data[0][1]
        testing = [x_test,y_test]

        data = [training,testing]


    def model():
        # hyperparameter subject to change to optimize accuracy of model
        model = Sequential()
        model.add(Conv2D())
        model.add(MaxPooling2D())
        model.add(Conv2D())
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense())
        model.add(Dense())
        # hyperparameters
        opt = SGD(learning_rate = .01)
        loss = BinaryCrossentropy()
        ac = Accuracy()

        model.compile(loss = loss, optimizer = opt, accuracy = ac)

        model.fit(data[0][0],
                data[0][1],
                validation_data = (data[1][0],data[1][1]),
                epochs = 25,
                verbose = 2)





        if __name__ == "__main__":
