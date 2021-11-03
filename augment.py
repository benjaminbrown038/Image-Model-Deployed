# import object for augmentation techniques
from keras.image.preprocess import ImageDataGenerator
# for cleaning data
import numpy as np
# for creating train and test folders for each class
import split_folders
import os
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
#%matplotlib inline

# need to save training and testing data so model.py can access it

# create Augment class
class Augment():
    def __init__(self,search_name):

'''
    Using ImageDataGenerator object's functionality to access training and testing images to augment images in folder
parameters:
    search_name: <string> this will be the class of the picture.
returns:
    training_data: an instance of augmented images (training) of search_name
    testing_data: an instance of augmented images (testing) of search_name
'''

        split_folders.ratio('Images', output="Augmented_Images", seed=1337, ratio=((0.8, 0.2)))

        # augmentation techniques for training data stored as an ImageDataGenerator object
        self.training = ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # augmentation techniques for testing data stored as an ImageDataGenerator object
        self.testing = ImageDataGenerator(
            scale_width = 1/255)
        # data will be in "/Images/Data/class/index.jpg"





    def training_exploration(folder_path):
        train_dir = folder_path # image folder
        # train_dir = 'DATA/train' # image folder
        # get the list of jpegs from sub image class folders
        normal_imgs = [fn for fn in os.listdir(f'{train_dir}/NORMAL') if fn.endswith('.jpeg')]
        pneumo_imgs = [fn for fn in os.listdir(f'{train_dir}/PNEUMONIA') if fn.endswith('.jpeg')]

        # randomly select 3 of each
        select_norm = np.random.choice(normal_imgs, 3, replace = False)
        select_pneu = np.random.choice(pneumo_imgs, 3, replace = False)

        # plotting 2 x 3 image matrix
        fig = plt.figure(figsize = (8,6))
        for i in range(6):
            if i < 3:
                fp = f'{train_dir}/NORMAL/{select_norm[i]}'
                label = 'NORMAL'
            else:
                fp = f'{train_dir}/PNEUMONIA/{select_pneu[i-3]}'
                label = 'PNEUMONIA'
                ax = fig.add_subplot(2, 3, i+1)

        # to plot without rescaling, remove target_size
        fn = image.load_img(fp, target_size = (100,100), color_mode='grayscale')
        plt.imshow(fn, cmap='Greys_r')
        plt.title(label)
        plt.axis('off')
        plt.show()

        # also check the number of files here
        len(normal_imgs), len(pneumo_imgs)
