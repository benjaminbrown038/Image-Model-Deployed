# import object for augmentation techniques
from keras.image.preprocess import ImageDataGenerator
# for cleaning data 
import numpy as np
# for creating train and test folders for each class
import split_folders

# need to save training and testing data so model.py can access it   

# create Augment class 
class Augment():

    def __init__(self,search_name):
        split_folders.ratio('Images', output="Data", seed=1337, ratio=((.8, 0.2)))
        
        # augmentation techniques for training data stored as an ImageDataGenerator object
        self.training_data = ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        
        # augmentation techniques for testing data stored as an ImageDataGenerator object
        self.testing_data = ImageDataGenerator(
            scale_width = 1/255)
        
    def aug(self,search_name):        
        # using ImageDataGenerator object's functionality to access training images and augment images in folder 
        # .flow_from_directory function in ImageDataGenerator class to augment, set image size, batch_size, classification type
        # creating training data from .flow_from_directory
        # grabbing images from training folder of each class and augmenting
        self.training_data = train_datagen.flow_from_directory('/Images/Data/train' + search_name,
                                                          target_size = (150,150),
                                                          batch_size = 32,
                                                          class_mode = 'binary')
        
        # using ImageDataGenerator object's functionality to access training images and augment images in folder 
        # .flow_from_directory function in ImageDataGenerator class to augment, set image size, batch_size, classification type
        # creating training data from .flow_from_directory    
        # grabbing images from testing folder of each class and augmenting
        self.testing_data = test_datagen.flow_from_directory('/Images/Data/test' + search_name,
                                                        target_size = (150,150),
                                                        batch_size=32,
                                                        class_mode = 'binary')
        
        # x training data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
        x_train = self.training_data[0][0]
        # training data split from 0 to 1 for activation functions
        x_train /= 255
        # change shape (add axis) to x training for model 
        x_train = np.rollaxis(self.x_train,3,1)
        # y training data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
        y_train = self.training_data[0][1]
        
        # x testing data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
        x_test = self.testing_data[0][0]
        # scale values in tx testing from 0 to 1 for activation functions in model 
        x_test /= 255
        # change shape (add axis) to x testing data for model 
        x_test = np.rollaxis(self.x_test,3,1)
        # y testing data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
        y_test = self.testing_data[0][1]
