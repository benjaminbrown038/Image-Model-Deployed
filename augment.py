# import object for augmentation techniques
from keras.image.preprocess import ImageDataGenerator
# for cleaning data 
import numpy as np

# need to save training and testing data so model.py can access it   

# create Augment class 
class Augment():
    
    def aug(self):
        # augmentation techniques for training data stored as an ImageDataGenerator object
        training_data = ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        
        # augmentation techniques for testing data stored as an ImageDataGenerator object
        testing_data = ImageDataGenerator(scale_width = 1/255)
        
        # using ImageDataGenerator object's functionality to access training images and augment images in folder 
                # .flow_from_directory function in ImageDataGenerator class to augment, set image size, batch_size, classification type
                    # creating training data from .flow_from_directory
        training_data = train_datagen.flow_from_directory('/Images/training',
                                                          target_size = (150,150),
                                                          batch_size = 32,
                                                          class_mode = 'binary')
        
         # using ImageDataGenerator object's functionality to access training images and augment images in folder 
                # .flow_from_directory function in ImageDataGenerator class to augment, set image size, batch_size, classification type
                    # creating training data from .flow_from_directory        
        testing_data = test_datagen.flow_from_directory('/Images/testing',
                                                        target_size = (150,150),
                                                        batch_size=32,
                                                        class_mode = 'binary')
        # x training data 
        x_train = training_data[0][0]
        # training data split from 0 to 1 for activation functions
        x_train /= 255
        # change shape (add axis) to x training for model 
        x_train = np.rollaxis(x_train,3,1)
        # y training data
        y_train = training_data[0][1]
        
        # x testing data 
        x_test = testing_data[0][0]
        # scale values in tx testing from 0 to 1 for activation functions in model 
        x_test /= 255
        # change shape (add axis) to x testing data for model 
        x_test = np.rollaxis(x_test,3,1)
        # y testing data 
        y_test = testing_data[0][1]
