# import object for augmentation techniques
from keras.image.preprocess import ImageDataGenerator
# for cleaning data
import numpy as np
# for creating train and test folders for each class
import split_folders
import os

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

        split_folders.ratio('Images', output="Data", seed=1337, ratio=((0.8, 0.2)))

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

    def aug(self,search_name):

'''
functions:
    flow_from_directory: function in ImageDataGenerator class to augment, set image size, batch_size, classification type
    creating training_data from
parameters:
    search_name: <string> this will be the class of the picture and label the image for classification
returns:
    training_data: objected containing training data (x and y)
    testing_data: object containing testing data (x and y)
'''

        # grabbing images from training folder of each class and augmenting
        training_directory = '/Images/Data/train/' + search_name + '/'
        training_batch_size = len(os.listdir(training_directory))
        self.training_data = training.flow_from_directory('/Images/Data/train/' + search_name,
                                                          target_size = (150,150),
                                                          batch_size = training_batch_size,
                                                          shuffle = False,
                                                          class_mode = 'binary')

        # endup creating batch size to the number of images in the folder
        testing_batch_size = len(os.listdir(testing_directory))
        self.testing_data = testing.flow_from_directory('/Images/Data/test/' + search_name,
                                                        target_size = (150,150),
                                                        batch_size= testing_batch_size,
                                                        shuffle = False,
                                                        class_mode = 'binary')

  '''
accessing examples from (training) data, scaling image data 0 to 1, creating a dimension on image data, accessing labels (training)

parameters:

returns: training images and training labels

'''

        # x training data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
        x_train = self.training_data[0][0]
        # training data split from 0 to 1 for activation functions
        x_train /= 255
        # change shape (add axis) to x training for model
        x_train = np.rollaxis(x_train,3,1)
        # y training data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
        y_train = self.training_data[0][1]

        return x_train
        return y_train


'''
accessing examples from (testing) data, scaling image data 0 to 1, creating a dimension on image data, accessing labels (testing)

parameters:

returns: testing images and testing labels

'''
        # x testing data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
        x_test = self.testing_data[0][0]
        # scale values in tx testing from 0 to 1 for activation functions in model
        x_test /= 255
        # change shape (add axis) to x testing data for model
        x_test = np.rollaxis(x_test,3,1)
        # y testing data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
        y_test = self.testing_data[0][1]

        return x_test
        return y_test
