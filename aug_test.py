import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
splitfolders.ratio('Images', output="Augmented_Images", seed=1337, ratio=((0.8, 0.2)))

        # augmentation techniques for training data stored as an ImageDataGenerator object
training = ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # augmentation techniques for testing data stored as an ImageDataGenerator object
validation = ImageDataGenerator(
            rescale = 1/255)
#

training_directory = "Augmented_Images/train/dogs"
training_batch_size = len(os.listdir(training_directory))
training_data = training.flow_from_directory('Augmented_Images/train/dogs',
                                                          target_size = (150,150),
                                                          batch_size = training_batch_size,
                                                          shuffle = False,
                                                          class_mode = 'binary')

# endup creating batch size to the number of images in the folder
validation_directory = 'Augmented_Images/val/dogs'
validation_batch_size = len(os.listdir(validation_directory))
validation_data = validation.flow_from_directory('Augmented_Images/val/dogs',
                                                        target_size = (150,150),
                                                        batch_size= validation_batch_size,
                                                        shuffle = False,
                                                        class_mode = 'binary')
