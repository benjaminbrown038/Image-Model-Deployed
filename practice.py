from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.applications.vgg16 import VGG16

# split data into validation and training after created new folder.
splitfolders.ratio('Images', output= 'Augmented_Images', seed = 1337, ratio = ((0.8,0.2)))

training = ImageDataGenerator(rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')

training_directory = 'Augmented_Images/train/'
# variable with batch size, from images in training dir
training_batch_size = len(os.listdir(training_directory))
training_data = training.flow_from_directory(training_directory,
                                        target_size=(150,150),
                                        batch_size = training_batch_size,
                                        shuffle = False,
                                        class_mode = 'binary')

validation = ImageDataGenerator(rescale = 1./255)

validation_directory = 'Augmented_Images/val/'
validation_batch_size = len(os.listdir(validation_directory))
validation_data = validation.flow_from_directory(validation_directory)

x_train = training_data
type(x_train[0])
model = Sequential()
model.add(Conv2D(8,(3,3),input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# optimizers
