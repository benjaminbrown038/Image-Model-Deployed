'''
    Using ImageDataGenerator object's functionality to access training and validation images to augment images in folder
parameters:
    search_name: <string> this will be the class of the picture.
returns:
    training_data: an instance of augmented images (training) of search_name
    validation_data: an instance of augmented images (validation) of search_name
'''


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import splitfolders
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.applications.vgg16 import VGG16


splitfolders.ratio('Images', output="Augmented_Images", seed=1337, ratio=((0.8, 0.2)))

'''
Augmenting Training Data
'''
# augmentation object for training data
training = ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

# creating augmented images from folder with keras user parameters
training_directory = 'Augmented_Images/train/'
training_batch_size = len(os.listdir(training_directory))
training_data = training.flow_from_directory(training_directory,
                                                target_size = (150,150),
                                                batch_size= training_batch_size,
                                                shuffle = False,
                                                class_mode = 'binary')

# training data variables from keras augmented training data object
x_train = training_data[0][0]
x_train /= 255
x_train = np.rollaxis(x_train,3,1)
y_train = training_data[0][1]

'''
Augmenting Validation Data
'''
validation = ImageDataGenerator(
                rescale = 1/255)

# creating validation data variables from validation folder images
validation_directory = 'Augmented_Images/val/'
validation_batch_size = len(os.listdir(validation_directory))
validation_data = validation.flow_from_directory(validation_directory,
                                                                target_size = (150,150),
                                                                batch_size= validation_batch_size,
                                                                shuffle = False,
                                                                class_mode = 'binary')
# validation variables from keras validation object
x_val = validation_data[0][0]
x_val /= 255
x_val = np.rollaxis(x_val,3,1)
y_val = validation_data[0][1]

'''
Model Building
'''

'''
classifier = Sequential()

classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Conv2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
'''

'''
1. Optimizer of model
2. Loss function for CNN
3. Metric used: Accuracy
'''
# parameters
opt = SGD(learning_rate=0.01)
loss = BinaryCrossentropy()
ac = Accuracy()

'''
Compile Parameters
'''
#classifier.compile(loss=loss,
#                   optimizer=opt,
#                   metrics=ac)
'''
Data Fitting to Model
'''

#classifier.fit(x_train,
#               y_train,
#               validation_data = (x_test,y_test),
#               epochs=100,
#               verbose=2)




#if __name__ == "__main__":
#    main()
