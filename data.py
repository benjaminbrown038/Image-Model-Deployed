'''
    Using ImageDataGenerator object's functionality to access training and validation images to augment images in folder
parameters:
    search_name: <string> this will be the class of the picture.
returns:
    training_data: an instance of augmented images (training) of search_name
    validation_data: an instance of augmented images (validation) of search_name
'''
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import split_folders
import os
import os
import numpy as np
import matplotlib.pyplot as plt



split_folders.ratio('Images', output="Augmented_Images", seed=1337, ratio=((0.8, 0.2)))
    
'''
Training Data
'''

training = ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
 
'''
Validation Data
'''
validation = ImageDataGenerator(
            scale_width = 1/255)

for label in classes:
    training_directory = "Data/train/"
    
    training_batch_size = len(os.listdir(training_directory))
        
    training_data = flow_from_directory(training_directory,
                                                      target_size = (150,150),
                                                      batch_size = training_batch_size,
                                                      shuffle = False,
                                                      class_mode = 'binary')
'''
Validation Data
'''
                                      
validation_directory = 'Data/val/'

validation_batch_size = len(os.listdir(validation_directory))

validation_data = validation.flow_from_directory(validation_directory,
                                                                target_size = (150,150),
                                                                batch_size= validation_batch_size,
                                                                shuffle = False,
                                                                class_mode = 'binary')
'''
Training Data - x & y 
'''
x_train = training_data[0][0]
x_train /= 255
x_train = np.rollaxis(x_train,3,1)
y_train = training_data[0][1]

'''
Validation Data - x & y
'''
x_val = validation_data[0][0]
x_val /= 255
x_val = np.rollaxis(x_val,3,1)
y_val = validation_data[0][1]

#if __name__ == "__main__":
#    main()
