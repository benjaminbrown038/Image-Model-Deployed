from keras.image.preprocess import ImageDataGenerator
import numpy as np

# need to save training and testing data so model.py can access it   

class Augment():
    
    def aug(self):
    
        training_data = ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        testing_data = ImageDataGenerator(scale_width = 1/255)

        training_data = train_datagen.flow_from_directory('/Images/training',
                                                          target_size = (150,150),
                                                          batch_size = 32,
                                                          class_mode = 'binary')

        testing_data = test_datagen.flow_from_directory('/Images/testing',
                                                        target_size = (150,150),
                                                        batch_size=32,
                                                        class_mode = 'binary')
        x_train = training_data[0][0]
        x_train /= 255
        x_train = np.rollaxis(x_train,3,1)
        y_train = training_data[0][1]

        x_test = testing_data[0][0]
        x_test /= 255
        x_test = np.rollaxis(x_test,3,1)
        y_test = testing_data[0][1]