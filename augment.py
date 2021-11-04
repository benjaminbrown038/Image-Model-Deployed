from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import split_folders
import os
import os
import numpy as np
import matplotlib.pyplot as plt

class Augment():
    def __init__(self,search_name):
'''
    Using ImageDataGenerator object's functionality to access training and validation images to augment images in folder
parameters:
    search_name: <string> this will be the class of the picture.
returns:
    training_data: an instance of augmented images (training) of search_name
    validation_data: an instance of augmented images (validation) of search_name
'''
        split_folders.ratio('Images', output="Augmented_Images", seed=1337, ratio=((0.8, 0.2)))
        self.training = ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        self.validation = ImageDataGenerator(
            scale_width = 1/255)

    def training_exploration(folder_path):
        train_dir = folder_path
        normal_imgs = [fn for fn in os.listdir(f'{train_dir}/NORMAL') if fn.endswith('.jpeg')]
        pneumo_imgs = [fn for fn in os.listdir(f'{train_dir}/PNEUMONIA') if fn.endswith('.jpeg')]
        select_norm = np.random.choice(normal_imgs, 3, replace = False)
        select_pneu = np.random.choice(pneumo_imgs, 3, replace = False)
        fig = plt.figure(figsize = (8,6))
        for i in range(6):
            if i < 3:
                fp = f'{train_dir}/NORMAL/{select_norm[i]}'
                label = 'NORMAL'
            else:
                fp = f'{train_dir}/PNEUMONIA/{select_pneu[i-3]}'
                label = 'PNEUMONIA'
                ax = fig.add_subplot(2, 3, i+1)
        fn = image.load_img(fp, target_size = (100,100), color_mode='grayscale')
        plt.imshow(fn, cmap='Greys_r')
        plt.title(label)
        plt.axis('off')
        plt.show()
        len(normal_imgs), len(pneumo_imgs)

    def aug(*classes):
        for label in classes:
            search_name = 'dogs'
            training_directory = "Data/train/" + label
            training_batch_size = len(os.listdir(training_directory))
            training_data = flow_from_directory(training_directory,
                                                                  target_size = (150,150),
                                                                  batch_size = training_batch_size,
                                                                  shuffle = False,
                                                                  class_mode = 'binary')
            validation_directory = 'Data/val/' + label
            validation_batch_size = len(os.listdir(validation_directory))
            validation_data = validation.flow_from_directory(validation_directory,
                                                                target_size = (150,150),
                                                                batch_size= validation_batch_size,
                                                                shuffle = False,
                                                                class_mode = 'binary')
            x_train = training_data[0][0]
            x_train /= 255
            x_train = np.rollaxis(x_train,3,1)
            y_train = training_data[0][1]

            x_val = validation_data[0][0]
            x_val /= 255
            x_val = np.rollaxis(x_val,3,1)
            y_val = validation_data[0][1]
