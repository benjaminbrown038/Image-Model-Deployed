import os
search_name = 'dogs'
# grabbing images from training folder of each class and augmenting
training_directory = "Data/train/dogs"
training_batch_size = len(os.listdir(training_directory))
training_data = flow_from_directory('Data/train/dogs',
                                                          target_size = (150,150),
                                                          batch_size = training_batch_size,
                                                          shuffle = False,
                                                          class_mode = 'binary')

# endup creating batch size to the number of images in the folder
testing_directory = 'Data/train/dogs'
testing_batch_size = len(os.listdir(testing_directory))
testing_data = testing.flow_from_directory('Data/val/dogs',
                                                        target_size = (150,150),
                                                        batch_size= testing_batch_size,
                                                        shuffle = False,
                                                        class_mode = 'binary')


# x training data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
x_train = training_data[0][0]
# training data split from 0 to 1 for activation functions
x_train /= 255
# change shape (add axis) to x training for model
x_train = np.rollaxis(x_train,3,1)
# y training data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
y_train = training_data[0][1]

#return y_train

# x testing data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
x_test = testing_data[0][0]
# scale values in tx testing from 0 to 1 for activation functions in model
x_test /= 255
# change shape (add axis) to x testing data for model
x_test = np.rollaxis(x_test,3,1)
# y testing data after applying .flow_from_directory from ImageDataGenerator class in keras.image.preprocess
y_test = testing_data[0][1]
