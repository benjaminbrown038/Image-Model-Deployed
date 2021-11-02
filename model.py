
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Optimizer, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy

opt = ['SGD', 'RMSprop', 'Optimizer', 'Nadam', 'Ftrl', 'Adamax', 'Adam', 'Adagrad', 'Adadelta']


'''
Reference for this architecture: https://github.com/girishkuniyal/Cat-Dog-CNN-Classifier
'''
# create Model class
#class Model():
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
Accessing examples from (training) data, scaling image data 0 to 1, creating a dimension on image data, accessing labels (training)

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

    #def model(filters,):
        classifier = Sequential()
        # filters = 32
        classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size=(2,2),strides=2)) #if stride not given it equal to pool filter size
        classifier.add(Conv2D(32,(3,3),activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
        classifier.add(Flatten())
        classifier.add(Dense(units=128,activation='relu'))
        classifier.add(Dense(units=1,activation='sigmoid'))
        opt = SGD(learning_rate=0.01)
        loss = BinaryCrossentropy()
        ac = Accuracy()
        classifier.compile(loss=loss, optimizer=opt,metrics=ac)
        classifier.fit(x_train,y_train,
            validation_data =(x_test,y_test),
            epochs=100,
            verbose=2,
