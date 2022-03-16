
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Optimizer, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.applications.vgg16 import VGG16

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
training_directory = 'Augmented_Images/train/' + search_name + '/'
        
training_batch_size = len(os.listdir(training_directory))
        
training_data = training.flow_from_directory('Augmented_Images/train/' + search_name,
                                                          target_size = (150,150),
                                                          batch_size = training_batch_size,
                                                          shuffle = False,
                                                          class_mode = 'binary')

 
training_directory = 'Augmented_Images/train/' + search_name + '/'

testing_batch_size = len(os.listdir(testing_directory))

testing_data = testing.flow_from_directory('Augmented_Images/test/' + search_name,
                                                        target_size = (150,150),
                                                        batch_size= testing_batch_size,
                                                        shuffle = False,
                                                        class_mode = 'binary')

'''
Accessing examples from (training) data, scaling image data 0 to 1, creating a dimension on image data, accessing labels (training)

parameters:

returns: training images and training labels
'''

x_train = self.training_data[0][0]
x_train /= 255
x_train = np.rollaxis(x_train,3,1)
y_train = self.training_data[0][1]



'''
Accessing examples from (testing) data, scaling image data 0 to 1, creating a dimension on image data, accessing labels (testing)

parameters:

returns: testing images and testing labels

'''

x_test = self.testing_data[0][0]
x_test /= 255
x_test = np.rollaxis(x_test,3,1)
y_test = self.testing_data[0][1]


classifier = Sequential()
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
        
classifier.compile(loss=loss, 
                   optimizer=opt,
                   metrics=ac)
        
classifier.fit(x_train,
               y_train,
               validation_data = (x_test,y_test),
               epochs=100,
               verbose=2)

#    if __name__ == "__main__":
#        main()
