
# import tensorflow as tf
# import Sequential model container to store model attributes
#import tensorflow
#from tensorflow import keras
from tensorflow.keras.models import Sequential
# model attribute brought in from layers module inside keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# import augment file that contains: x training, y training, x testing, y testing (training and testing data)
#import augment
'''
Reference for this architecture: https://github.com/girishkuniyal/Cat-Dog-CNN-Classifier
'''
# create Model class
#class Model():


classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2)) #if stride not given it equal to pool filter size
classifier.add(Conv2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
from tensorflow.keras.optimizers import SGD
opt = SGD(learning_rate=0.01)
from tensorflow.keras.losses import BinaryCrossentropy
loss = BinaryCrossentropy()
from tensorflow.keras.metrics import Accuracy
ac = Accuracy()
classifier.compile(loss=loss, optimizer=opt,metrics=ac)
#model.fit(x_train,y_train,
#        validation_data =(x_test,y_test),
#        epochs=100,
#        verbose=2,
#        batch_size = 32)

print("DOne")

    # instantiate Sequential object to store model attributes
#model = Sequential()

    # add attributes (conv2D layer) using .add() from keras Sequential class; convolutions extract features from images, adding complexity in features in the later layers
    # number of filters (feature extraction), kernel_size (size of filters),
    # padding ('same' adds a white border around image) for better feature extraction
    # input includes color channel (RGB=3), along with width and heighth
    # relu to scale output of layer (to not return negative values)
#model.add(Conv2D(8,
#                 kernel_size = 3,
#                 padding = 'same',
#                 input_shape = (3,150,150),
#                 activation = 'relu'))
    # add attributes (layers) using .add() from keras Sequential class
    #
#model.add(MaxPool2D(pool_size = (2,2),
#                    padding = 'same'))

    # add attributes (layers) using .add() from keras Sequential class
#    model.add(Conv2D(64, kernel_size = (3,3),
#                 padding = 'same',
#                 activation = 'relu'))
    # add attributes (layers) using .add() from keras Sequential class
#    model.add(MaxPool2D(pool_size(2,2),
#                    activation = 'relu'))
    # add attributes (layers) using .add() from keras Sequential class
#    model.add(Conv2D(128,
#                 activation = 'softmax',
#                 padding = 'same'))
    # add attributes (layers) using .add() from keras Sequential class
#    model.add(MaxPool2D(pool_size = (4,4),
#                    padding = 'same'))

    # add attributes (layers) using .add() from keras Sequential class
#    model.add(Flatten())

    # insantiate SGD optimizer from keras optimizers library; use learning rate input parameter to be set for .01 (this is standard)
    #opt = tf.keras.optimizers.SGD(learning_rate=0.01)

    # using losses libary from keras for binary crossentropy (2 class classification); keeping all inputs standard.
    # loss is a measurable function for comparing hypothesis function against true data on 1 iteration (See Andrew Ng's Coursera course on Machine Learning for more info)
    #loss = tf.keras.losses.BinaryCrossentropy()

    # using metrics class from keras for accuracy to measure overall model performance
    # this is used to test all inputs & weights (at once) against expected output
    #ac = tf.keras.metrics.Accuracy()

    # info for compile: loss , metrics , learning rate, optimizer
    #model.compile(loss=loss, optimizer=opt,metrics=acc)

    # info for fit : data (training and testing), epochs (number of times all data runs through model), verbose (selected output from keras after each epoch),
    # batch_size: number of SETS training data
    #model.fit(x_train,y_train, validation_data =(x_test,y_test), epochs=100, verbose=2, batch_size = 32)
