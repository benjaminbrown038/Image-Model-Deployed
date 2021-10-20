
# import tensorflow as tf
# import Sequential model container to store model attributes
from keras.models import Sequential
# model attribute brought in from layers module inside keras
from keras.layers import Conv2D, MaxPool2D, Flatten
# import augment file that contains: x training, y training, x testing, y testing (training and testing data)
import augment

# create Model class 
class Model():
    
    # instantiate Sequential object to store model attributes
    model = Sequential()
    
    # add attributes (conv2D layer) using .add() from keras Sequential class; convolutions extract features from images, adding complexity in features in the later layers
    # number of filters (feature extraction), kernel_size (size of filters), 
    # padding ('same' adds a white border around image) for better feature extraction
    # input includes color channel (RGB=3), along with width and heighth
    # relu to scale output of layer (to not return negative values)
    model.add(Conv2D(8,
                 kernel_size = 3,
                 padding = 'same',
                 input_shape = (3,150,150),
                 activation = 'relu'))
    # add attributes (layers) using .add() from keras Sequential class
    # 
    model.add(MaxPool2D(pool_size = (2,2),
                    padding = 'same'))
    
    # add attributes (layers) using .add() from keras Sequential class
    model.add(Conv2D(64, kernel_size = (3,3),
                 padding = 'same',
                 activation = 'relu'))
    # add attributes (layers) using .add() from keras Sequential class
    model.add(MaxPool2D(pool_size(2,2),
                    activation = 'relu'))
    # add attributes (layers) using .add() from keras Sequential class
    model.add(Conv2D(128,
                 activation = 'softmax',
                 padding = 'same'))
    # add attributes (layers) using .add() from keras Sequential class
    model.add(MaxPool2D(pool_size = (4,4),
                    padding = 'same'))

    # add attributes (layers) using .add() from keras Sequential class
    model.add(Flatten())

    # insantiate SGD optimizer from keras optimizers library; use learning rate input parameter to be set for .01 (this is standard) 
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    
    # using losses libary from keras for binary crossentropy (2 class classification); keeping all inputs standard. 
    # loss is a measurable function for comparing hypothesis function against true data on 1 iteration (See Andrew Ng's Coursera course on Machine Learning for more info)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    # using metrics class from keras for accuracy to measure overall model performance
    # this is used to test all inputs & weights (at once) against expected output 
    ac = tf.keras.metrics.Accuracy()
    
    # info for compile: loss , metrics , learning rate, optimizer
    model.compile(loss=loss, optimizer=opt,metrics=acc)

    # info for fit : data (training and testing), epochs (number of times all data runs through model), verbose (selected output from keras after each epoch),
    # batch_size: number of SETS training data 
    model.fit(x_train,y_train, validation_data =(x_test,y_test), epochs=100, verbose=2, batch_size = 32)
