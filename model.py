
# import tensorflow as tf
# import Sequential model container to store model attributes
from keras.models import Sequential
# model attribute brought in from layers module inside keras
from keras.layers import Conv2D, MaxPool2D, Flatten
# import augment file that contains: x training, y training, x testing, y testing 
import augment

# create Model class 
class Model():
    
    # instantiate Sequential object to store model attributes
    model = Sequential()
    
    # add attributes (layers) using .add() from keras Sequential class
    # 
    model.add(Conv2D(8,
                 kernel_size = 3,
                 padding = 'same',
                 input_shape = (3,150,150),
                 activation = 'relu'))
    # add attributes (layers) using .add() from keras Sequential class
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

    # info for fit : data (training and testing),
    model.fit(training_data, testing_data, epochs=100, verbose=2 )

