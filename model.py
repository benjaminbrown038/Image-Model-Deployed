
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten
import augment
class Model():

    model = Sequential()

    model.add(Conv2D(8,
                 kernel_size = 3,
                 padding = 'same',
                 input_shape = (3,150,150),
                 activation = 'relu'))

    model.add(MaxPool2D(pool_size = (2,2),
                    padding = 'same'))

    model.add(Conv2D(64, kernel_size = (3,3),
                 padding = 'same',
                 activation = 'relu'))

    model.add(MaxPool2D(pool_size(2,2),
                    activation = 'relu'))

    model.add(Conv2D(128,
                 activation = 'softmax',
                 padding = 'same'))

    model.add(MaxPool2D(pool_size = (4,4),
                    padding = 'same'))


    model.add(Flatten())

    # info for compile: loss , metrics , learning rate, optimizer,
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)

    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    # info for fit : data (training and testing),
    model.fit(training_data, testing_data, epochs=100, verbose=2, )

