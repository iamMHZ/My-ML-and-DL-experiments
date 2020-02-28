import keras.backend as keras_backend
from keras.layers import MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential


class LeNet:

    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)

        if keras_backend.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # define model
        model = Sequential()

        # CONV = > RELU = > POOL
        # add layers
        # 20 filters of size 5*5
        model.add(Conv2D(20, (5, 5), input_shape=input_shape, padding='same'))
        model.add(Activation(activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # CONV => RELU => POOL
        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation(activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # FC = > RELU
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation='relu'))

        model.add(Dense(classes))
        model.add(Activation(activation='softmax'))

        return model
