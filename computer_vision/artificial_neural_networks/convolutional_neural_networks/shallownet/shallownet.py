from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential


class ShallowNet:

    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        model = Sequential()
        # CONV => RELU
        # this layer will have 32 filters (K)
        # each of which are 3 × 3
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
