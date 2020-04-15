import keras.backend as keras_backend
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential


class AlexNet:

    @staticmethod
    def build(height=227, width=227, depth=3, classes=17):
        input_shape = (height, width, depth)

        if keras_backend.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        model = Sequential()

        model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        model.add(Flatten())
        model.add(Dense(4096, input_shape=(height * width * depth,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(classes))
        model.add(Activation('relu'))

        model.summary()

        return model
