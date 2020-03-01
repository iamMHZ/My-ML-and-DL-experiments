import keras.backend as keras_backend
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


class MiniVGGNet:

    @staticmethod
    def build(width, height, depth, classes, batch_normalization=True):
        # image input shape based on keras backend
        input_shape = (height, width, depth)
        # batchNormalization is applied over channels , so orders must be known
        channel_dim = -1

        if keras_backend.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            channel_dim = 1
        # model architecture :

        model = Sequential()

        # CONV = > RELU = > CONV = > RELU = > POOL

        model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, padding='same'))
        model.add(Activation('relu'))
        if batch_normalization:
            model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        if batch_normalization:
            model.add(BatchNormalization(axis=channel_dim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))

        # CONV = > RELU = > CONV = > RELU = > POOL

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        if batch_normalization:
            model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        if batch_normalization:
            model.add(BatchNormalization(axis=channel_dim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))

        #  FC and softmax
        model.add(Flatten())
        model.add(Dense(512))

        model.add(Activation('relu'))
        if batch_normalization:
            model.add(BatchNormalization(axis=channel_dim))

        model.add(Dropout(rate=0.5))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
