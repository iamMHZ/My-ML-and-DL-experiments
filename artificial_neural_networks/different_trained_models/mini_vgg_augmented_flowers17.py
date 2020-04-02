import numpy as np
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from artificial_neural_networks.convolutional_neural_networks.vgg_net.mini_vgg_net import MiniVGGNet
from utils.loaders.image_loader import ImageLoader
from monitoring.training_monitoring import TrainingMonitor
from utils.preprocessors.image_preprocessors import AspectAwareResizePreprocessor, ImageToArrayPreprocessor

input_width = 64
input_height = 64
input_depth = 3

epochs = 100
batch_size = 32

learning_rate = 0.05


def train_network(data, labels, num_classes):
    # convert label to vectors
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)

    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)

    # build models
    model = MiniVGGNet.build(input_width, input_height, input_depth, num_classes, batch_normalization=True)

    print('[INFO] Compiling model...')
    sgd = SGD(learning_rate=learning_rate)
    model.compile(optimizer=sgd, loss=['categorical_crossentropy'], metrics=['accuracy'])

    # training

    # checkpoints = get_model_checkpoint_callback('VGG_flowers17', only_best_model=True)

    # model babysitting:
    TrainingMonitor('VGG_flowers17')

    callbacks = [TrainingMonitor('VGG_flowers17')]

    # data augmentation:

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    number_batch_per_epoch = int(len(train_x) / 32)

    print('[INFO] Training...')

    model_history = model.fit_generator(aug.flow(train_x, train_y), number_batch_per_epoch, epochs=epochs,
                                        callbacks=callbacks, validation_data=(test_x, test_y), verbose=1)

    # save trained model to disk
    print('[INFO] Serializing model...')
    model.save('./MiniVGG.hdf5')

    # evaluating
    print("[INFO] EVALUATING...")

    predictions = model.predict(test_x, batch_size=batch_size, verbose=1)

    print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=[name for name in label_encoder.classes_]))


if __name__ == '__main__':
    # loading data
    data_path = 'D:\Programming\database of image\Datasets\Flower17\\train'

    resizer = AspectAwareResizePreprocessor(input_width, input_height)
    image_to_array = ImageToArrayPreprocessor()
    loader = ImageLoader(preprocessors=[resizer, image_to_array])

    data, labels = loader.load(data_path)
    # normalize data
    data = data.astype('float') / 255.0

    unique_labels = np.unique(labels)
    classes = len(unique_labels)

    print(unique_labels, classes)

    train_network(data, labels, classes)
