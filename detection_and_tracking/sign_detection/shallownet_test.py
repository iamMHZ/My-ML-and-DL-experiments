from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from artificial_neural_networks.convolutional_neural_networks.shallownet.shallownet import ShallowNet
from utils.loaders.image_loader import ImageLoader
from utils.model_monitoring.checkpoints import get_model_checkpoint_callback
from utils.model_monitoring.training_monitoring import TrainingMonitor
from utils.preprocessors.image_preprocessors import ImageToArrayPreprocessor, ResizePreprocessor

input_width = 32
input_height = 32
input_depth = 3

num_classes = 3

batch_size = 32
epochs = 100
learning_rate = 0.0001


def train_and_test(data, labels):
    # splitting dataset
    # data = data.astype('float') / 255.0
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)

    # convert labels to vectors
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)
    y_train = lb.fit_transform(y_train)

    # y_test = np_utils.to_categorical(y_test, num_classes=num_classes)
    # y_train = np_utils.to_categorical(y_train, num_classes=num_classes)

    print('[INFO] DATA PREPARED')

    # build model
    model = ShallowNet.build(input_width, input_height, input_depth, num_classes)
    # plot_model(model, to_file='ShallowNet.png', show_shapes=True)

    print('[INFO] COMPILING MODEL...')
    sgd = SGD(learning_rate=learning_rate)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print('[INFO] TRAINING...')

    # checkpoint model and monitor loss and accuracy
    callbacks = [TrainingMonitor('shallowNetHazmat')]
    checkpoint = get_model_checkpoint_callback(file_name='shallowNetSign', only_best_model=True)
    callbacks.append(checkpoint)

    model_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                              validation_data=(x_test, y_test),
                              callbacks=callbacks)

    # evaluating network
    print('[INFO] EVALUATING network...')
    predictions = model.predict(x_test, batch_size=batch_size, verbose=1)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=[str(l) for l in lb.classes_]))


def main():
    dataset_path = 'D:\Programming\database of image\HAZMAT\Datasets\smallDataset'

    augmented_dataset_path = 'D:\Programming\database of image\HAZMAT\Datasets\smallAugmentedDataset'

    image_to_array = ImageToArrayPreprocessor()
    resizer = ResizePreprocessor(input_width, input_height)

    loader = ImageLoader(preprocessors=[resizer, image_to_array])

    data, labels = loader.load(augmented_dataset_path, display_data=True)

    # pprint.pprint(labels)

    train_and_test(data, labels)


if __name__ == '__main__':
    main()
