# from tensorflow import keras
import matplotlib.pyplot  as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from numpy import genfromtxt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

batch_size = 4
epochs = 200
num_classes = 2

learning_rate = 0.00001


def load_data():
    my_data = genfromtxt('data_breast_canser.csv', delimiter=',')
    trai_per = 0.8
    np.random.shuffle(my_data)

    y = my_data[:, -1]
    x = my_data[:, :-1]

    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    y = to_categorical(y, num_classes=num_classes)

    x = (x - x.min(axis=0) / x.max(axis=0) - x.min(axis=0))

    test_boundary = int(x.shape[0] * trai_per)

    train_x = x[0:test_boundary]
    train_y = y[0:test_boundary]
    test_x = x[test_boundary:]
    test_y = y[test_boundary:]

    return train_x, train_y, test_x, test_y


def build_model():
    model = Sequential()
    model.add(Dense(16, input_shape=(3,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model


def train(model, train_x, train_y, test_x, test_y):
    sgd = SGD(learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model_history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size)

    return model_history


def plot(model_history):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), model_history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), model_history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), model_history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), model_history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('MNIST_plot.png')
    plt.show()


if __name__ == '__main__':
    my_data = genfromtxt('data_breast_canser.csv', delimiter=',')
    trai_per = 0.8
    np.random.shuffle(my_data)

    y = my_data[:, -1]
    x = my_data[:, :-1]

    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    y = to_categorical(y, num_classes=num_classes)

    x = (x - x.min(axis=0) / x.max(axis=0) - x.min(axis=0))

    test_boundary = int(x.shape[0] * trai_per)

    train_x = x[0:test_boundary]
    train_y = y[0:test_boundary]
    test_x = x[test_boundary:]
    test_y = y[test_boundary:]

    print(train_x.shape)

    model = build_model()

    model_history = train(model, train_x, train_y, test_x, test_y)

    print('Saving model...')
    model.save('./model.hdf5')
    # evaluating
    print("EVALUATING...")
    predictions = model.predict(test_x, batch_size=batch_size, verbose=1)

    print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=[str(l) for l in lb.classes_]))

    plot(model_history)
