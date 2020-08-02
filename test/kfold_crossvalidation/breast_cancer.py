import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

LEARNING_RATE = 0.1

EPOCHS = 50
BATCH_SIZE = 4
NUM_FOLDS = 3

CLASSES = 2


def load_data():
    data = np.genfromtxt('../data_breast_cancer.csv', delimiter=',')

    # data cleaning, replacing nan value with zero
    data = np.nan_to_num(data)

    np.random.shuffle(data)

    x = data[:, :-1]
    y = data[:, -1]

    print(data)
    print(x)
    print(y)

    return x, y


def build_model(input_shape):
    model = Sequential()

    model.add(Dense(8, input_shape=input_shape, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(CLASSES, activation='softmax'))

    model.summary()

    sgd = SGD(learning_rate=LEARNING_RATE)
    model.compile(optimizer=sgd, metrics=['accuracy'], loss=['binary_crossentropy'])

    return model


def plot(model_history):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), model_history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), model_history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), model_history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), model_history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('plot.png')
    plt.show()


def train_model(model, data, labels):
    # encoding labels
    lb = LabelEncoder()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels, num_classes=CLASSES)

    # split to train and test
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.1)

    # --- KFOLD Cross validation ---
    kfold = KFold(n_splits=NUM_FOLDS)

    for train_index, val_index in kfold.split(X=train_x):
        train_data_x = train_x[train_index]
        train_data_y = train_y[train_index]

        val_x, val_y = train_x[val_index], train_y[val_index]

        model_history = model.fit(train_data_x, train_data_y, validation_data=(val_x, val_y), epochs=EPOCHS,
                                  batch_size=BATCH_SIZE)

        loss_accuracy = model.evaluate(x=test_x, y=test_y, batch_size=BATCH_SIZE, verbose=1)
        print(f'TEST LOSS AND ACCURACY :  {loss_accuracy}')
        plot(model_history)


# ---- main ----
if __name__ == '__main__':
    x, y = load_data()

    model = build_model(x[0].shape)

    train_model(model, x, y)
