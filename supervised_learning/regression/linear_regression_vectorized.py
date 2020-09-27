import matplotlib.pyplot as plt
import numpy as np


def load_data():
    # https://www.kaggle.com/aungpyaeap/fish-market?select=Fish.csv
    data_file = np.genfromtxt('../../utils/datasets/supervised dataset/fish.csv', delimiter=',')

    X = data_file[1:, 1]  # weight
    y = data_file[1:, 2]  # vertical length

    plt.scatter(X, y)
    plt.show()

    return X, y


def compute_gradient(X, y_predict, y_true):
    errors = y_predict - y_true
    # compute the epoch loss
    epoch_loss = 0.5 * np.sum(errors ** 2)
    # compute the gradient
    gradient = np.matmul(errors, X)

    return epoch_loss, gradient


def fit(X, y, learning_rate=0.001, epochs=30):
    # start the weights randomly
    weights = np.random.random((1, 2))

    losses = []
    for i in range(epochs):
        prediction = np.matmul(weights, X.T)

        epoch_loss, gradients = compute_gradient(X, y_predict=prediction, y_true=y)

        weights += -learning_rate * gradients

        print(f'Epoch {i}, Loss {epoch_loss}')

        losses.append(epoch_loss)

    plt.plot(np.arange(0, epochs), losses)
    plt.show()
    print(weights)


def main():
    X, y = load_data()

    # adding an extra column to the X ==> it makes the training and the gradient computation
    # of bias and weights easier and all
    # in vectorized format in linear_regression.py they are separate

    X = X.reshape(X.shape[0], 1)
    column = np.ones((X.shape[0], 1), np.int)
    X = np.append(X, column, axis=1)

    # print(X.shape)

    y = y.reshape(1, y.shape[0])

    fit(X, y, 0.000000005, epochs=30)


main()
