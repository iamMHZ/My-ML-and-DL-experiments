"""
Regression with L2 regularization
"""

import matplotlib.pyplot as plt
import numpy as np


def load_data():
    # https://www.kaggle.com/aungpyaeap/fish-market?select=Fish.csv
    data_file = np.genfromtxt('../../utils/datasets/supervised dataset/fish.csv', delimiter=',')

    X = data_file[1:, 5:]  # height and width of fishes
    y = data_file[1:, 1]  # weight of fishes

    print(X[:5, :])
    print(y[:5])

    return X, y


def compute_loss(y_true, y_pred, weights, landa):
    error = y_pred - y_true

    epoch_loss = np.sum(error ** 2)

    # apply regularization
    # No regularization for the bias
    epoch_loss += 0.5 * landa * np.sum(weights[1:] ** 2)

    return epoch_loss


def compute_gradients(X, y_true, y_pred, weights, landa):
    error = y_pred - y_true

    gradients = np.matmul(X.T, error)

    # add derivatives of L2 regularization
    # Note that no regularization is applied on the bias
    gradients[1:] += landa * weights[1:]

    return gradients


def fit(X, y, learning_rate=0.00001, epochs=30, landa=0.0001):
    # initialize the weights

    weights = np.random.random((X.shape[1], 1))

    losses = []

    for i in range(epochs):
        # make a prediction
        y_pred = np.matmul(X, weights)

        epoch_loss = compute_loss(y, y_pred, weights, landa)

        # update the wights
        gradients = compute_gradients(X, y, y_pred, weights, landa)
        weights += -learning_rate * gradients
        print(f'Epoch = {i} , Loss = {epoch_loss}')

        losses.append(epoch_loss)

    # plot the training loss
    plt.plot(np.arange(0, epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    print('Weights: ' + str(weights))


if __name__ == '__main__':
    X, y = load_data()

    # from sklearn.datasets import make_regression
    # X, y = make_regression(n_samples=300, n_features=2)

    # add an extra column for the bias this way you don't need to compute the gradients separately
    ones_column = np.ones((X.shape[0], 1), np.float)
    X = np.append(ones_column, X, axis=1)

    y = y.reshape(y.shape[0], 1)

    fit(X, y, learning_rate=0.00001, epochs=30, landa=0)
