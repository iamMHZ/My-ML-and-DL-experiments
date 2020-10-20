"""
Using numpy for implementing logistic regression
"""

import numpy as np
from matplotlib import pyplot as plt


def load_data():
    # https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival
    data_file = np.genfromtxt('../../utils/datasets/supervised dataset/haberman.txt', delimiter=',')

    X = data_file[:, :2]
    y = data_file[:, 3]

    #  labels are 1 (survived) and 2 (died)
    # change 2 to 0

    y[y == 2] = 0

    return X, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_loss(y_pred, y_true):
    # calculate loss
    epoch_loss = (-y_true * np.log(y_pred)) - ((1 - y_true) * np.log(1 - y_pred))
    epoch_loss = np.sum(epoch_loss)

    #  Replace NaN with zero and infinity with large finite number
    # because the -log(x) and -log(1-x) have the tendency to return NaN or INF so we need to make it a number
    # making sure that the over all loss does not become INF

    epoch_loss = np.nan_to_num(epoch_loss)
    return epoch_loss


# TODO: i am suspicious
def compute_gradient(X, y_pred, y_true):
    # calculate the gradient vector
    error = y_pred - y_true

    gradients = np.matmul(X.T, error)

    return gradients


def fit(X, y, learning_rate=0.0001, epochs=50):
    # initialize weights randomly
    weights = np.random.random((X.shape[1], 1))

    losses = []
    for i in range(epochs):
        # make a prediction
        prediction = sigmoid(np.matmul(X, weights))
        # compute loss
        epoch_loss = compute_loss(prediction, y)

        # update the weights
        gradients = compute_gradient(X, prediction, y)
        weights += -learning_rate * gradients

        print(f'Epoch = {i} , Loss = {epoch_loss}')

        losses.append(epoch_loss)

    # plot the training loss
    plt.plot(np.arange(0, epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    return weights

if __name__ == '__main__':
    X, y = load_data()
    #
    # from sklearn.datasets import make_blobs
    #
    # X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=16)
    #
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    # add a column for the bias (bias trick) ==> everything is vectorized
    ones_column = np.ones((X.shape[0], 1), np.float)
    X = np.append(X, ones_column, axis=1)

    y = y.reshape(y.shape[0], 1)

    fit(X, y, learning_rate=0.001, epochs=100)
