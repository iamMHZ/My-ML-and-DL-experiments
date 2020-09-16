import time

import matplotlib.pyplot as plt
import numpy as np


# TODO remove duplications
def plot(x, y, w):
    """
    Plots the data and the predicted line

    :param x:  data
    :param y:  grand truth labels
    :param w:  weight matrix
    :return: None
    """

    line = np.matmul(x, w)

    # x[:, 1:] we discard bios column
    plt.scatter(x[:, 1:], y)
    plt.plot(x[:, 1:], line, c='r')
    plt.show()


def compute_loss(x, y, w):
    """
    Calculates the sum of squared loss

    :param x:  data
    :param y:  grand truth labels
    :param w:  weight matrix
    :return: sum of squared loss
    """

    predictions = np.matmul(x, w)

    errors = predictions - y

    loss = 0.5 * np.sum(errors ** 2)

    print(f'Errors {errors}')
    print(f'sum of squared loss {loss}')


def compute_gradient(x, y, w):
    """
    Computes the derivatives with respect to each w

    :param x:  data points
    :param y:  grand truth labels
    :param w:  weight matrix
    :return: return the gradient vector
    """

    predictions = np.matmul(x, w)

    error = predictions - y

    delta0 = np.sum(error)
    delta1 = np.sum(error * x[:, 1:])

    gradient = np.array([[delta0], [delta1]])

    return gradient


def vanilla_gradient(x, y, w, learning_rate, epochs=20):
    for i in range(epochs):
        gradient = compute_gradient(x, y, w)
        w = w - (learning_rate * gradient)

        plot(x, y, w)

        # time.sleep(0.5)
    return w


if __name__ == '__main__':
    x = np.array([[1, 1], [2, 2]])

    y = np.array([[1], [2]])
    w = np.array([[0], [0]])

    # plot(x, y, w)
    # compute_loss(x, y, w)
    #
    # gradient = compute_gradient(x, y, w)
    # print(gradient)

    w = vanilla_gradient(x, y, w, learning_rate=0.01, epochs=100)

    print(w)
