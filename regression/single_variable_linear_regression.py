import matplotlib.pyplot as plt
import numpy as np


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

    delta0 = np.sum(predictions - y)
    delta1 = np.sum((predictions - y) * x[:, 1:])

    gradient = np.array([[delta0], [delta1]])

    return gradient


if __name__ == '__main__':
    x = np.array([[1, 1], [1, 2]])

    y = np.array([[1], [2]])
    w = np.array([[0], [1]])

    plot(x, y, w)
    compute_loss(x, y, w)

    gradient = compute_gradient(x, y, w)
    print(gradient)
