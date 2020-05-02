import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, w):
    """
     Plots the data and the predicted line

    :param x: the data
    :param y:  grand truth labels
    :param w:  the predicted line
    :return: None
    """

    line = np.matmul(x, w)

    plt.scatter(x[:, 1:], y)
    plt.plot(x[:, 1:], line)
    plt.show()


def loss(x, y, w):
    """
     Plots the data and the pridicted line

    :param x: the data
    :param y:  grand truth labels
    :param w:  the predicted line
    :return: None
    """


if __name__ == '__main__':
    x = np.array([[1, 1], [1, 2]])

    y = np.array([[1], [2]])
    w = np.array([[0], [1]])

    plot(x, y, w)
