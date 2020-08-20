import numpy as np
from matplotlib import pyplot as plt


def load_data():

    # https://www.kaggle.com/aungpyaeap/fish-market?select=Fish.csv
    data_file = np.genfromtxt('../utils/datasets/regression dataset/fish.csv', delimiter=',')

    data = data_file[1:, 1]  # weight
    labels = data_file[1:, 2]  # vertical length

    return data, labels


def plot(x, y):
    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    data, labels = load_data()

    plot(data, labels)
