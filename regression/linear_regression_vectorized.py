import matplotlib.pyplot as plt
import numpy as np


def load_data():
    # https://www.kaggle.com/aungpyaeap/fish-market?select=Fish.csv
    data_file = np.genfromtxt('../utils/datasets/regression dataset/fish.csv', delimiter=',')

    X = data_file[1:, 1]  # weight
    y = data_file[1:, 2]  # vertical length

    plt.scatter(X, y)
    plt.show()

    return X, y


def compute_gradient(y_predict, y_true):
    pass


def fit(X, y, learning_rate=0.001, epochs=30):
    pass


def main():
    X, y = load_data()

    # adding an extra column to the X ==> it makes the training of bias and weights easier and all
    # in vectorized format in linear_regression.py they are trained separately

    X = X.reshape(X.shape[0], 1)
    column = np.ones((X.shape[0], 1), np.int)
    X = np.append(X, column, axis=1)

    print(X.shape)


main()
