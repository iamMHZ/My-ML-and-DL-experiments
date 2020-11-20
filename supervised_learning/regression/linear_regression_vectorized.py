"""
A vectorized implementation of the single variable regression using numpy
"""

import matplotlib.pyplot as plt
import numpy as np


class LinearRegressionVectorized:
    def __init__(self):
        # keep track of loss of each epoch
        self.losses = []

        # load the specific dataset
        self.x, self.y = self.load_data()

        # adding an extra column to the X ==> it makes the training
        #  and the gradient computation of bias and weights easier and all
        # in vectorized format in linear_regression.py they are separate
        self.x = self.x.reshape(self.x.shape[0], 1)
        column = np.ones((self.x.shape[0], 1), np.int)
        self.x = np.append(self.x, column, axis=1)
        # reshape y
        self.y = self.y.reshape(1, self.y.shape[0])

        # initialize weights randomly
        self.weights = np.random.random((1, self.x.shape[1]))

    def load_data(self):
        # TODO change this method to be compatible with other dataset
        # https://www.kaggle.com/aungpyaeap/fish-market?select=Fish.csv
        data_file = np.genfromtxt('../../utils/datasets/supervised dataset/fish.csv', delimiter=',')

        x = data_file[1:, 1]  # weight
        y = data_file[1:, 2]  # vertical length

        plt.scatter(x, y)
        plt.show()

        return x, y

    def compute_gradient(self, y_predict):
        errors = y_predict - self.y
        # compute the epoch loss
        epoch_loss = 0.5 * np.sum(errors ** 2)
        # compute the gradient
        gradient = np.matmul(errors, self.x)

        return epoch_loss, gradient

    def fit(self, learning_rate=0.001, epochs=30):
        for i in range(epochs):
            prediction = np.matmul(self.weights, self.x.T)

            epoch_loss, gradients = self.compute_gradient(prediction)

            self.weights += -learning_rate * gradients

            print(f'Epoch {i}, Loss {epoch_loss}')

            self.losses.append(epoch_loss)

        # plot the training loss curve
        plt.plot(np.arange(0, epochs), self.losses)
        plt.show()
        print(self.weights)


if __name__ == '__main__':
    lr = LinearRegressionVectorized()
    lr.fit(0.000000005, epochs=30)
