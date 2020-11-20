"""
A vectorized implementation of the multi-variable regression
"""

import numpy as np
from matplotlib import pyplot as plt


class MultiVariableRegression:

    def __init__(self):
        # keep track of loss of each epoch
        self.losses = []

        # load the specific dataset
        self.x, self.y = self.load_data()

        # adding an extra column to the X ==> it makes the training
        #  and the gradient computation of bias and weights easier and all
        # in vectorized format
        column = np.ones((self.x.shape[0], 1), np.int)
        self.x = np.append(self.x, column, axis=1)
        # reshape y
        self.y = self.y.reshape(1, self.y.shape[0])

        # initialize weights randomly
        self.weights = np.random.random((1, self.x.shape[1]))

    def load_data(self):
        data_file = np.genfromtxt('../../utils/datasets/supervised dataset/fish.csv', delimiter=',')

        x = data_file[1:, 5:]  # height and width of fishes
        y = data_file[1:, 1]  # weight of fishes

        print(x[:5, :])
        print(y[:5])

        return x, y

    def compute_gradient(self, y_pred):
        errors = y_pred - self.y

        loss = 0.5 * np.sum(errors ** 2)

        gradients = np.matmul(errors, self.x)

        return loss, gradients

    def fit(self, learning_rate=0.001, epochs=30):
        # full batch gradient descent
        for i in range(epochs):
            # make a prediction
            y_pred = np.matmul(self.weights, self.x.transpose())

            # compute the loss and the gradients
            loss, gradients = self.compute_gradient(y_pred)

            # update the weights
            self.weights += -learning_rate * gradients

            print(f'EPOCH {i}, LOSS {loss}')

            self.losses.append(loss)

        plt.plot(np.arange(0, epochs), self.losses)
        plt.show()

        print('\n\nWeights: ' + str(self.weights))


if __name__ == '__main__':
    lr = MultiVariableRegression()
    lr.fit(learning_rate=0.00001, epochs=30)
