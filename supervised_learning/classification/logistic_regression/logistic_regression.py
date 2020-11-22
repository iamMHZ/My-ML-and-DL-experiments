"""
Using numpy for implementing logistic regression
"""

import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression:

    def __init__(self):
        # load data
        self.x, self.y = self.load_data()
        # add a column for the bias (bias trick) ==> everything is vectorized
        ones_column = np.ones((self.x.shape[0], 1), np.float)
        self.x = np.append(self.x, ones_column, axis=1)

        self.y = self.y.reshape(self.y.shape[0], 1)

        # initialize weights randomly
        self.weights = np.random.random((self.x.shape[1], 1))

        self.losses = []

    def load_data(self):
        # https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival
        data_file = np.genfromtxt('../../../utils/datasets/supervised dataset/haberman.txt', delimiter=',')

        x = data_file[:, :2]
        y = data_file[:, 3]

        #  labels are 1 (survived) and 2 (died)
        # change 2 to 0

        y[y == 2] = 0

        return x, y

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_pred):
        # calculate loss

        y_pred[y_pred == 1] = 0.99  # helps not facing overflow

        #  Replace NaN with zero and infinity with large finite number
        # because the -log(x) and -log(1-x) have the tendency to return NaN or INF so we need to make it a number
        epoch_loss = (-self.y * np.nan_to_num(np.log(y_pred))) - ((1 - self.y) * np.nan_to_num(np.log(1 - y_pred)))
        epoch_loss = np.sum(epoch_loss)

        # making sure that the over all loss does not become INF
        epoch_loss = np.nan_to_num(epoch_loss)
        return epoch_loss

    def compute_gradient(self, y_pred):
        # calculate the gradient vector
        error = y_pred - self.y

        gradients = np.matmul(self.x.T, error)

        return gradients

    def fit(self, learning_rate=0.0001, epochs=50):
        for i in range(epochs):
            # make a prediction
            prediction = self.sigmoid(np.matmul(self.x, self.weights))
            # compute loss
            epoch_loss = self.compute_loss(prediction)

            # update the weights
            gradients = self.compute_gradient(prediction)
            self.weights += -learning_rate * gradients

            print(f'Epoch = {i} , Loss = {epoch_loss}')

            self.losses.append(epoch_loss)

        # plot the training loss
        plt.plot(np.arange(0, epochs), self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        return self.weights


if __name__ == '__main__':
    lr = LogisticRegression()
    lr.fit(learning_rate=0.000001, epochs=200)
