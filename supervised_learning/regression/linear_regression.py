"""
A very simple implementation of the regression using numpy
"""

import numpy as np
from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self):
        # keep track of loss of each epoch
        self.losses = []

        # load the specific dataset
        self.x, self.y = self.load_data()

    def load_data(self):
        # TODO change this method to be compatible with other dataset
        # https://www.kaggle.com/aungpyaeap/fish-market?select=Fish.csv
        data_file = np.genfromtxt('../../utils/datasets/supervised dataset/fish.csv', delimiter=',')

        x = data_file[1:, 1]  # weight
        y = data_file[1:, 2]  # vertical length

        plt.scatter(x, y)
        plt.show()

        return x, y

    def plot_hypothesis(self, predictions):
        # plot the predicted line
        plt.plot(self.x, predictions, c='r')
        # plot the data
        plt.scatter(self.x, self.y)
        plt.show()

    def compute_gradient(self, predictions):
        # compute the total loss
        temp = predictions - self.y
        total_loss = 0.5 * np.sum(temp ** 2)

        # compute the derivatives with respect to each variable(gradient)
        delta0 = np.sum(temp)
        delta1 = np.sum(temp * self.x)

        return total_loss, delta0, delta1

    # train the linear regression
    def fit(self, learning_rate, epochs=30):
        # initialize the parameters
        a = 0.0
        bias = 0.0

        # apply full batch gradient descent and update the parameters

        for i in range(epochs):
            predictions = (a * self.x) + bias

            loss, delta0, delta1 = self.compute_gradient(predictions)
            # add this epochs loss to the overall loss of the model for plotting the loss over time
            self.losses.append(loss)

            a += -learning_rate * delta1
            bias += -learning_rate * delta0

            self.plot_hypothesis(predictions)

            # time.sleep(2)

            print(f'Epoch {i}, loss {loss}')

        # plot the losses over the training
        plt.plot(np.arange(0, epochs), self.losses)
        plt.title('loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


if __name__ == '__main__':
    lr = LinearRegression()
    lr.fit(learning_rate=0.000000005, epochs=30)
