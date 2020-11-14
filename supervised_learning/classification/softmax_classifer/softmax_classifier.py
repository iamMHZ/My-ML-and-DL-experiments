"""
Softmax classifier
"""

import numpy as np
from matplotlib import pyplot as plt


class SoftmaxClassifier:

    def __init__(self, num_classes, num_features, encode_labels=True):
        """
        Softmax classifier
        :param num_classes: number of the classes in data
        :param num_features: dimensionality of  data
        :param encode_labels: a flag that show if labels are hot-encoded or not
        """
        self.num_features = num_features
        self.num_classes = num_classes

        self.encode_labels = encode_labels

        # initialize weights
        self.weights = np.zeros(shape=(num_features, num_classes))

        self.losses = []

    def fit(self, train_x, train_y, epochs, learning_rate):
        """
        Train the softmax classifier
        :param train_x: training data
        :param train_y: training labels
        :param epochs: number of iterations of the optimization algorithm
        :param learning_rate: the step size of each iteration of the optimization algorithm
        :return:
        """
        if self.encode_labels:
            train_y = self.hot_encode(train_y)

        for i in range(epochs):
            # make a prediction
            predictions = self.softmax(np.matmul(train_x, self.weights))

            epoch_loss = self.compute_loss(predictions, train_y)
            self.losses.append(epoch_loss)

            print(f'Epoch = {i} , Loss = {epoch_loss}')

            gradients = self.compute_gradients(train_x, train_y, predictions)
            self.weights += -learning_rate * gradients

        self.plot(epochs)

    def predict(self, test_x):
        """
        Make a prediction on the test data
        :param test_x: test data
        :return: probabilities
        """
        return np.matmul(test_x, self.weights)

    def softmax(self, scores):
        """
        Softmax function
        :param scores: score vector (a matrix in multi-class problems)
        :return: probabilities
        """
        exps = np.exp(scores)

        # calculate the denominator of the softmax equation
        sums = np.sum(exps, axis=1)
        sums = sums.reshape(sums.shape[0], 1)
        # repeat the columns of the sums vector as each exps
        # must be divided by the sum of each prediction vector
        # and in terms of the multi-class classification each row
        # has self.num_classes of elements for each data point
        sums = np.repeat(sums, self.num_classes, axis=1)

        probabilities = exps / sums

        return probabilities

    def compute_loss(self, y_pred, y_true):
        """
        Computes softmax loss
        :param y_pred: predictions ( probabilities)
        :param y_true:  true labels
        :return: sum of losses
        """
        epoch_loss = -np.log(y_pred) * y_true

        return np.sum(epoch_loss)

    def compute_gradients(self, X, y_true, y_pred):
        """
        Full batch gradient descent
        :param X: training data
        :param y_pred: predictions ( probabilities)
        :param y_true:  true labels
        :return: gradients
        """
        gradients = np.matmul(X.T, (y_pred - y_true))

        return gradients

    def hot_encode(self, labels):
        """
        Hot encodes labels
        :param labels: trai
        :return: hot-encoded labels
        """
        encoded_labels = np.zeros(shape=(labels.shape[0], self.num_classes))

        for i in range(labels.shape[0]):
            encoded_labels[i, labels[i]] = 1

        return encoded_labels

    def plot(self, epochs):
        """
        Plots loss curve of the trained model
        :param epochs: number of iterations of the optimization algorithm
        :return: None
        """
        plt.plot(np.arange(0, epochs), self.losses)
        plt.show()


if __name__ == '__main__':
    NUM_CLASSES = 3
    NUM_FEATURES = 2

    # make dummy data with sklearn and plot them
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=5000, n_features=NUM_FEATURES, centers=NUM_CLASSES, random_state=42)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    # add a column for the bias (bias trick) ==> everything is vectorized
    ones_column = np.ones((X.shape[0], 1), np.float)
    X = np.append(X, ones_column, axis=1)
    y = y.reshape(y.shape[0], 1)

    # num_features=NUM_FEATURES +1 : +1 is the bias
    softmax_classifier = SoftmaxClassifier(num_classes=NUM_CLASSES, num_features=NUM_FEATURES + 1)

    softmax_classifier.fit(X, y, epochs=100, learning_rate=0.001)
