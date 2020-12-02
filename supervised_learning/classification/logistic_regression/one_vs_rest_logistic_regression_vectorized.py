"""
Vectorized implementation of the one versus rest logistic regression classifier

"""

import matplotlib.pyplot as plt
import numpy as np


class LogisticRegressionOVR:

    def __init__(self, num_classes, num_features, are_labels_encoded=False):
        self.num_classes = num_classes
        self.num_features = num_features
        self.are_labels_encoded = are_labels_encoded

        # initialize weights for classifiers
        self.weights = np.zeros(shape=(num_features, num_classes))

        self.losses = []

    def fit(self, train_x, train_y, epochs, learning_rate):

        # if labels are not encoded
        if self.are_labels_encoded is False:
            train_y = self.encode_labels(train_y)

        for i in range(epochs):
            # let each classifier to make a prediction
            prediction = self.sigmoid(np.matmul(train_x, self.weights))

            # these comments are the non-vectorized form
            # for j in range(self.num_classes):
            #     prediction_j = self.sigmoid(np.matmul(train_x, self.weights[:, j]))
            #
            #     prediction[:, j] = prediction_j.copy()

            # compute loss
            epoch_loss = self.compute_loss(prediction, train_y)

            # update the weights
            gradients = self.compute_gradients(train_x, y_pred=prediction, y_true=train_y)
            self.weights += -learning_rate * gradients

            print(f'Epoch = {i} , Loss = {epoch_loss}')

            self.losses.append(epoch_loss)

        # plot the training loss
        plt.plot(np.arange(0, epochs), self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def compute_loss(self, y_pred, y_true):

        y_pred[y_pred == 1] = 0.99  # helps not facing overflow

        #  Replace NaN with zero and infinity with large finite number
        # because the -log(x) and -log(1-x) have the tendency to return NaN or INF so we need to make it a number
        epoch_loss = (-y_true * np.nan_to_num(np.log(y_pred))) - ((1 - y_true) * np.nan_to_num(np.log(1 - y_pred)))
        epoch_loss = np.sum(epoch_loss)

        # making sure that the over all loss does not become INF
        epoch_loss = np.nan_to_num(epoch_loss)
        return epoch_loss

    def compute_gradients(self, X, y_pred, y_true):
        # calculate the gradient vector
        error = y_pred - y_true

        gradients = np.matmul(X.T, error)

        return gradients

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def encode_labels(self, old_labels):

        new_labels = np.zeros(shape=(old_labels.shape[0], self.num_classes))

        for i in range(old_labels.shape[0]):
            new_labels[i][old_labels[i]] = 1

        return new_labels

    def predict(self, test_x):
        return np.matmul(test_x, self.weights)


if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    NUM_CLASSES = 3
    NUM_FEATURES = 2

    # make dummy data with sklearn and plot them
    X, y = make_blobs(n_samples=5000, n_features=NUM_FEATURES, centers=NUM_CLASSES, random_state=42)
    color = ['red', 'green', 'blue']
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    # add a column for the bias (bias trick) ==> everything is vectorized
    ones_column = np.ones((X.shape[0], 1), np.float)
    X = np.append(X, ones_column, axis=1)
    y = y.reshape(y.shape[0], 1)

    # +1 is the bias
    ovr = LogisticRegressionOVR(num_classes=NUM_CLASSES, num_features=NUM_FEATURES + 1)

    ovr.fit(X, y, epochs=300, learning_rate=0.0001)
