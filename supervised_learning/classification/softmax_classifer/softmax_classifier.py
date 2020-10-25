"""
Softmax classifier
"""

import numpy as np


class SoftmaxClassifier:

    def __init__(self, num_classes, num_features, encode_labels=True):
        self.num_features = num_features
        self.num_classes = num_classes

        self.encode_labels = encode_labels

        # initialize weights
        self.weights = np.zeros(shape=(num_features, num_classes))

        self.losses = []

    def fit(self, train_x, train_y, epochs, learning_rate):
        if self.encode_labels:
            train_y = self.hot_encode(train_y)

        for i in range(epochs):
            # make a prediction
            predictions = self.softmax(np.matmul(train_x, self.weights))

    def predict(self):
        pass

    def softmax(self, data):
        """
        Softmax function
        :param data: score vector (matrics in multi-class problems)
        :return: probabilities
        """
        exps = np.exp(data)

        exps = exps / np.sum(exps, axis=1)

        return exps

    def loss(self, y_pred, y_true):
        epoch_loss = -np.log(y_pred) * y_true

        return np.sum(epoch_loss)

    def get_gradients(self):
        pass

    def hot_encode(self, labels):
        encoded_labels = np.zeros(shape=(labels.shape[0], self.num_classes))

        for i in range(labels.shape[0]):
            encoded_labels[i, labels[i]] = 1

        return encoded_labels

    def plot(self):
        pass


if __name__ == '__main__':
    NUM_CLASSES = 3
    INPUT_SHAPE = (3,)

    # X, y = make_blobs(n_samples=5000, n_features=NUM_FEATURES , centers=NUM_CLASSES, random_state=42)
    # color = ['red', 'green', 'blue']
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # # plt.scatter(X, y, c=y)
    # plt.show()

    softmax_classifier = SoftmaxClassifier(num_classes=3, num_features=(4,))
