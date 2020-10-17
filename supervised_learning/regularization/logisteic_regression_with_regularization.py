"""
implementation of Logistic regression with L2 regularization
"""
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    # https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival
    data_file = np.genfromtxt('../../utils/datasets/supervised dataset/haberman.txt', delimiter=',')

    X = data_file[:, :2]
    y = data_file[:, 3]

    #  labels are 1 (survived) and 2 (died)
    # change 2 to 0

    y[y == 2] = 0

    return X, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_loss(y_true, y_pred, weights, landa):
    # calculate loss
    epoch_loss = (-y_true * np.log(y_pred)) - ((1 - y_true) * np.log(1 - y_pred))
    epoch_loss = np.sum(epoch_loss)

    # add L2 regularization
    # W.T@W = sum(W^2)
    epoch_loss += 0.5 * landa * (np.matmul(weights.T, weights)[0])
    # No regularization on the bias so cancel it
    epoch_loss -= weights[0]**2

    #  Replace NaN with zero and infinity with large finite number
    # because the -log(x) and -log(1-x) have the tendency to return NaN or INF so we need to make it a number
    # making sure that the over all loss does not become INF
    epoch_loss = np.nan_to_num(epoch_loss)

    return epoch_loss


def compute_gradients(X, y_true, y_pred, weights, landa):
    error = y_pred - y_true

    # compute gradients
    gradients = np.matmul(X.T, error)

    # the regularization derivative too
    gradients = gradients + (landa * weights)

    # Dont apply regularization on the bias so, cancel it
    gradients[0] -= landa * weights[0]

    return gradients


def fit(X, y, learning_rate=0.0001, epochs=100, landa=0.01):
    # initialize the weights

    weights = np.random.random((X.shape[1], 1))

    losses = []

    for i in range(epochs):
        # make a prediction
        y_pred = sigmoid(np.matmul(X, weights))

        epoch_loss = compute_loss(y, y_pred, weights, landa)

        # update the wights
        gradients = compute_gradients(X, y, y_pred, weights, landa)
        weights += -learning_rate * gradients
        print(f'Epoch = {i} , Loss = {epoch_loss}')

        losses.append(epoch_loss)

    # plot the training loss
    plt.plot(np.arange(0, epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    print('Weights: ' + str(weights))


if __name__ == '__main__':
    #X, y = load_data()

    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=1000, n_features=2, centers=2,  random_state=14)

    # add a column for the bias (bias trick) ==> everything is vectorized
    ones_column = np.ones((X.shape[0], 1), np.float)
    X = np.append(ones_column, X, axis=1)

    y = y.reshape(y.shape[0], 1)

    fit(X, y, learning_rate=0.0001, epochs=100, landa=0)
