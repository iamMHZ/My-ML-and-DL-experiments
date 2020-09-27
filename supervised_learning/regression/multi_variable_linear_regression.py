import numpy as np
from matplotlib import pyplot as plt


def load_data():
    # https://www.kaggle.com/aungpyaeap/fish-market?select=Fish.csv
    data_file = np.genfromtxt('../../utils/datasets/supervised dataset/fish.csv', delimiter=',')

    X = data_file[1:, 5:]  # height and width of fishes
    y = data_file[1:, 1]  # weight of fishes

    print(X[:5, :])
    print(y[:5])

    return X, y


def compute_gradient(X, y_pred, y_true):
    errors = y_pred - y_true

    loss = 0.5 * np.sum(errors ** 2)

    gradients = np.matmul(errors, X)

    return loss, gradients


def fit(X, y, learning_rate=0.001, epochs=30):  # full batch gradient descent

    # initialize the weights randomly
    weights = np.random.random((1, X.shape[1]))

    losses = []
    for i in range(epochs):
        # make a prediction
        y_pred = np.matmul(weights, X.transpose())

        # compute the loss and the gradients
        loss, gradients = compute_gradient(X, y_pred=y_pred, y_true=y)

        # update the weights
        weights += -learning_rate * gradients

        print(f'EPOCH {i}, LOSS {loss}')

        losses.append(loss)

    plt.plot(np.arange(0, epochs), losses)
    plt.show()


if __name__ == '__main__':
    X, y = load_data()

    # add an extra column for the bias this way you don't need to compute the gradients separately
    ones_column = np.ones((X.shape[0], 1), np.float)
    X = np.append(X, ones_column, axis=1)

    y = y.reshape(1, y.shape[0])

    fit(X,y,learning_rate= 0.00001,epochs=30)
