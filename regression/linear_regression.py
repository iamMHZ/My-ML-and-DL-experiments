import numpy as np
from matplotlib import pyplot as plt


def load_data():
    # https://www.kaggle.com/aungpyaeap/fish-market?select=Fish.csv
    data_file = np.genfromtxt('../utils/datasets/regression dataset/fish.csv', delimiter=',')

    X = data_file[1:, 1]  # weight
    y = data_file[1:, 2]  # vertical length

    plt.scatter(X, y)
    plt.show()

    return X, y


def plot(x, y, predictions):
    # plot the predicted line
    plt.plot(x, predictions, c='r')
    # plot the data
    plt.scatter(x, y)
    plt.show()


def compute_gradient(X, y, predictions):
    # compute the total loss
    total_loss = np.sum(0.5 * ((predictions - y) ** 2))

    # compute the derivatives with respect to each variable(gradient)
    delta0 = np.sum(predictions - y)
    delta1 = np.sum((predictions - y) * X)

    return total_loss, delta0, delta1


# train the linear regression
def fit(X, y, learning_rate, epochs=30):
    # initialize the parameters
    a = 0.0
    bias = 0.0

    # apply full batch gradient descent and update the parameters
    for i in range(epochs):
        predictions = (a * X) + bias

        loss, delta0, delta1 = compute_gradient(X, y, predictions)

        a += -learning_rate * delta1
        bias += -learning_rate * delta0

        plot(X, y, predictions)

        # time.sleep(2)

        print(f'Epoch {i}, loss {loss}')


if __name__ == '__main__':
    X, y = load_data()

    print(X.shape)
    print(X[0])
    print(type(X))
    print(y.shape)

    # X = np.array([0,1])
    # y = np.array([0,1])
    # fit(X, y, learning_rate=0.05, epochs=300)

    fit(X, y, learning_rate=0.000000005, epochs=30)
