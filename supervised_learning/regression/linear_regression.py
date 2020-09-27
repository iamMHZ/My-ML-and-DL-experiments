import numpy as np
from matplotlib import pyplot as plt


def load_data():
    # https://www.kaggle.com/aungpyaeap/fish-market?select=Fish.csv
    data_file = np.genfromtxt('../../utils/datasets/supervised dataset/fish.csv', delimiter=',')

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
    temp = predictions - y
    total_loss = 0.5 * np.sum(temp ** 2)

    # compute the derivatives with respect to each variable(gradient)
    delta0 = np.sum(temp)
    delta1 = np.sum((temp) * X)

    return total_loss, delta0, delta1


# train the linear regression
def fit(X, y, learning_rate, epochs=30):
    # initialize the parameters
    a = 0.0
    bias = 0.0

    # apply full batch gradient descent and update the parameters
    losses = []

    for i in range(epochs):
        predictions = (a * X) + bias

        loss, delta0, delta1 = compute_gradient(X, y, predictions)
        # add this epochs loss to the overall loss of the model for plotting the loss over time
        losses.append(loss)

        a += -learning_rate * delta1
        bias += -learning_rate * delta0

        plot(X, y, predictions)

        # time.sleep(2)

        print(f'Epoch {i}, loss {loss}')

    # plot the losses over the training
    plt.plot(np.arange(0, epochs), losses)
    plt.title('loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


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
