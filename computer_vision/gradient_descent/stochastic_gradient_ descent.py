from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


def sigmoid_activation(x):
    sigmoid = 1.0 / (1 + np.exp(-x))

    return sigmoid


def predict(x, w):
    prediction = sigmoid_activation(x.dot(w))

    prediction[prediction <= 0.5] = 0
    prediction[prediction > 0] = 1

    return prediction


def get_color(color):
    if color == 1:
        return 'black'

    return 'brown'


def next_batch(x, y, batch_size):
    for i in range(0, x.shape[0], batch_size):
        yield (x[i:i + batch_size], y[i:i + batch_size])


def main():
    epochs = 100
    batch_size = 64
    alpha = 0.01

    (x, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
    y = y.reshape(y.shape[0], 1)
    x = np.c_[x, np.ones((x.shape[0]))]

    (train_x, test_x, train_y, test_y) = train_test_split(x, y, test_size=0.5, random_state=42)

    print("[info] training...")
    w = np.random.randn(x.shape[1], 1)

    losses = []

    for epoch in range(epochs):
        epoch_loss = []

        for batch_x, batch_y in next_batch(x, y, batch_size):
            prediction = sigmoid_activation(batch_x.dot(w))

            error = prediction - batch_y
            epoch_loss.append(np.sum(error ** 2))

            gradient = batch_x.T.dot(error)

            w += -alpha * gradient

        loss = np.average(epoch_loss)
        losses.append(loss)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print("epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

    print('evaluating ...')

    prediction = predict(test_x, w)

    print(classification_report(test_y, prediction))

    print('plotting...')
    plt.style.use('ggplot')
    plt.figure()
    plt.title("Data")
    # plt.scatter(test_x[:, 0], test_x[:, 1], marker="o", c=test_y, s=30)
    # plt.scatter(test_x[:, 0], test_x[:, 1], marker="o", s=30)
    for i in range(len(test_x)):
        # s is  the area of the marker.
        plt.scatter(test_x[i, 0], test_x[i, 1], marker="o", c=get_color(test_y[i, 0]), s=20)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()
