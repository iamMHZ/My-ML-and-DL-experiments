import numpy as np


class Perceptron:

    def __init__(self, n, alpha=0.1):
        self.alpha = alpha
        # 1 extra column for applying bias trick
        self.weights = np.random.randn(n + 1) / np.sqrt(n)

    def step_function(self, data):
        if data > 0:
            return 1
        return 0

    def fit(self, x, y, epochs=20):
        # apply bias trick:
        x = np.c_[x, np.ones((x.shape[0]))]

        for epoch in range(epochs):

            for (data, target) in zip(x, y):

                prediction = self.step_function(np.dot(data, self.weights))

                # apply weight update only if the predition was wrong
                if prediction != target:
                    #  delta rule:
                    error = prediction - target

                    self.weights += -self.alpha * error * data

                    print(f'[training] uprating weight matrix EPOCH {epoch} , WEIGHTS = {self.weights}')

        print('\nTRAINING FINISHED')
        print("################################################################################\n")

    def predict(self, x, add_bios=True):
        # print(x)
        # make sure that data (x) is a 2d array
        x = np.atleast_2d(x)
        # print(x)

        # check if we need to add bios :

        if add_bios:
            x = np.c_[x, np.ones((x.shape[0]))]

        perdition = self.step_function(np.dot(x, self.weights))

        return perdition
