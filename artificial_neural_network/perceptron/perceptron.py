import numpy as np


class Perceptron:

    def __init__(self, n, alpha=0.01):
        self.alpha = alpha
        # 1 extra column for applying bias trick
        self.weights = np.random.randn(n + 1) / np.sqrt(n)

    def step_function(self, data):
        if data > 0:
            return 1
        return 0

    def fit(self, x, y, epochs=20):
        # apply bias trick:
        x = np.c_(np.ones(x.shape[0]))

        for epoch in epochs:

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

        # check if we need to add bios :

        if add_bios:
            x = np.c_(np.ones(x.shape[0]))

        perdition = self.step_function(np.dot(x, self.weights))

        return perdition
