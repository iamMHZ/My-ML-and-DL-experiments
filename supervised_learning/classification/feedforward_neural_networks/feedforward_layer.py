import numpy as np


class FeedForwardLayer:
    def __init__(self, num_neurons, activation_function):
        self.num_neurons = num_neurons
        self.activation_function = activation_function

        # INITIALIZE
        self.weighted_input = np.zeros(shape=(self.num_neurons, 1))
        self.layer_activations = np.zeros(shape=(self.num_neurons, 1))
        # TODO how to determine the shape of weights dynamically
        self.weights = None
        self.biases = np.zeros(shape=(self.num_neurons, 1))

    def forward(self):
        pass

    def backward(self):
        pass
