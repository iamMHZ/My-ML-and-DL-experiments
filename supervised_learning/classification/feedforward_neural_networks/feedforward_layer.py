import numpy as np

from supervised_learning.classification.feedforward_neural_networks.activations_and_costs import ActivationFunction


class FeedForwardLayer:
    def __init__(self, num_neurons, activation_function: ActivationFunction):
        self.num_neurons = num_neurons
        self.activation_function = activation_function

        # INITIALIZE
        self.layer_weighted_input = np.zeros(shape=(self.num_neurons, 1))
        self.layer_activations = np.zeros(shape=(self.num_neurons, 1))
        # TODO how to determine the shape of weights dynamically
        self.layer_weights = None
        self.layer_biases = np.zeros(shape=(self.num_neurons, 1))

    def forward(self, data):
        self.layer_weighted_input = np.matmul(data, self.layer_weights)
        self.layer_activations = self.activation_function.make_activation(self.layer_weighted_input)

    def backward(self):
        pass
