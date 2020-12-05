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
        # TODO  the if in self.forward is doing it but is there a better way
        self.layer_weights = None
        self.layer_biases = np.zeros(shape=(self.num_neurons, 1))

    def forward(self, previouse_activation):
        # initialize weights
        if self.layer_weights is None:
            # TODO is there any other way of doing the initialization here
            # TODO like using model class or sth else
            self.layer_weights = np.zeros(shape=(previouse_activation.shape[1], self.num_neurons))

        self.layer_weighted_input = np.matmul(previouse_activation, self.layer_weights)
        self.layer_activations = self.activation_function.make_activation(self.layer_weighted_input)

        return self.layer_activations

    def backward(self):
        pass
