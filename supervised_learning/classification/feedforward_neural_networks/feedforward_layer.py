import numpy as np

from supervised_learning.classification.feedforward_neural_networks.activations_and_costs import ActivationFunction


class FeedForwardLayer:
    def __init__(self, num_neurons, activation_function: ActivationFunction):
        self.num_neurons = num_neurons
        self.activation_function = activation_function

        # INITIALIZE
        self.layer_weighted_input = None  # OR Zeroes ?
        self.layer_activations = None  # OR Zeroes ?
        # TODO how to determine the shape of weights dynamically
        # TODO  the if in self.forward is doing it but is there a better way
        self.layer_weights = None
        self.layer_biases = np.random.rand(1, self.num_neurons)

        self.previous_activation = None
        self.layer_error = None

    def forward(self, previous_activation):
        self.previous_activation = previous_activation

        # initialize weights
        if self.layer_weights is None:
            # TODO is there any other way of doing the initialization here
            # TODO like using model class or sth else
            # shape of weights is number of features in previous activation * number of neurons of this layer
            self.layer_weights = np.random.rand(previous_activation.shape[1], self.num_neurons)

        self.layer_weighted_input = np.matmul(previous_activation, self.layer_weights)
        self.layer_weighted_input = self.layer_weighted_input + self.layer_biases
        self.layer_activations = self.activation_function.make_activation(self.layer_weighted_input)

        return self.layer_activations

    def backward(self, up_coming_error, learning_rate):
        # TODO check and complete backward pass
        # TODO separate the optimization part
        self.layer_error = up_coming_error * self.activation_function.activation_derivatives(self.layer_weighted_input)

        # Update the weights and biases

        self.update_weights(learning_rate)

        # (W layer + 1.T @ delta layer + 1)
        self.layer_error = self.layer_error @ self.layer_weights.T

        return self.layer_error

    def update_weights(self, learning_rate):
        self.layer_biases += -learning_rate * np.sum(self.layer_error, axis=0)
        self.layer_weights += -learning_rate * (self.previous_activation.T @ self.layer_error)
