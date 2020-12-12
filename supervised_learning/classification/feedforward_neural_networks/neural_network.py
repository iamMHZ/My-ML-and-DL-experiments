import numpy as np

from supervised_learning.classification.feedforward_neural_networks.activations_and_costs import CostFunction


class FeedForwardNeuralNetwork:

    def __init__(self, num_classes, input_shape, cost_fuction: CostFunction):
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.layers = []

        self.cost_function = cost_fuction

        self.losses = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, train_x, train_y, epochs=10, learning_rate=0.01):
        # TODO check backward pass

        for j in range(epochs):

            # for the first layer
            activation = self.layers[0].forward(train_x)
            # TODO remove duplicate code
            # from the second layer to the last layer do the forward pass

            # Forward
            for i in range(1, len(self.layers)):
                activation = self.layers[i].forward(activation)

            epoch_loss = self.cost_function.cost(train_y, activation)
            print('Epoch {}, Loss {} '.format(j, epoch_loss))
            # for plotting purposes
            self.losses.append(epoch_loss)

            # backward pass

            up_coming_error = self.cost_function.cost_derivatives(train_y, activation) * \
                              self.layers[-1].activation_function.activation_derivatives(
                                  self.layers[-1].layer_weighted_input)

            # update last layer weights and biases
            self.layers[-1].layer_error = up_coming_error

            self.layers[-1].layer_biases += -learning_rate * np.sum(self.layers[-1].layer_error, axis=0)
            self.layers[-1].layer_weights += -learning_rate * (
                    self.layers[-1].previous_activation.T @ self.layers[-1].layer_error)

            # (W layer + 1 .T @ delta layer + 1 )
            last_layer_weights = self.layers[-1].layer_weights.T

            up_coming_error = up_coming_error @ last_layer_weights

            for i in range(-2, -len(self.layers) - 1, -1):
                up_coming_error = self.layers[i].backward(up_coming_error, learning_rate)

    def evaluate(self, data):
        # uses only the forward pass

        # for the first layer
        activation = self.layers[0].forward(data)

        # from the second layer to the last layer do the forward pass
        for i in range(1, len(self.layers)):
            activation = self.layers[i].forward(activation)

        return activation

    def predict(self):
        pass

    def on_training_ended(self):
        pass
