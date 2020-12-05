from supervised_learning.classification.feedforward_neural_networks.activations_and_costs import CostFunction


class FeedForwardNeuralNetwork:

    def __init__(self, num_classes, input_shape, cost_fuction: CostFunction):
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.layers = []

        self.cost_function = cost_fuction

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, train_x, train_y, epochs=1, learning_rate=0.01):
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

            # backward pass

            up_coming_error = self.cost_function.cost_derivatives(train_y, activation) * \
                              self.layers[-1].activation_function.activation_derivatives(
                                  self.layers[-1].layer_weighted_input)

            up_coming_error =  up_coming_error @ self.layers[-1].layer_weights.T

            for i in range(-len(self.layers) + 1, 0):
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
