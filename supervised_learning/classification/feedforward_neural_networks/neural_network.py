class FeedForwardNeuralNetwork:

    def __init__(self, num_classes, input_shape):
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, train_x, epochs=1, learning_rate=0.01):
        # TODO add backward pass

        for j in range(epochs):

            # for the first layer
            activation = self.layers[0].forward(train_x)
            # TODO remove duplicate code
            # from the second layer to the last layer do the forward pass
            for i in range(1, len(self.layers)):
                activation = self.layers[i].forward(activation)

    def predict(self, data):
        # uses only the forward pass

        # for the first layer
        activation = self.layers[0].forward(data)

        # from the second layer to the last layer do the forward pass
        for i in range(1, len(self.layers)):
            activation = self.layers[i].forward(activation)

        return activation.argmax(axis=1)

    def evaluate(self):
        pass
