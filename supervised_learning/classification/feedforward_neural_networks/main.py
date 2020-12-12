import numpy as np

from supervised_learning.classification.feedforward_neural_networks.activations_and_costs import MSECost
from supervised_learning.classification.feedforward_neural_networks.activations_and_costs import \
    SigmoidActivationFunction
from supervised_learning.classification.feedforward_neural_networks.feedforward_layer import FeedForwardLayer
from supervised_learning.classification.feedforward_neural_networks.neural_network import FeedForwardNeuralNetwork

cost_function = MSECost()

model = FeedForwardNeuralNetwork(1, 3, cost_function)

# create the model
model.add_layer(FeedForwardLayer(3, SigmoidActivationFunction()))
model.add_layer(FeedForwardLayer(2, SigmoidActivationFunction()))
model.add_layer(FeedForwardLayer(1, SigmoidActivationFunction()))

data = np.random.randint(0, 10, size=(120, 3))
labels = np.random.randint(0, 2, size=(120, 1))

# TODO hot encode labels
model.train(data, labels, epochs=100, learning_rate=0.0001)

# print(model.evaluate(data[0]))
