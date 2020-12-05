import numpy as np

from supervised_learning.classification.feedforward_neural_networks.activations_and_costs import \
    SigmoidActivationFunction
from supervised_learning.classification.feedforward_neural_networks.feedforward_layer import FeedForwardLayer
from supervised_learning.classification.feedforward_neural_networks.neural_network import FeedForwardNeuralNetwork

model = FeedForwardNeuralNetwork(1, 3)

# create the model
model.add_layer(FeedForwardLayer(3, SigmoidActivationFunction()))
model.add_layer(FeedForwardLayer(2, SigmoidActivationFunction()))
model.add_layer(FeedForwardLayer(1, SigmoidActivationFunction()))

data = np.random.randint(0, 10, size=(120, 3))
labels = np.random.randint(0, 2, size=(120, 1))

model.train(data, epochs=2)

print(model.evaluate(data[0]))
