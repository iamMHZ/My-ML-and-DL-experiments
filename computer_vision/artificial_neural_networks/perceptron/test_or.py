import numpy as np
from computer_vision.artificial_neural_networks.perceptron.perceptron import Perceptron

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

perceptron = Perceptron(x.shape[1])
perceptron.fit(x, y)

# now that our network is trained, loop over the data points
for (data, target) in zip(x, y):
    prediction = perceptron.predict(data)

    print(f"[TESTING] data={data}, ground-truth={target}, prediction={prediction}")
