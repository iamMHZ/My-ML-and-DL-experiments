from artificial_neural_network.perceptron.perceptron import Perceptron
import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

perceptron = Perceptron(2)

perceptron.fit(x, y)
for (data, target) in zip(x, y):
    prediction = perceptron.predict(data)
    print(f"[TESTING] data={data}, ground-truth={target}, prediction={prediction}")
