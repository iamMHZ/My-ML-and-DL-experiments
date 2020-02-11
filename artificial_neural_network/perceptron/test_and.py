from artificial_neural_network.perceptron.perceptron import Perceptron

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [0], [0], [1]]

perceptron = Perceptron(2)

perceptron.fit(x, y)
perceptron.predict(x)
