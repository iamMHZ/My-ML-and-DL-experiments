from abc import ABC, abstractmethod

import numpy as np


class CostFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def cost(self, y_true, y_pred):
        pass

    @abstractmethod
    def cost_derivatives(self, y_true, y_pred):
        pass


class ActivationFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_activation(self, x):
        pass

    @abstractmethod
    def activation_derivatives(self, x):
        pass


class SigmoidActivationFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    def make_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivatives(self, x):
        # The derivative of sigmoid is : sigmoid(x) * (1-sigmoid(x))
        return self.make_activation(x) * (1 - self.make_activation(x))


class SoftmaxActivationFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    def make_activation(self, x):
        pass

    def activation_derivatives(self, x):
        pass


class MSECost(CostFunction):
    def __init__(self):
        super().__init__()

    def cost(self, y_true, y_pred):
        return 1 / len(y_true) * (np.sum((y_true - y_pred)) ** 2)

    def cost_derivatives(self, y_true, y_pred):
        return 2 / len(y_true) * np.sum(y_true - y_pred)
