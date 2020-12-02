from abc import ABC, abstractmethod


class Activation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_activation(self, x):
        pass

    @abstractmethod
    def activation_derivatives(self):
        pass


class CostFunction:
    def __init__(self):
        pass
