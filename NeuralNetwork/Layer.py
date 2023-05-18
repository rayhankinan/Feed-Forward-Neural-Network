from . import ActivationFunction


class Layer:
    num_of_perceptron: int
    activation_function: ActivationFunction

    def __init__(self, num_of_perceptron: int, activation_function: ActivationFunction) -> None:
        self.num_of_perceptron = num_of_perceptron
        self.activation_function = activation_function
