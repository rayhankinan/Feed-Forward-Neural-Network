import numpy as np
from typing import NamedTuple
from . import Perceptron, ActivationFunction


class Layer(NamedTuple):
    list_of_perceptron: list[Perceptron]
    activation_function: ActivationFunction

    def get_output(self, x: np.ndarray) -> np.ndarray:
        array_of_weight = np.array(
            [perceptron.weight for perceptron in self.list_of_perceptron]
        )
        weighted_x = np.array(np.dot(array_of_weight, x))
        activated_x = self.activation_function(weighted_x)

        return activated_x
