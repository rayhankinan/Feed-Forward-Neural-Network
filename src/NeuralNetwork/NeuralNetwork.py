import numpy as np
from . import Layer


class NeuralNetwork:
    def __init__(self, list_of_layer: list[Layer]) -> None:
        self.list_of_layer = list_of_layer

    def get_output(self, x: np.ndarray) -> np.ndarray:
        x_copy: np.ndarray = x.copy()

        for layer in self.list_of_layer:
            bias = np.array([1.])
            biased_x_copy = np.insert(x_copy, 0, bias, axis=0)
            x_copy = layer.get_output(biased_x_copy)

        return x_copy
