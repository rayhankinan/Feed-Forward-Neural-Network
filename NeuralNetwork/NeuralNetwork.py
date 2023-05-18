import numpy as np
from typing import Any
from Serialization.Compilable import insert_single_bias, insert_batch_bias
from . import Layer


class NeuralNetwork:
    list_of_layer: list[Layer]

    def __init__(self, list_of_layer: list[Layer]) -> None:
        self.list_of_layer = list_of_layer

    def get_output(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        x_copy: np.ndarray[Any, np.dtype[np.float64]] = x.copy()

        for layer in self.list_of_layer:
            biased_x_copy = insert_single_bias(x_copy)
            x_copy = layer.get_output(biased_x_copy)

        return x_copy

    def get_batch_output(self, batch_x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        batch_x_copy: np.ndarray[Any, np.dtype[np.float64]] = batch_x.copy()

        for layer in self.list_of_layer:
            biased_batch_x_copy = insert_batch_bias(batch_x_copy)
            batch_x_copy = layer.get_batch_output(biased_batch_x_copy)

        return batch_x_copy

    def get_all_output(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
        x_copy: np.ndarray[Any, np.dtype[np.float64]] = x.copy()
        all_output: list[np.ndarray[Any, np.dtype[np.float64]]] = []

        for layer in self.list_of_layer:
            biased_x_copy = insert_single_bias(x_copy)
            x_copy = layer.get_output(biased_x_copy)
            all_output.append(x_copy)

        return all_output

    def get_all_batch_output(self, batch_x: np.ndarray[Any, np.dtype[np.float64]]) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
        batch_x_copy: np.ndarray[Any, np.dtype[np.float64]] = batch_x.copy()
        all_output: list[np.ndarray[Any, np.dtype[np.float64]]] = []

        for layer in self.list_of_layer:
            biased_batch_x_copy = insert_batch_bias(batch_x_copy)
            batch_x_copy = layer.get_batch_output(biased_batch_x_copy)
            all_output.append(batch_x_copy)

        return all_output

    def get_weight(self) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
        return [layer.get_weight() for layer in self.list_of_layer]
