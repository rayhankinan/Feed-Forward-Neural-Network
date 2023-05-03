import numpy as np
from typing import NamedTuple
from . import Layer


class NeuralNetwork(NamedTuple):
    list_of_layer: list[Layer]

    def get_output(self, x: np.ndarray) -> np.ndarray:
        x_copy: np.ndarray = x.copy()

        for layer in self.list_of_layer:
            bias = np.ones(1)
            biased_x_copy = np.insert(x_copy, 0, bias, axis=0)
            x_copy = layer.get_output(biased_x_copy)

        return x_copy

    def get_batch_output(self, batch_x: np.ndarray) -> np.ndarray:
        N = batch_x.shape[0]
        batch_x_copy: np.ndarray = batch_x.copy()

        for layer in self.list_of_layer:
            bias = np.ones(N)
            biased_batch_x_copy = np.array(np.c_[bias, batch_x_copy])
            batch_x_copy = layer.get_batch_output(biased_batch_x_copy)

        return batch_x_copy

    def get_all_output(self, x: np.ndarray) -> list[np.ndarray]:
        x_copy: np.ndarray = x.copy()
        all_output: list[np.ndarray] = []

        for layer in self.list_of_layer:
            bias = np.ones(1)
            biased_x_copy = np.insert(x_copy, 0, bias, axis=0)
            x_copy = layer.get_output(biased_x_copy)
            all_output.append(x_copy)

        return all_output

    def get_all_batch_output(self, batch_x: np.ndarray) -> list[np.ndarray]:
        N = batch_x.shape[0]
        batch_x_copy: np.ndarray = batch_x.copy()
        all_output: list[np.ndarray] = []

        for layer in self.list_of_layer:
            bias = np.ones(N)
            biased_batch_x_copy = np.array(np.c_[bias, batch_x_copy])
            batch_x_copy = layer.get_batch_output(biased_batch_x_copy)
            all_output.append(batch_x_copy)

        return all_output

    def get_weight(self) -> list[np.ndarray]:
        return [layer.get_weight() for layer in self.list_of_layer]
