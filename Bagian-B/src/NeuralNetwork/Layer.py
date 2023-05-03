from __future__ import annotations
import numpy as np
from typing import NamedTuple
from . import Row, ActivationFunction


class Layer(NamedTuple):
    list_of_row: list[Row]
    activation_function: ActivationFunction

    def get_output(self, x: np.ndarray) -> np.ndarray:
        array_of_weight = np.array(
            [row.weight for row in self.list_of_row]
        )
        weighted_x = np.array(np.dot(array_of_weight.T, x))
        activated_x = self.activation_function(weighted_x)

        return activated_x

    def get_batch_output(self, batch_x: np.ndarray) -> np.ndarray:
        array_of_weight = np.array(
            [row.weight for row in self.list_of_row]
        )
        weighted_batch_x = np.array(np.dot(batch_x, array_of_weight))
        activated_batch_x = self.activation_function(weighted_batch_x)

        return activated_batch_x

    def get_weight(self) -> np.ndarray:
        return np.array(
            [row.weight for row in self.list_of_row]
        )

    def get_updated_weight(self, delta_weight: np.ndarray) -> Layer:
        array_of_weight = np.array(
            [row.weight for row in self.list_of_row]
        )

        new_array_of_weight = np.add(array_of_weight, delta_weight)
        list_of_row: list[Row] = [
            Row(weight) for weight in new_array_of_weight
        ]

        return Layer(list_of_row, self.activation_function)
