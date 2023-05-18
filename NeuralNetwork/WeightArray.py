import numpy as np
from typing import Any
from Serialization.Compilable import get_single, get_batch, add_batch
from . import ActivationFunction


class WeightArray:
    array_of_weight: np.ndarray[Any, np.dtype[np.float64]]
    activation_function: ActivationFunction

    def __init__(self, array_of_weight: np.ndarray, activation_function: ActivationFunction) -> None:
        self.array_of_weight = array_of_weight
        self.activation_function = activation_function

    def get_output(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        weighted_x = get_single(self.array_of_weight, x)
        activated_x = self.activation_function.get_output(weighted_x)
        return activated_x

    def get_batch_output(self, batch_x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        weighted_batch_x = get_batch(self.array_of_weight, batch_x)
        activated_batch_x = self.activation_function.get_output(
            weighted_batch_x
        )
        return activated_batch_x

    def get_weight(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self.array_of_weight

    def add_weight(self, delta_weight: np.ndarray[Any, np.dtype[np.float64]]) -> None:
        self.array_of_weight = add_batch(self.array_of_weight, delta_weight)
