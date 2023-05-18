import numpy as np
from typing import Any
from Serialization.Compilable import insert_single_bias, insert_batch_bias
from Backpropagation import Backpropagation
from . import WeightArray, Layer, ErrorFunction


class NeuralNetwork:
    list_of_weight_array: list[WeightArray]
    error_function: ErrorFunction

    def __init__(self, input_size: int, list_of_layer: list[Layer], error_function: ErrorFunction) -> None:
        prev_input = input_size
        self.list_of_weight_array = []
        self.error_function = error_function

        for layer in list_of_layer:
            weight_array = WeightArray(
                np.random.uniform(
                    -0.05,
                    0.05,
                    (prev_input + 1, layer.num_of_perceptron)
                ),
                layer.activation_function
            )
            self.list_of_weight_array.append(weight_array)
            prev_input = layer.num_of_perceptron

    def get_output(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        x_copy: np.ndarray[Any, np.dtype[np.float64]] = x.copy()

        for weight_array in self.list_of_weight_array:
            biased_x_copy = insert_single_bias(x_copy)
            x_copy = weight_array.get_output(biased_x_copy)

        return x_copy

    def get_batch_output(self, batch_x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        batch_x_copy: np.ndarray[Any, np.dtype[np.float64]] = batch_x.copy()

        for weight_array in self.list_of_weight_array:
            biased_batch_x_copy = insert_batch_bias(batch_x_copy)
            batch_x_copy = weight_array.get_batch_output(biased_batch_x_copy)

        return batch_x_copy

    def get_all_output(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
        x_copy: np.ndarray[Any, np.dtype[np.float64]] = x.copy()
        all_output: list[np.ndarray[Any, np.dtype[np.float64]]] = []

        for weight_array in self.list_of_weight_array:
            biased_x_copy = insert_single_bias(x_copy)
            x_copy = weight_array.get_output(biased_x_copy)
            all_output.append(x_copy)

        return all_output

    def get_all_batch_output(self, batch_x: np.ndarray[Any, np.dtype[np.float64]]) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
        batch_x_copy: np.ndarray[Any, np.dtype[np.float64]] = batch_x.copy()
        all_output: list[np.ndarray[Any, np.dtype[np.float64]]] = []

        for weight_array in self.list_of_weight_array:
            biased_batch_x_copy = insert_batch_bias(batch_x_copy)
            batch_x_copy = weight_array.get_batch_output(biased_batch_x_copy)
            all_output.append(batch_x_copy)

        return all_output

    def get_weight(self) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
        return [weight_array.get_weight() for weight_array in self.list_of_weight_array]

    def train(self, learning_data: np.ndarray[Any, np.dtype[np.float64]], learning_target: np.ndarray[Any, np.dtype[np.float64]], learning_rate: float = 0.01, mini_batch_size: int = 1, max_iter: int = 10000, threshold: float = 0.0) -> None:
        backpropagation = Backpropagation(
            self,
            learning_data,
            learning_target
        )

        backpropagation.learn(
            learning_rate,
            mini_batch_size,
            max_iter,
            threshold
        )
