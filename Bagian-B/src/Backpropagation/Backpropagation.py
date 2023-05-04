import numpy as np
from typing import NamedTuple
from NeuralNetwork import NeuralNetwork
from . import MiniBatch, ErrorFunction


class Backpropagation(NamedTuple):
    neural_network: NeuralNetwork
    learning_data: np.ndarray
    learning_target: np.ndarray

    def learn(self, learning_rate: float, mini_batch_size: int, max_iter: int, threshold: float, error_function: ErrorFunction) -> NeuralNetwork:
        current_error = np.inf
        data_length = self.learning_data.shape[0]
        result: NeuralNetwork = self.neural_network

        for _ in range(max_iter):
            if current_error <= threshold:
                break

            start_index = 0
            current_error = 0
            while start_index < data_length:
                end_index = min(start_index + mini_batch_size, data_length)
                partitioned_learning_data = self.learning_data[start_index:end_index]
                partitioned_learning_target = self.learning_target[start_index:end_index]

                mini_batch = MiniBatch(
                    result,
                    partitioned_learning_data,
                    partitioned_learning_target,
                )
                result, new_error = mini_batch.learn(
                    learning_rate,
                    error_function,
                )
                start_index += mini_batch_size
                current_error += new_error

        return result
