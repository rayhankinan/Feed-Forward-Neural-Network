import numpy as np
import time
from typing import NamedTuple
from NeuralNetwork import NeuralNetwork
from . import MiniBatch, ErrorFunction


class Backpropagation(NamedTuple):
    neural_network: NeuralNetwork
    learning_data: np.ndarray
    learning_target: np.ndarray

    def learn(self, learning_rate: float, mini_batch_size: int, max_iter: int, threshold: float, error_function: ErrorFunction) -> NeuralNetwork:
        data_length = self.learning_data.shape[0]
        result: NeuralNetwork = self.neural_network
        current_error = np.inf
        index = 0

        # Until the error is less than or equal to the threshold or the maximum iteration is reached
        while current_error > threshold and index < max_iter:
            start_time = time.time()

            # Get Output
            current_output = result.get_batch_output(self.learning_data)

            # Mini-Batch Learning
            start_index = 0
            while start_index < data_length:
                end_index = min(start_index + mini_batch_size, data_length)
                partitioned_learning_data = self.learning_data[start_index:end_index]
                partitioned_learning_target = self.learning_target[start_index:end_index]

                mini_batch = MiniBatch(
                    result,
                    partitioned_learning_data,
                    partitioned_learning_target,
                )
                result = mini_batch.learn(learning_rate)
                start_index += mini_batch_size

            # Get Error
            current_error = error_function(
                current_output,
                self.learning_target
            )

            finish_time = time.time()

            print(
                f"Epoch {index + 1}\t|\tError: {round(current_error, 2)}\t|\tTime: {round(1000 * (finish_time - start_time), 2)} ms"
            )

            index += 1

        return result
