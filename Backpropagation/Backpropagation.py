import numpy as np
import time
from typing import Any
from NeuralNetwork import NeuralNetwork
from . import MiniBatch


class Backpropagation:
    neural_network: NeuralNetwork
    learning_data: np.ndarray[Any, np.dtype[np.float64]]
    learning_target: np.ndarray[Any, np.dtype[np.float64]]

    def __init__(self, neural_network: NeuralNetwork, learning_data: np.ndarray[Any, np.dtype[np.float64]], learning_target: np.ndarray[Any, np.dtype[np.float64]]) -> None:
        self.neural_network = neural_network
        self.learning_data = learning_data
        self.learning_target = learning_target

    def learn(self, learning_rate: float, mini_batch_size: int, max_iter: int, threshold: float) -> None:
        # Shuffle Learning Data
        combined = np.column_stack((self.learning_data, self.learning_target))
        np.random.shuffle(combined)

        # Split Learning Data and Learning Target
        _, col = self.learning_target.shape
        shuffled_learning_data = combined[:, :-col]
        shuffled_learning_target = combined[:, -col:]

        data_length = shuffled_learning_data.shape[0]
        current_error = np.inf
        index = 0

        # Until the error is less than or equal to the threshold or the maximum iteration is reached
        while current_error > threshold and index < max_iter:
            start_time = time.perf_counter()

            # Get Output
            current_output = self.neural_network.get_batch_output(
                shuffled_learning_data
            )

            # Mini-Batch Learning
            start_index = 0
            while start_index < data_length:
                end_index = min(start_index + mini_batch_size, data_length)
                partitioned_learning_data = shuffled_learning_data[start_index:end_index]
                partitioned_learning_target = shuffled_learning_target[start_index:end_index]

                mini_batch = MiniBatch(
                    self.neural_network,
                    partitioned_learning_data,
                    partitioned_learning_target,
                )
                mini_batch.learn(learning_rate)
                start_index += mini_batch_size

            # Get Error
            current_error = self.neural_network.error_function.get_output(
                current_output,
                shuffled_learning_target
            )

            finish_time = time.perf_counter()

            print(
                f"Epoch {index + 1}\t|\tError: {round(current_error, 4)}\t|\tTime: {round(1000 * (finish_time - start_time), 4)} ms"
            )

            index += 1
