import numpy as np
from typing import NamedTuple
from NeuralNetwork import NeuralNetwork


class MiniBatch(NamedTuple):
    neural_network: NeuralNetwork
    partitioned_learning_data: np.ndarray
    partitioned_learning_target: np.ndarray

    def learn(self, learning_rate: float) -> NeuralNetwork:
        list_of_output = self.neural_network.get_all_batch_output(
            self.partitioned_learning_data
        )
        result: NeuralNetwork = self.neural_network

        return result
