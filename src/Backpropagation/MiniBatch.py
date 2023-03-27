import numpy as np
from typing import NamedTuple
from NeuralNetwork import NeuralNetwork


class MiniBatch(NamedTuple):
    neural_network: NeuralNetwork
    partitioned_learning_data: np.ndarray

    def learn(self, learning_rate: float) -> NeuralNetwork:
        # TODO: Implement this method
        return self.neural_network
