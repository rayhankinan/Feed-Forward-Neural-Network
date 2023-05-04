import numpy as np
from typing import NamedTuple
from NeuralNetwork import NeuralNetwork, Layer


class MiniBatch(NamedTuple):
    neural_network: NeuralNetwork
    partitioned_learning_data: np.ndarray
    partitioned_learning_target: np.ndarray

    def learn(self, learning_rate: float) -> NeuralNetwork:
        list_of_output = self.neural_network.get_all_batch_output(
            self.partitioned_learning_data
        )
        result: NeuralNetwork = self.neural_network
        previous_delta_error: np.ndarray = None
        new_layer: list[Layer] = [
            None for _ in range(len(list_of_output))
        ]

        for i in range(len(list_of_output) - 1, -1, -1):
            raw_x = self.partitioned_learning_data if i == 0 else list_of_output[i - 1]
            x = np.array(np.c_[np.ones(raw_x.shape[0]), raw_x])

            o = list_of_output[i]
            t = self.partitioned_learning_target
            derivated_output = result.list_of_layer[i].activation_function.get_derivative_output(
                o
            )

            if i == len(list_of_output) - 1:
                delta_error = np.multiply(np.subtract(t, o), derivated_output)

                delta_weight = np.array(
                    np.dot(learning_rate, np.dot(x.T, delta_error))
                )
                new_layer[i] = result.list_of_layer[i].get_updated_weight(
                    delta_weight
                )
                previous_delta_error = delta_error

            else:
                output_weight = result.list_of_layer[i + 1].get_weight()[1:]
                delta_error = np.multiply(
                    np.dot(
                        previous_delta_error, output_weight.T
                    ),
                    derivated_output
                )
                delta_weight = np.array(
                    np.dot(learning_rate, np.dot(x.T, delta_error))
                )
                new_layer[i] = result.list_of_layer[i].get_updated_weight(
                    delta_weight
                )
                previous_delta_error = delta_error

        for i in range(len(new_layer)):
            result.list_of_layer[i] = new_layer[i]

        return result
