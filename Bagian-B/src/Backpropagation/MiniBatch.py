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
        previous_delta_error: np.ndarray = np.array([])

        for i in range(len(list_of_output) - 1, -1):
            o = list_of_output[i]
            t = self.partitioned_learning_target[i]
            derivated_output = result.list_of_layer[i].activation_function.get_derivative_output(
                o
            )

            if i == 0:
                x = self.partitioned_learning_data
                output_weight = result.list_of_layer[i + 1].get_weight()
                delta_error = np.multiply(
                    np.dot(
                        previous_delta_error, output_weight.T
                    ),
                    derivated_output
                )
                delta_weight = np.array(
                    np.dot(learning_rate, np.dot(x.T, delta_error))
                )
                result.list_of_layer[i] = result.list_of_layer[i].get_updated_weight(
                    delta_weight
                )
                previous_delta_error = delta_error

            elif i == len(list_of_output) - 1:
                x = list_of_output[i - 1]
                delta_error = np.multiply(np.subtract(t, o), derivated_output)
                delta_weight = np.array(
                    np.dot(learning_rate, np.dot(x.T, delta_error))
                )
                result.list_of_layer[i] = result.list_of_layer[i].get_updated_weight(
                    delta_weight
                )
                previous_delta_error = delta_error

            else:
                x = list_of_output[i - 1]
                output_weight = result.list_of_layer[i + 1].get_weight()
                delta_error = np.multiply(
                    np.dot(
                        previous_delta_error, output_weight.T
                    ),
                    derivated_output
                )
                delta_weight = np.array(
                    np.dot(learning_rate, np.dot(x.T, delta_error))
                )
                result.list_of_layer[i] = result.list_of_layer[i].get_updated_weight(
                    delta_weight
                )
                previous_delta_error = delta_error

        # TODO: Update Weight

        return result
