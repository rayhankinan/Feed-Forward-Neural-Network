import numpy as np
from typing import Any
from NeuralNetwork import NeuralNetwork, SoftmaxActivationFunction
from Serialization.Compilable import insert_batch_bias, subtract_batch, hadamard_product, dot_product, scalar_product


class MiniBatch:
    neural_network: NeuralNetwork
    partitioned_learning_data: np.ndarray[Any, np.dtype[np.float64]]
    partitioned_learning_target: np.ndarray[Any, np.dtype[np.float64]]

    def __init__(self, neural_network: NeuralNetwork, learning_data: np.ndarray[Any, np.dtype[np.float64]], learning_target: np.ndarray[Any, np.dtype[np.float64]]) -> None:
        self.neural_network = neural_network
        self.partitioned_learning_data = learning_data
        self.partitioned_learning_target = learning_target

    def learn(self, learning_rate: float) -> None:
        list_of_output = self.neural_network.get_all_batch_output(
            self.partitioned_learning_data
        )
        previous_delta_error = np.array([], dtype=np.float64)
        list_of_delta_weight: list[np.ndarray[Any, np.dtype[np.float64]]] = []

        for i in range(len(list_of_output) - 1, -1, -1):
            raw_x = self.partitioned_learning_data if i == 0 else list_of_output[i - 1]
            x = insert_batch_bias(raw_x)

            o = list_of_output[i]
            t = self.partitioned_learning_target
            derivated_output = self.neural_network.list_of_weight_array[i].activation_function.get_derivative_output(
                o, t
            )

            # Output Weight
            if i == len(list_of_output) - 1:
                delta_error: np.ndarray[Any, np.dtype[np.float64]]
                if type(self.neural_network.list_of_weight_array[i].activation_function) is not SoftmaxActivationFunction:
                    delta_error = hadamard_product(
                        subtract_batch(t, o),
                        derivated_output
                    )
                else:
                    delta_error = scalar_product(-1, derivated_output)

                delta_weight = scalar_product(
                    learning_rate,
                    dot_product(x.T, delta_error)
                )
                list_of_delta_weight.append(delta_weight)
                previous_delta_error = delta_error

            # Hidden Weight
            else:
                output_weight_array = self.neural_network.list_of_weight_array[i + 1]
                output_weight = output_weight_array.get_weight()[1:]
                delta_error = hadamard_product(
                    dot_product(previous_delta_error, output_weight.T),
                    derivated_output
                )
                delta_weight = scalar_product(
                    learning_rate,
                    dot_product(x.T, delta_error)
                )
                list_of_delta_weight.append(delta_weight)
                previous_delta_error = delta_error

        list_of_delta_weight.reverse()

        for i in range(len(list_of_delta_weight)):
            self.neural_network.list_of_weight_array[i].add_weight(
                list_of_delta_weight[i]
            )
