import numpy as np
from NeuralNetwork import Layer, Row, ActivationFunction, LinearActivationFunction, ReLUActivationFunction, SigmoidActivationFunction, SoftmaxActivationFunction, NeuralNetwork
from Backpropagation import Backpropagation


class FileSystem:
    @staticmethod
    def read_neural_network(path) -> NeuralNetwork:
        with open(path, 'r') as file:
            input_size = int(file.readline().rstrip('\n'))
            num_of_layers = int(file.readline().rstrip('\n'))

            prev_num_of_perceptrons = input_size
            list_of_layer: list[Layer] = []

            for _ in range(num_of_layers):
                num_of_perceptrons = int(file.readline().rstrip('\n'))
                list_of_weight_row: list[Row] = []

                for _ in range(prev_num_of_perceptrons + 1):
                    weight = np.array(
                        list(map(float, file.readline().rstrip('\n').split()))
                    )
                    row = Row(weight)
                    list_of_weight_row.append(row)

                activation_function_type = file.readline().rstrip('\n')
                activation_function: ActivationFunction

                match activation_function_type:
                    case "linear":
                        activation_function = LinearActivationFunction()
                    case "relu":
                        activation_function = ReLUActivationFunction()
                    case "sigmoid":
                        activation_function = SigmoidActivationFunction()
                    case "softmax":
                        activation_function = SoftmaxActivationFunction()
                    case _:
                        raise NotImplementedError()

                layer = Layer(list_of_weight_row, activation_function)
                list_of_layer.append(layer)

                prev_num_of_perceptrons = num_of_perceptrons

            initial_neural_network = NeuralNetwork(list_of_layer)

            test_case_size = int(file.readline().rstrip('\n'))
            input_array = np.array([])
            target_array = np.array([])

            for _ in range(test_case_size):
                input_vector = list(
                    map(float, file.readline().rstrip('\n').split())
                )

                input_array = np.append(input_array, input_vector)

            for _ in range(test_case_size):
                target_vector = list(
                    map(float, file.readline().rstrip('\n').split())
                )
                target_array = np.append(target_array, target_vector)

            learning_rate = float(file.readline().rstrip('\n'))
            mini_batch_size = int(file.readline().rstrip('\n'))
            max_iter = int(file.readline().rstrip('\n'))
            threshold = float(file.readline().rstrip('\n'))

            backpropagation = Backpropagation(
                initial_neural_network, input_array, target_array
            )

            return backpropagation.learn(learning_rate, mini_batch_size, max_iter, threshold)
