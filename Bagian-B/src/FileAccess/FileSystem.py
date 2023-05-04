import numpy as np
from NeuralNetwork import Layer, Row, ActivationFunction, LinearActivationFunction, ReLUActivationFunction, SigmoidActivationFunction, SoftmaxActivationFunction, NeuralNetwork
from Backpropagation import Backpropagation, ErrorFunction, SumOfSquaredErrorFunction, CrossEntropyErrorFunction


class FileSystem:
    @staticmethod
    def load_from_file(path: str) -> NeuralNetwork:
        with open(path, "r") as file:
            input_size = int(file.readline().rstrip())
            num_of_layers = int(file.readline().rstrip())

            prev_num_of_perceptrons = input_size
            list_of_layer: list[Layer] = []
            latest_activation_function_type: str = None

            for _ in range(num_of_layers):
                num_of_perceptrons = int(file.readline().rstrip())
                list_of_weight_row: list[Row] = []

                for _ in range(prev_num_of_perceptrons + 1):
                    weight = np.array(
                        list(map(float, file.readline().rstrip().split()))
                    )
                    row = Row(weight)
                    list_of_weight_row.append(row)

                activation_function_type = file.readline().rstrip()

                if latest_activation_function_type == "softmax":
                    raise NotImplementedError()

                activation_function: ActivationFunction
                if activation_function_type == "linear":
                    activation_function = LinearActivationFunction()
                elif activation_function_type == "relu":
                    activation_function = ReLUActivationFunction()
                elif activation_function_type == "sigmoid":
                    activation_function = SigmoidActivationFunction()
                elif activation_function_type == "softmax":
                    activation_function = SoftmaxActivationFunction()
                else:
                    raise NotImplementedError()

                layer = Layer(list_of_weight_row, activation_function)
                list_of_layer.append(layer)

                prev_num_of_perceptrons = num_of_perceptrons
                latest_activation_function_type = activation_function_type

            neural_network = NeuralNetwork(list_of_layer)

            return neural_network

    @staticmethod
    def save_to_file(neural_network: NeuralNetwork, path: str) -> None:
        with open(path, "w") as file:
            input_weight = neural_network.list_of_layer[0].get_weight()
            input_size = input_weight.shape[0] - 1
            file.write(f"{input_size}\n")

            num_of_layers = len(neural_network.list_of_layer)
            file.write(f"{num_of_layers}\n")

            for layer in neural_network.list_of_layer:
                weight = layer.get_weight()
                num_of_perceptrons = weight.shape[1]
                file.write(f"{num_of_perceptrons}\n")

                for i in range(weight.shape[0]):
                    for j in range(weight.shape[1]):
                        file.write(f"{weight[i][j]} ")
                    file.write("\n")

                activation_function_type: str = None
                if type(layer.activation_function) is LinearActivationFunction:
                    activation_function_type = "linear"
                elif type(layer.activation_function) is ReLUActivationFunction:
                    activation_function_type = "relu"
                elif type(layer.activation_function) is SigmoidActivationFunction:
                    activation_function_type = "sigmoid"
                elif type(layer.activation_function) is SoftmaxActivationFunction:
                    activation_function_type = "softmax"
                else:
                    raise NotImplementedError()

                file.write(f"{activation_function_type}\n")

    @staticmethod
    def learn_from_file(path: str) -> NeuralNetwork:
        with open(path, "r") as file:
            input_size = int(file.readline().rstrip())
            num_of_layers = int(file.readline().rstrip())

            prev_num_of_perceptrons = input_size
            list_of_layer: list[Layer] = []
            latest_activation_function_type: str = None

            for _ in range(num_of_layers):
                num_of_perceptrons = int(file.readline().rstrip())
                list_of_weight_row: list[Row] = []

                for _ in range(prev_num_of_perceptrons + 1):
                    weight = np.array(
                        list(map(float, file.readline().rstrip().split()))
                    )
                    row = Row(weight)
                    list_of_weight_row.append(row)

                activation_function_type = file.readline().rstrip()

                if latest_activation_function_type == "softmax":
                    raise NotImplementedError()

                activation_function: ActivationFunction
                if activation_function_type == "linear":
                    activation_function = LinearActivationFunction()
                elif activation_function_type == "relu":
                    activation_function = ReLUActivationFunction()
                elif activation_function_type == "sigmoid":
                    activation_function = SigmoidActivationFunction()
                elif activation_function_type == "softmax":
                    activation_function = SoftmaxActivationFunction()
                else:
                    raise NotImplementedError()

                layer = Layer(list_of_weight_row, activation_function)
                list_of_layer.append(layer)

                prev_num_of_perceptrons = num_of_perceptrons
                latest_activation_function_type = activation_function_type

            initial_neural_network = NeuralNetwork(list_of_layer)

            test_case_size = int(file.readline().rstrip())
            input_array: list[np.ndarray] = []
            target_array: list[np.ndarray] = []

            for _ in range(test_case_size):
                input_vector = list(
                    map(float, file.readline().rstrip().split())
                )

                input_array.append(np.array(input_vector))

            for _ in range(test_case_size):
                target_vector = list(
                    map(float, file.readline().rstrip().split())
                )
                target_array.append(np.array(target_vector))

            learning_rate = float(file.readline().rstrip())
            mini_batch_size = int(file.readline().rstrip())
            max_iter = int(file.readline().rstrip())
            threshold = float(file.readline().rstrip())

            backpropagation = Backpropagation(
                initial_neural_network,
                np.array(input_array),
                np.array(target_array)
            )

            error_function: ErrorFunction
            if latest_activation_function_type == "linear" or latest_activation_function_type == "relu" or latest_activation_function_type == "sigmoid":
                error_function = SumOfSquaredErrorFunction()
            elif latest_activation_function_type == "softmax":
                error_function = CrossEntropyErrorFunction()
            else:
                raise NotImplementedError()

            return backpropagation.learn(learning_rate, mini_batch_size, max_iter, threshold, error_function)
