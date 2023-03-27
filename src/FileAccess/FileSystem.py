import numpy as np
from NeuralNetwork import Layer, Perceptron, ActivationFunction, LinearActivationFunction, ReLUActivationFunction, SigmoidActivationFunction, SoftmaxActivationFunction, NeuralNetwork


class FileSystem:
    @staticmethod
    def read_neural_network(path) -> NeuralNetwork:
        with open(path, 'r') as file:
            num_of_layers = int(file.readline().rstrip('\n'))
            list_of_layer: list[Layer] = []

            for _ in range(num_of_layers):
                num_of_perceptrons = int(file.readline().rstrip('\n'))
                list_of_perceptron: list[Perceptron] = []

                for _ in range(num_of_perceptrons):
                    weight = np.array(
                        list(map(float, file.readline().rstrip('\n').split()))
                    )
                    perceptron = Perceptron(weight)
                    list_of_perceptron.append(perceptron)

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

                layer = Layer(list_of_perceptron, activation_function)
                list_of_layer.append(layer)

            return NeuralNetwork(list_of_layer)
