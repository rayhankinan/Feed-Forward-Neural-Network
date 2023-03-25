import numpy as np
from NeuralNetwork import Perceptron, Layer, SigmoidActivationFunction, NeuralNetwork


if __name__ == "__main__":
    neural_network = NeuralNetwork(
        list_of_layer=[
            Layer(
                list_of_perceptron=[
                    Perceptron(weight=np.array([0.1, 0.2, 0.3])),
                    Perceptron(weight=np.array([0.4, 0.5, 0.6])),
                ],
                activation_function=SigmoidActivationFunction(),
            )
        ]
    )

    output_x = neural_network.get_output(np.array([1, 2]))
    print(output_x)
