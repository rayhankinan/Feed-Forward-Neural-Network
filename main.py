import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork, Layer, SigmoidActivationFunction, SoftmaxActivationFunction, CrossEntropyErrorFunction

if __name__ == "__main__":
    # Load Iris Dataset
    iris = load_iris()
    learning_data = np.array(iris["data"], dtype=np.float64)  # type: ignore
    learning_target = np.array(iris["target"], dtype=np.int64)  # type: ignore

    one_hot_learning_target = np.zeros(
        (learning_target.size, learning_target.max() + 1)
    )
    one_hot_learning_target[
        np.arange(learning_target.size),
        learning_target
    ] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        learning_data,
        one_hot_learning_target,
        test_size=0.2,
        random_state=42
    )
    numpy_X_train = np.array(X_train, dtype=np.float64)
    numpy_X_test = np.array(X_test, dtype=np.float64)
    numpy_y_train = np.array(y_train, dtype=np.float64)
    numpy_y_test = np.array(y_test, dtype=np.float64)

    # Random Seed
    np.random.seed(42)

    # Create Neural Network
    neural_network = NeuralNetwork(
        4,
        [
            Layer(8, SigmoidActivationFunction()),
            Layer(8, SigmoidActivationFunction()),
            Layer(3, SoftmaxActivationFunction()),
        ],
        CrossEntropyErrorFunction()
    )

    # Train Neural Network
    neural_network.train(
        learning_data=numpy_X_train,
        learning_target=numpy_y_train,
        learning_rate=0.01,
        mini_batch_size=1,
        max_iter=10000,
        threshold=0.01 * len(numpy_X_train),
    )

    # Test Neural Network
    y_pred = neural_network.get_batch_output(numpy_X_test)

    cross_entropy = CrossEntropyErrorFunction()

    print(f"Prediction:\n{y_pred}")
    print()

    print(f"Actual:\n{numpy_y_test}")
    print()

    error = cross_entropy.get_output(y_pred, numpy_y_test)
    print(f"Error: {error} / {round(100 * error / len(numpy_X_test), 2)}%")
    print()
