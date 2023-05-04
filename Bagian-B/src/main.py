import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from FileAccess import FileSystem
from Backpropagation import Backpropagation
from Backpropagation import CrossEntropyErrorFunction


if __name__ == "__main__":
    iris = load_iris()

    learning_data = np.array(iris.data)
    learning_target = np.array(iris.target)

    one_hot_learning_target = np.zeros(
        (learning_target.size, learning_target.max() + 1)
    )
    one_hot_learning_target[np.arange(
        learning_target.size), learning_target] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        learning_data,
        one_hot_learning_target,
        test_size=0.2,
        random_state=42
    )
    numpy_X_train = np.array(X_train)
    numpy_X_test = np.array(X_test)
    numpy_y_train = np.array(y_train)
    numpy_y_test = np.array(y_test)

    neural_network = FileSystem.load_from_file(
        "./Bagian-B/model/iris-raw.txt"
    )
    backpropagation = Backpropagation(
        neural_network,
        numpy_X_train,
        numpy_y_train
    )

    cross_entropy = CrossEntropyErrorFunction()

    new_neural_network = backpropagation.learn(
        learning_rate=0.1,
        mini_batch_size=1,
        max_iter=1000,
        threshold=0.1 * len(numpy_X_train),
        error_function=cross_entropy
    )
    print()

    print("Weight:")
    for weight in neural_network.get_weight():
        print(weight)
    print()

    y_pred = new_neural_network.get_batch_output(numpy_X_test)

    print(f"Prediction:\n{y_pred}")
    print()

    print(f"Actual:\n{numpy_y_test}")
    print()

    error = cross_entropy(y_pred, numpy_y_test)
    print(f"Error: {error} / {round(100 * error / len(numpy_X_test), 2)}%")

    FileSystem.save_to_file(
        new_neural_network,
        "./Bagian-B/model/iris-learn.txt"
    )
