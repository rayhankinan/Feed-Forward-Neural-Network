import numpy as np
from sklearn.datasets import load_iris
from FileAccess import FileSystem


if __name__ == "__main__":
    iris = load_iris()

    learning = np.array(iris.data)
    target = np.array(iris.target)

    one_hot_target = np.zeros((target.size, target.max() + 1))
    one_hot_target[np.arange(target.size), target] = 1

    neural_network = FileSystem.load_from_file(
        "./Bagian-B/model/iris-raw.txt"
    )

    print("Weight:")
    for weight in neural_network.get_weight():
        print(weight)
