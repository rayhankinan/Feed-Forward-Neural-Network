import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from FileAccess import FileSystem
from Backpropagation import Backpropagation
from Backpropagation import CrossEntropyErrorFunction


if __name__ == "__main__":
    neural_network = FileSystem.learn_from_file(
        "./Bagian-B/test-cases/txt/mlp.txt"
    )
    for weight in neural_network.get_weight():
        print(weight)
        print()
