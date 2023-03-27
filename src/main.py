import numpy as np
from FileAccess import FileSystem


if __name__ == "__main__":
    neural_network = FileSystem.read_neural_network(
        "./test-cases/test-case-1.txt"
    )
    print(neural_network)

    output_x = neural_network.get_output(
        np.array([1., 2.])
    )
    print(output_x)
