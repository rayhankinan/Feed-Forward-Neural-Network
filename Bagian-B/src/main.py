import numpy as np
from FileAccess import FileSystem


if __name__ == "__main__":
    neural_network = FileSystem.read_neural_network(
        "./Bagian-B/test-cases/test-case-1.txt"
    )

    output_x = neural_network.get_output(
        np.array([1., 2.])
    )
    print(output_x)
