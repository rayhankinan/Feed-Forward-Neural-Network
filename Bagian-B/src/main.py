from FileAccess import FileSystem


if __name__ == "__main__":
    neural_network = FileSystem.read_neural_network(
        "./Bagian-B/test-cases/txt/linear_two_iteration.txt"
    )

    for weight in neural_network.get_weight():
        print(weight)
        print()
