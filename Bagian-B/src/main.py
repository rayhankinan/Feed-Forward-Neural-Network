from FileAccess import FileSystem


if __name__ == "__main__":
    test_case = ["linear_small_lr", "linear_two_iteration", "linear", "mlp", "relu", "sigmoid_mini_batch_GD", "sigmoid_stochastic_GD", "sigmoid", "softmax_error_only", "softmax", "sse_only"]
    for i in range(len(test_case)):
        print(test_case[i])
        neural_network = FileSystem.read_neural_network(
            "./Bagian-B/test-cases/txt/" + test_case[i] + ".txt"
        )

        for weight in neural_network.get_weight():
            print(weight)
            print()