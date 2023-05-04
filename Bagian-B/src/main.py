from FileAccess import FileSystem


if __name__ == "__main__":
    test_case = ["linear_small_lr", "linear_two_iteration", "linear", "mlp", "relu", "sigmoid_mini_batch_GD",
                 "sigmoid_stochastic_GD", "sigmoid", "softmax_error_only", "softmax", "sse_only"]

    for i in range(len(test_case)):
        print(test_case[i])
        neural_network = FileSystem.learn_from_file(
            f"./Bagian-B/test-cases/txt/{test_case[i]}.txt"
        )

        print("Weight:")
        for weight in neural_network.get_weight():
            print(weight)
        print()

        FileSystem.save_to_file(
            neural_network, f"./Bagian-B/model/{test_case[i]}.txt"
        )

    # new_neural_network = FileSystem.load_from_file(
    #     "./Bagian-B/model/linear.txt")

    # print("Weight:")
    # for weight in new_neural_network.get_weight():
    #     print(weight)
    # print()
