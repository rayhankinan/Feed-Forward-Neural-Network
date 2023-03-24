import numpy as np


class ActivationFunction:
    def __init__(self, function: np.vectorize, derivative: np.vectorize) -> None:
        self.function = function
        self.derivative = derivative

    def __call__(self, x) -> np.ndarray:
        return np.array(self.function(x))

    def get_derivative(self, x) -> np.ndarray:
        return np.array(self.derivative(x))


class LinearActivationFunction(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(np.vectorize(lambda x: x), np.vectorize(lambda _: 1))


class ReLUActivationFunction(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(
            np.vectorize(lambda x: max(0, x)),
            np.vectorize(lambda x: 1 if x > 0 else 0),
        )


class SigmoidActivationFunction(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(
            np.vectorize(lambda x: 1 / (1 + np.exp(-x))),
            np.vectorize(lambda x: x * (1 - x)),
        )

# TODO: Menambahkan fungsi aktivasi SoftMax
