from __future__ import annotations
import numpy as np
from typing import NamedTuple, Callable


class ActivationFunction(NamedTuple):
    function: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray], np.ndarray]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.function(x)

    def get_derivative(self, x: np.ndarray) -> np.ndarray:
        return self.derivative(x)


class LinearActivationFunction(ActivationFunction):
    def __new__(cls) -> LinearActivationFunction:
        # Inner Function
        def linear(x: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda x: x)(x))

        # Inner Function
        def derivative_linear(x: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda _: 1)(x))

        self = super(LinearActivationFunction, cls).__new__(
            cls,
            linear,
            derivative_linear,
        )

        return self


class ReLUActivationFunction(ActivationFunction):
    def __new__(cls) -> ReLUActivationFunction:
        # Inner Function
        def relu(x: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda x: max(0, x))(x))

        # Inner Function
        def derivative_relu(x: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda x: 1 if x > 0 else 0)(x))

        self = super(ReLUActivationFunction, cls).__new__(
            cls,
            relu,
            derivative_relu,
        )

        return self


class SigmoidActivationFunction(ActivationFunction):
    def __new__(cls) -> SigmoidActivationFunction:
        # Inner Function
        def sigmoid(x: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda x: 1 / (1 + np.exp(-x)))(x))

        # Inner Function
        def derivative_sigmoid(x: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2)(x))

        self = super(SigmoidActivationFunction, cls).__new__(
            cls,
            sigmoid,
            derivative_sigmoid,
        )

        return self


class SoftmaxActivationFunction(ActivationFunction):
    def __new__(cls) -> SoftmaxActivationFunction:
        # Inner Function
        def softmax(x: np.ndarray) -> np.ndarray:
            shift_x = np.array(x - np.max(x))
            exps = np.exp(shift_x)
            return np.array(exps / np.sum(exps, axis=0))

        # Inner Function
        def derivative_softmax(x: np.ndarray) -> np.ndarray:
            s = softmax(x).reshape(-1, 1)
            return np.array(np.diagflat(s) - np.dot(s, s.T))

        self = super(SoftmaxActivationFunction, cls).__new__(
            cls,
            softmax,
            derivative_softmax,
        )

        return self
