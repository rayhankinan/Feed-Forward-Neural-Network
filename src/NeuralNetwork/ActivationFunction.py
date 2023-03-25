from __future__ import annotations
import numpy as np
from typing import NamedTuple


class ActivationFunction(NamedTuple):
    function: np.vectorize
    derivative: np.vectorize

    def __call__(self, x) -> np.ndarray:
        return np.array(self.function(x))

    def get_derivative(self, x) -> np.ndarray:
        return np.array(self.derivative(x))


class LinearActivationFunction(ActivationFunction):
    def __new__(cls) -> LinearActivationFunction:
        self = super(LinearActivationFunction, cls).__new__(
            cls,
            np.vectorize(lambda x: x),
            np.vectorize(lambda _: 1),
        )

        return self


class ReLUActivationFunction(ActivationFunction):
    def __new__(cls) -> ReLUActivationFunction:
        self = super(ReLUActivationFunction, cls).__new__(
            cls,
            np.vectorize(lambda x: max(0, x)),
            np.vectorize(lambda x: 1 if x > 0 else 0),
        )

        return self


class SigmoidActivationFunction(ActivationFunction):
    def __new__(cls) -> SigmoidActivationFunction:
        self = super(SigmoidActivationFunction, cls).__new__(
            cls,
            np.vectorize(lambda x: 1 / (1 + np.exp(-x))),
            np.vectorize(lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2),
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
            np.vectorize(softmax),
            np.vectorize(derivative_softmax),
        )

        return self
