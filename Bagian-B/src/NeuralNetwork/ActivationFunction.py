from __future__ import annotations
import numpy as np
from typing import NamedTuple, Callable


class ActivationFunction(NamedTuple):
    function: Callable[[np.ndarray], np.ndarray]
    derivative_output: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __call__(self, o: np.ndarray) -> np.ndarray:
        return self.function(o)

    def get_derivative_output(self, o: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.derivative_output(o, t)


class LinearActivationFunction(ActivationFunction):
    def __new__(cls) -> LinearActivationFunction:
        # Inner Function
        def linear(x: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda x: x)(x))

        # Inner Function
        def derivative_output_linear(o: np.ndarray, _: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda _: 1)(o))

        self = super(LinearActivationFunction, cls).__new__(
            cls,
            linear,
            derivative_output_linear,
        )

        return self


class ReLUActivationFunction(ActivationFunction):
    def __new__(cls) -> ReLUActivationFunction:
        # Inner Function
        def relu(x: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda x: max(0, x))(x))

        # Inner Function
        def derivative_output_relu(o: np.ndarray, _: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda o: 1 if o > 0 else 0)(o))

        self = super(ReLUActivationFunction, cls).__new__(
            cls,
            relu,
            derivative_output_relu,
        )

        return self


class SigmoidActivationFunction(ActivationFunction):
    def __new__(cls) -> SigmoidActivationFunction:
        # Inner Function
        def sigmoid(x: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda x: 1 / (1 + np.exp(-x)))(x))

        # Inner Function
        def derivative_output_sigmoid(o: np.ndarray, _: np.ndarray) -> np.ndarray:
            return np.array(np.vectorize(lambda o: o * (1 - o))(o))

        self = super(SigmoidActivationFunction, cls).__new__(
            cls,
            sigmoid,
            derivative_output_sigmoid,
        )

        return self


class SoftmaxActivationFunction(ActivationFunction):
    def __new__(cls) -> SoftmaxActivationFunction:
        # Inner Function
        def softmax(x: np.ndarray) -> np.ndarray:
            shift_x = np.array(x - np.max(x))
            exps = np.exp(shift_x)
            return np.array(exps / np.sum(np.sum(exps, axis=1), axis=0))

        def derivative_output_softmax(o: np.ndarray, t: np.ndarray) -> np.ndarray:
            return np.array(np.subtract(o, t))

        self = super(SoftmaxActivationFunction, cls).__new__(
            cls,
            softmax,
            derivative_output_softmax,
        )

        return self
