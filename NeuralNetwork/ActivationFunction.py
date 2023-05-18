import numpy as np
from typing import Callable, Any
from Serialization.Compilable import linear, derivative_output_linear, relu, derivative_output_relu, sigmoid, derivative_output_sigmoid, softmax, derivative_output_softmax


class ActivationFunction:
    function: Callable[
        [
            np.ndarray[Any, np.dtype[np.float64]]
        ],
        np.ndarray[Any, np.dtype[np.float64]]
    ]
    derivative_output: Callable[
        [
            np.ndarray[Any, np.dtype[np.float64]],
            np.ndarray[Any, np.dtype[np.float64]]
        ],
        np.ndarray[Any, np.dtype[np.float64]]
    ]

    def __init__(self, function: Callable[[np.ndarray[Any, np.dtype[np.float64]]], np.ndarray[Any, np.dtype[np.float64]]], derivative_output: Callable[[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]], np.ndarray[Any, np.dtype[np.float64]]]) -> None:
        self.function = function
        self.derivative_output = derivative_output

    def get_output(self, o: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self.function(o)

    def get_derivative_output(self, o: np.ndarray[Any, np.dtype[np.float64]], t: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self.derivative_output(o, t)


class LinearActivationFunction(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(linear, derivative_output_linear)


class ReLUActivationFunction(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(relu, derivative_output_relu)


class SigmoidActivationFunction(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(sigmoid, derivative_output_sigmoid)


class SoftmaxActivationFunction(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(softmax, derivative_output_softmax)
