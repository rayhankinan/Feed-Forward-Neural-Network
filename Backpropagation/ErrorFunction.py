import numpy as np
from typing import Callable, Any
from Serialization.Compilable import sum_of_squared_error, cross_entropy_error


class ErrorFunction:
    function: Callable[
        [
            np.ndarray[Any, np.dtype[np.float64]],
            np.ndarray[Any, np.dtype[np.float64]]
        ],
        float
    ]

    def __init__(self, function: Callable[
        [
            np.ndarray[Any, np.dtype[np.float64]],
            np.ndarray[Any, np.dtype[np.float64]]
        ],
        float
    ]) -> None:
        self.function = function

    def get_output(self, o: np.ndarray[Any, np.dtype[np.float64]], t: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        return self.function(o, t)


class SumOfSquaredErrorFunction(ErrorFunction):
    def __init__(self) -> None:
        super().__init__(sum_of_squared_error)


class CrossEntropyErrorFunction(ErrorFunction):
    def __init__(self) -> None:
        super().__init__(cross_entropy_error)
