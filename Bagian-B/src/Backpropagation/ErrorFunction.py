from __future__ import annotations
import numpy as np
from typing import NamedTuple, Callable


class ErrorFunction(NamedTuple):
    function: Callable[[np.ndarray, np.ndarray], float]

    def __call__(self, o: np.ndarray, t: np.ndarray) -> float:
        return self.function(o, t)


class SumOfSquaredErrorFunction(ErrorFunction):
    def __new__(cls) -> SumOfSquaredErrorFunction:
        # Inner Function
        def sum_of_squared_error(o: np.ndarray, t: np.ndarray) -> float:
            return np.sum(np.square(o - t)) / 2

        self = super(SumOfSquaredErrorFunction, cls).__new__(
            cls,
            sum_of_squared_error,
        )

        return self

# TODO: Cek apakah Cross Entropy Error Function sudah sesuai dengan spek


class CrossEntropyErrorFunction(ErrorFunction):
    def __new__(cls) -> CrossEntropyErrorFunction:
        # Inner Function
        def cross_entropy_error(o: np.ndarray, t: np.ndarray) -> float:
            return -np.sum(t * np.log(o))

        self = super(CrossEntropyErrorFunction, cls).__new__(
            cls,
            cross_entropy_error,
        )

        return self
