import numba
import math


@numba.vectorize([numba.float64(numba.float64)], cache=True)
def linear_vector(x: float) -> float:
    return x


@numba.vectorize([numba.float64(numba.float64)], cache=True)
def derived_linear_vector(_: float) -> float:
    return 1.0


@numba.vectorize([numba.float64(numba.float64)], cache=True)
def relu_vector(x: float) -> float:
    return max(0, x)


@numba.vectorize([numba.float64(numba.float64)], cache=True)
def derived_relu_vector(x: float) -> float:
    return 1 if x > 0 else 0


@numba.vectorize([numba.float64(numba.float64)], cache=True)
def sigmoid_vector(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@numba.vectorize([numba.float64(numba.float64)], cache=True)
def derived_sigmoid_vector(x: float) -> float:
    return x * (1 - x)
