import numpy as np
import numba
from typing import Any
from .Vectorizable import linear_vector, derived_linear_vector, relu_vector, derived_relu_vector, sigmoid_vector, derived_sigmoid_vector


@numba.njit(numba.float64[:, :](numba.float64[:, :]), cache=True)
def linear(x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return linear_vector(x)  # type: ignore


@numba.njit(numba.float64[:, :](numba.float64[:, :], numba.float64[:, :]), cache=True)
def derivative_output_linear(o: np.ndarray[Any, np.dtype[np.float64]], _: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return derived_linear_vector(o)  # type: ignore


@numba.njit(numba.float64[:, :](numba.float64[:, :]), cache=True)
def relu(x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return relu_vector(x)  # type: ignore


@numba.njit(numba.float64[:, :](numba.float64[:, :], numba.float64[:, :]), cache=True)
def derivative_output_relu(o: np.ndarray[Any, np.dtype[np.float64]], _: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return derived_relu_vector(o)  # type: ignore


@numba.njit(numba.float64[:, :](numba.float64[:, :]), cache=True)
def sigmoid(x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return sigmoid_vector(x)  # type: ignore


@numba.njit(numba.float64[:, :](numba.float64[:, :], numba.float64[:, :]), cache=True)
def derivative_output_sigmoid(o: np.ndarray[Any, np.dtype[np.float64]], _: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return derived_sigmoid_vector(o)  # type: ignore


@numba.njit(numba.float64(numba.float64[:, :], numba.float64[:, :]), cache=True)
def sum_of_squared_error(o: np.ndarray[Any, np.dtype[np.float64]], t: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return np.sum(np.sum(np.square(o - t), axis=1) / 2, axis=0)


@numba.njit(numba.float64(numba.float64[:, :], numba.float64[:, :]), cache=True)
def cross_entropy_error(o: np.ndarray[Any, np.dtype[np.float64]], t: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return np.sum(-np.sum(t * np.log(o), axis=1), axis=0)


@numba.njit(numba.float64[:](numba.float64[:]), cache=True)
def insert_single_bias(x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    bias = np.ones(1)
    return np.append(bias, x)


@numba.njit(numba.float64[:, :](numba.float64[:, :]), cache=True)
def insert_batch_bias(x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    bias = np.ones((x.shape[0], 1))
    return np.append(bias, x, axis=1)


@numba.njit(numba.float64[:, :](numba.float64[:, :]), cache=True)
def softmax(x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    shift_x = np.subtract(x, np.max(x))  # Normalized
    exps = np.exp(shift_x)
    sums = np.sum(exps, axis=1)[:, np.newaxis]
    return np.divide(exps, sums)


@numba.njit(numba.float64[:, :](numba.float64[:, :], numba.float64[:, :]), cache=True)
def derivative_output_softmax(o: np.ndarray[Any, np.dtype[np.float64]], t: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.subtract(o, t)


@numba.njit(numba.float64[:](numba.float64[:, :], numba.float64[:]), cache=True)
def get_single(w: np.ndarray[Any, np.dtype[np.float64]], x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    A = np.ascontiguousarray(w.T)
    B = np.ascontiguousarray(x)
    return np.dot(A, B)


@numba.njit(numba.float64[:, :](numba.float64[:, :], numba.float64[:, :]), cache=True)
def get_batch(w: np.ndarray[Any, np.dtype[np.float64]], x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    A = np.ascontiguousarray(x)
    B = np.ascontiguousarray(w)
    return np.dot(A, B)


@numba.njit(numba.float64[:, :](numba.float64[:, :], numba.float64[:, :]), cache=True)
def add_batch(w: np.ndarray[Any, np.dtype[np.float64]], d: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.add(w, d)


@numba.njit(numba.float64[:, :](numba.float64[:, :], numba.float64[:, :]), cache=True)
def subtract_batch(w: np.ndarray[Any, np.dtype[np.float64]], d: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.subtract(w, d)


@numba.njit(numba.float64[:, :](numba.float64[:, :], numba.float64[:, :]), cache=True)
def dot_product(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    A = np.ascontiguousarray(x)
    B = np.ascontiguousarray(y)
    return np.dot(A, B)


@numba.njit(numba.float64[:, :](numba.float64[:, :], numba.float64[:, :]), cache=True)
def hadamard_product(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.multiply(x, y)


@numba.njit(numba.float64[:, :](numba.float64, numba.float64[:, :]), cache=True)
def scalar_product(k: float, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return k * x
