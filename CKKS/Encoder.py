import numpy as np
from numpy.polynomial import Polynomial
from typing import Any


class Encoder:
    xi: np.complex128
    M: int

    def __init__(self, M: int) -> None:
        # Atribut xi merupakan M-th root of unity yang akan digunakan sebagai basis perhitungan
        self.xi = np.exp(2 * np.pi * 1j / M)
        self.M = M

    def vandermonde(self) -> np.ndarray[Any, np.dtype[np.complex128]]:
        # Menghasilkan matriks Vandermonde
        matrix = np.array([], dtype=np.complex128).reshape(0, self.M // 2)

        for i in range(self.M // 2):
            root = self.xi ** (2 * i + 1)
            row = np.array([], dtype=np.complex128)

            for j in range(self.M // 2):
                row = np.append(row, root ** j)

            matrix = np.vstack([matrix, row])

        return matrix

    def sigma_inverse(self, b: np.ndarray[Any, np.dtype[np.complex128]]) -> Polynomial:
        # Melakukan encoding dari vector ke polynomial
        A = self.vandermonde()

        # Mencari solusi dari Ax = b
        coefficients = np.linalg.solve(A, b)

        # Mengembalikan polynomial dengan koefisien yang telah ditemukan
        return Polynomial(coefficients)

    def sigma(self, p: Polynomial) -> np.ndarray[Any, np.dtype[np.complex128]]:
        # Melakukan decoding dari polynomial ke vector
        outputs = np.array([], dtype=np.complex128)
        N = self.M // 2

        # Mengaplikasikan polynomial ke M-th root of unity
        for i in range(N):
            root = self.xi ** (2 * i + 1)
            output = np.complex128(p(root))

            # Imajiner bernilai mendekati 0
            outputs = np.append(outputs, output)

        return outputs


if __name__ == "__main__":
    encoder = Encoder(8)

    b1 = np.array([1., 2., 3., 4.], dtype=np.complex128)
    p1 = encoder.sigma_inverse(b1)

    b2 = np.array([1., -2., 3., -4.], dtype=np.complex128)
    p2 = encoder.sigma_inverse(b2)

    b_add = b1 + b2
    b_mul = b1 * b2

    new_b1 = encoder.sigma(p1)
    print(new_b1)
    print(np.allclose(b1, new_b1))

    new_b2 = encoder.sigma(p2)
    print(new_b2)
    print(np.allclose(b2, new_b2))

    modulo = Polynomial([1, 0, 0, 0, 1])

    p_add = p1 + p2
    p_mul = p1 * p2 % modulo

    new_b_add = encoder.sigma(p_add)
    new_b_mul = encoder.sigma(p_mul)

    print(new_b_add)
    print(np.allclose(b_add, new_b_add))
    print(new_b_mul)
    print(np.allclose(b_mul, new_b_mul))
