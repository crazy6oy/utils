import numpy as np


def kappa_equation(matrix: np.ndarray) -> int:
    matrix = matrix.astype(np.uint64)
    unit_matrix = np.identity(matrix.shape[0])
    po = np.sum(matrix * unit_matrix) / np.sum(matrix)
    pe = np.sum(np.sum(matrix, axis=0) * np.sum(matrix.T, axis=0)) / (np.square(np.sum(matrix)))
    return ((po - pe) / (1 - pe)).item()


def list_2_2D_matrix(first_dimension_data: list, second_dimension_data: list) -> np.ndarray:
    matrix_size = max(first_dimension_data + second_dimension_data) + 1

    matrix = np.zeros((matrix_size, matrix_size), dtype=np.uint64)
    if len(second_dimension_data) != len(first_dimension_data):
        raise ValueError("两序列长度不一致！")

    for i in range(len(first_dimension_data)):
        if first_dimension_data[i] < 0 or second_dimension_data[i] < 0:
            continue
        try:
            matrix[first_dimension_data[i], second_dimension_data[i]] += 1
        except IndexError:
            raise IndexError("input must be int type")
    return matrix


if __name__ == '__main__':
    a = [0, 2, 1, 1, -1, 1]
    b = [0, 2, -1, 2, -1, 1]

    cm = list_2_2D_matrix(a, b)
    kappa_value = kappa_equation(cm)
    stop = 0
