# FII - 3A5 - Tema 1 - Leagan Dan & Tablan Andrei
import random

import numpy as np


def ex_1(u):
    while (1 + u) != 1:
        u = u / 10
    return u


def ex_2a(x, y, z):
    return (x + y) + z == x + (y + z)


def multiply(a, b):
    return a * b


def verify(x, y, z):
    a = multiply(multiply(x, y), z)
    b = multiply(x, multiply(y, z)),
    return a, b, a == b


def ex_2b(u):
    x = u
    y = np.random.uniform(0.0, 10.2)
    z = np.random.uniform(0.0, 10.2)
    print("The numbers are: ", x, y, z)
    return verify(x, y, z)


# exercise 3

def create_matrix(n):
    size = 2 ** n
    matrix = []

    for row in range(0, size):
        row_list = []
        for col in range(0, size):
            row_list.append(random.randint(0, 9))
        matrix.append(row_list)

    return matrix


def split_matrix(matrix):
    n = len(matrix)
    if n % 2 != 0:
        raise ValueError("Matrix size must be even")

    half_n = n // 2
    top_left = [row[:half_n] for row in matrix[:half_n]]
    top_right = [row[half_n:] for row in matrix[:half_n]]
    bottom_left = [row[:half_n] for row in matrix[half_n:]]
    bottom_right = [row[half_n:] for row in matrix[half_n:]]

    return top_left, top_right, bottom_left, bottom_right


def multiply_list_of_lists(lists):
    result = [1] * len(lists[0])
    for lst in lists:
        for i in range(len(lst)):
            result[i] *= lst[i]
    return result


def combine_matrix(matrix1, matrix2, matrix3, matrix4):
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])
    rows3, cols3 = len(matrix3), len(matrix3[0])
    rows4, cols4 = len(matrix4), len(matrix4[0])

    if rows1 != rows2 or rows1 != rows3 or rows1 != rows4:
        raise ValueError("Matrices have different number of rows.")
    if cols1 != cols2 or cols1 != cols3 or cols1 != cols4:
        raise ValueError("Matrices have different number of columns.")

    top = [matrix1[i] + matrix2[i] for i in range(rows1)]
    bottom = [matrix3[i] + matrix4[i] for i in range(rows3)]
    result = top + bottom

    return result


def multiply_matrices(matrix1, matrix2):
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])
    result = []
    for i in range(rows1):
        row = []
        for j in range(cols1):
            row.append(0)
        result.append(row)
    for i in range(rows1):
        for j in range(cols1):
            for k in range(rows2):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result


def multiply_strassen(A, B, n_min):
    if len(A) == n_min:
        return multiply_matrices(A, B)

    a, b, c, d = split_matrix(A)
    e, f, g, h = split_matrix(B)

    p1 = multiply_strassen(a, subtracting_matrix(f, h), n_min)
    p2 = multiply_strassen(add_matrix(a, b), h, n_min)
    p3 = multiply_strassen(add_matrix(c, d), e, n_min)
    p4 = multiply_strassen(d, subtracting_matrix(g, e), n_min)
    p5 = multiply_strassen(add_matrix(a, d), add_matrix(e, h), n_min)
    p6 = multiply_strassen(subtracting_matrix(b, d), add_matrix(g, h), n_min)
    p7 = multiply_strassen(subtracting_matrix(a, c), add_matrix(e, f), n_min)

    c11 = add_matrix(subtracting_matrix(add_matrix(p5, p4), p2), p6)
    c12 = add_matrix(p1, p2)
    c21 = add_matrix(p3, p4)
    c22 = subtracting_matrix(subtracting_matrix(add_matrix(p1, p5), p3), p7)

    C = combine_matrix(c11, c12, c21, c22)
    return C


def add_matrix(matrix1, matrix2):
    matrix1_rows = len(matrix1)
    matrix2_rows = len(matrix2)
    matrix1_col = len(matrix1[0])
    matrix2_col = len(matrix2[0])

    if matrix1_rows != matrix2_rows or matrix1_col != matrix2_col:
        return "ERROR: dimensions of the two arrays must be the same"

    matrix = []
    rows = []

    for i in range(0, matrix1_rows):
        for j in range(0, matrix2_col):
            rows.append(0)
        matrix.append(rows.copy())
        rows = []

    for i in range(0, matrix1_rows):
        for j in range(0, matrix2_col):
            matrix[i][j] = matrix1[i][j] + matrix2[i][j]

    return matrix


def subtracting_matrix(matrix1, matrix2):
    matrix1_rows = len(matrix1)
    matrix2_rows = len(matrix2)
    matrix1_col = len(matrix1[0])
    matrix2_col = len(matrix2[0])

    if matrix1_rows != matrix2_rows or matrix1_col != matrix2_col:
        return "ERROR: dimensions of the two arrays must be the same"

    matrix = []
    rows = []

    for i in range(0, matrix1_rows):
        for j in range(0, matrix2_col):
            rows.append(0)
        matrix.append(rows.copy())
        rows = []

    for i in range(0, matrix1_rows):
        for j in range(0, matrix2_col):
            matrix[i][j] = matrix1[i][j] - matrix2[i][j]
    return matrix


def ex_3(n):
    A = create_matrix(n)
    B = create_matrix(n)
    A1 = [[3, 8, 5, 3, 4, 0, 5, 6], [1, 6, 0, 7, 7, 8, 0, 3], [0, 8, 4, 1, 5, 2, 9, 8], [5, 2, 1, 7, 5, 3, 9, 1],
          [3, 0, 1, 0, 3, 9, 6, 3], [4, 7, 7, 2, 2, 6, 7, 2], [1, 0, 6, 1, 0, 8, 0, 7], [7, 0, 6, 3, 5, 9, 8, 6]]

    B2 = [[9, 7, 8, 1, 0, 9, 2, 8], [7, 0, 2, 2, 6, 9, 8, 7], [1, 9, 3, 5, 3, 7, 8, 0], [3, 4, 9, 6, 8, 0, 2, 7],
          [6, 2, 0, 0, 3, 2, 5, 8], [8, 6, 9, 1, 6, 9, 4, 3], [2, 5, 7, 2, 9, 3, 8, 0], [0, 7, 7, 3, 9, 1, 5, 8]]

    A11 = [[2, 0], [0, 2]]
    B22 = [[1, 1], [1, 1]]
    n_min = 2
    C = multiply_strassen(A, B, n_min)
    print("----------A------------")
    print(A)
    print("----------B------------")
    print(B)
    print("----------C------------")
    print(C)
    print("---------A*B------------")
    print(multiply_matrices(A,B))


if __name__ == '__main__':
    # ex 1
    u = ex_1(1)
    print(u)

    # ex 2
    print("The answer for (x+y)+z==x+(y+z) is ", ex_2a(1.0, u, u))
    print("The answers for (xy)z and x(yz) are:", ex_2b(u))

    # ex 3
    ex_3(3)
