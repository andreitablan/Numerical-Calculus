# FII - 3A5 - Leagan Dan & Tablan Andrei
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
def add_matrix(matrix1, matrix2):
    matrix1_rows = len(matrix1)
    matrix2_rows = len(matrix2)
    matrix1_col = len(matrix1[0])
    matrix2_col = len(matrix2[0])

    # base case
    if matrix1_rows != matrix2_rows or matrix1_col != matrix2_col:
        return "ERROR: dimensions of the two arrays must be the same"

    # make a matrix of the same size as matrix 1 and matrix 2
    matrix = []
    rows = []

    for i in range(0, matrix1_rows):
        for j in range(0, matrix2_col):
            rows.append(0)
        matrix.append(rows.copy())
        rows = []

    # loop through the two matricies and the summation should be placed in the
    # matrix
    for i in range(0, matrix1_rows):
        for j in range(0, matrix2_col):
            matrix[i][j] = matrix1[i][j] + matrix2[i][j]

    return matrix


def subtracting_matrix(matrix1, matrix2):
    matrix1_rows = len(matrix1)
    matrix2_rows = len(matrix2)
    matrix1_col = len(matrix1[0])
    matrix2_col = len(matrix2[0])

    # base case
    if matrix1_rows != matrix2_rows or matrix1_col != matrix2_col:
        return "ERROR: dimensions of the two arrays must be the same"

    # make a matrix of the same size as matrix 1 and matrix 2
    matrix = []
    rows = []

    for i in range(0, matrix1_rows):
        for j in range(0, matrix2_col):
            rows.append(0)
        matrix.append(rows.copy())
        rows = []

    # loop through the two matricies and the summation should be placed in the
    # matrix
    for i in range(0, matrix1_rows):
        for j in range(0, matrix2_col):
            matrix[i][j] = matrix1[i][j] - matrix2[i][j]
    return matrix


def ex_3():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]

    A11 = A[0][0]
    A12 = A[0][1]
    A21 = A[1][0]
    A22 = A[1][1]
    B11 = B[0][0]
    B12 = B[0][1]
    B21 = B[1][0]
    B22 = B[1][1]
    P1 = (A11 + A22) * (B11 + B22)
    P2 = (A21 + A22) * B11
    P3 = A11 * (B12 - B22)
    P4 = A22 * (B21 - B11)
    P5 = (A11 + A12) * B22
    P6 = (A21 - A11) * (B11 + B12)
    P7 = (A12 - A22) * (B21 + B22)

    C = [[P1 + P4 - P5 + P7, P3 + P5], [P2 + P4, P1 + P3 - P2 + P6]]
    return C


if __name__ == '__main__':
    # ex 1
    u = ex_1(1)
    print(u)

    # ex 2
    print("The answer for (x+y)+z==x+(y+z) is ", ex_2a(1.0, u, u))
    print("The answers for (xy)z and x(yz) are:", ex_2b(u))

    # ex 3
    print(ex_3())
