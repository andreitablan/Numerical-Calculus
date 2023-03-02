# FII - 3A5 - Leagan Dan & Tablan Andrei
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

def multiply_Strassen(A, B,n_min):
    """
    Computes matrix product by divide and conquer approach, recursively.
    Input: nxn matrices x and y
    Output: nxn matrix, product of x and y
    """

    if len(A) == 1:
        return [[A[0][0] * B[0][0]]]
    # Base case when size of matrices is 1x1

    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.

    a, b, c, d = split_matrix(A)
    e, f, g, h = split_matrix(B)


    # Computing the 7 products, recursively (p1, p2...p7)

    p1 = multiply_Strassen(a, subtracting_matrix(f, h),n_min)
    p2 = multiply_Strassen(add_matrix(a,b), h,n_min)
    p3 = multiply_Strassen(add_matrix(c,d), e,n_min)
    p4 = multiply_Strassen(d, subtracting_matrix(g,e),n_min)
    p5 = multiply_Strassen(add_matrix(a,d), add_matrix(e,h),n_min)
    p6 = multiply_Strassen(subtracting_matrix(b,d), add_matrix(g,h),n_min)
    p7 = multiply_Strassen(subtracting_matrix(a,c), add_matrix(e,f),n_min)

    # Computing the values of the 4 quadrants of the final matrix c
    c11 = add_matrix(subtracting_matrix(add_matrix(p5,p4),p2),p6)
    c12 = add_matrix(p1,p2)
    c21 = add_matrix(p3,p4)
    c22 = subtracting_matrix(subtracting_matrix(add_matrix(p1,p5),p3),p7)

    print(c11,c12,c21,c22)
    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    C = [[c11, c12], [c21, c22]]

    return C


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


def ex_3(n):
    A = create_matrix(n)
    B = create_matrix(n)
    n_min = 1
    C = multiply_Strassen(A, B,n_min)
    print(C)


if __name__ == '__main__':
    # ex 1
    u = ex_1(1)
    print(u)

    # ex 2
    print("The answer for (x+y)+z==x+(y+z) is ", ex_2a(1.0, u, u))
    print("The answers for (xy)z and x(yz) are:", ex_2b(u))

    # ex 3
    ex_3(2)
