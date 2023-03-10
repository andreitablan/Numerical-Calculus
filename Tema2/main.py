# LEAGAN DAN ADRIAN - TABLAN ANDREI RAZVAN - TEMA 2
import numpy as np
import random

global epsilon


def set_epsilon(number):
    global epsilon
    epsilon = number


def get_transpose(A):
    return np.transpose(A)


def get_diagonal_matrix(A):
    return np.diag(np.diag(A))


def multiply_matrices(A, B):
    return A.dot(B)


def verify_matrix(A):
    for line in A:
        for element in line:
            if element < 0:
                raise ValueError("A is not positive!")


def verify_divide(number):
    global epsilon
    if abs(number) <= epsilon:
        raise ValueError("CANNOT DIVIDE!")


def random_symmetric_matrix(n):
    A = [[0.0] * n for i in range(n)]
    for i in range(n):
        for j in range(i + 1):
            A[i][j] = random.randint(1, 100)
            A[j][i] = A[i][j]
    return A


def random_vector(n):
    b = [0.0] * n
    for i in range(n):
        b[i] = random.randint(1, 100)
    return b


def cholesky_decomposition_with_L_D_LT(A_init):
    verify_matrix(A_init)
    n = len(A_init)
    A = [[0.0] * n for i in range(n)]
    L = [[0.0] * n for i in range(n)]
    D = [0.0] * n

    for i in range(n):
        for j in range(i):
            s = 0.0
            for k in range(j):
                s += L[i][k] * L[j][k] * D[k]
            verify_divide(D[j])
            L[i][j] = (A_init[i][j] - s) / D[j]
        s = 0.0
        for k in range(i):
            s += L[i][k] ** 2 * D[k]

        D[i] = A_init[i][i] - s
        L[i][i] = 1.0

    LT = [[L[j][i] for j in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(i + 1):
                A[i][j] += L[i][k] * D[k] * L[j][k]

    return L, D, LT, A


def cholesky_decomposition(A_init):
    verify_matrix(A_init)
    n = len(A_init)
    A = [[0.0] * n for i in range(n)]
    D = [0.0] * n

    for i in range(n):
        for j in range(i):
            s = 0.0
            for k in range(j):
                s += A[i][k] * A[j][k] * D[k]
            verify_divide(D[j])
            A[i][j] = s + ((A_init[i][j] - s) / D[j]) * D[j]
            A[j][i] = A[i][j]
        s = 0.0
        for k in range(i):
            s += A[i][k] ** 2 * D[k]

        D[i] = A_init[i][i] - s
        A[i][i] = s + D[i]
    return A


def calculate_determinant(A_init):
    verify_matrix(A_init)
    L, D, LT, A = cholesky_decomposition_with_L_D_LT(A_init)
    n = len(A_init)
    det_L = 1.0
    for i in range(n):
        det_L *= L[i][i]

    det_D = 1.0
    for i in range(n):
        det_D *= D[i]

    det_LT = 1.0
    for i in range(n):
        det_LT *= L[i][i]

    det_A = det_L * det_D * det_LT
    print("det A = ", det_A)


def cholesky_solve(A, b):
    global epsilon
    n = len(A)
    x = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += A[i][j] * x[j]
        verify_divide(A[i][i])
        x[i] = (b[i] - s) / A[i][i]

    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i + 1, n):
            s += A[i][j] * y[j]
        verify_divide(A[i][i])
        y[i] = (b[i] - s) / A[i][i]
    return x, y


def cholesky_solve_bonus_y(D, z):
    global epsilon
    n = len(D)
    y = [0.0] * n
    for i in range(n):
        verify_divide(D[i])
        y[i] = z[i] / D[i]
    return y


def solve_system_bonus(A_init, b):
    verify_matrix(A_init)
    L, D, LT, A = cholesky_decomposition_with_L_D_LT(A_init)
    z, z1 = cholesky_solve(L, b)
    y = cholesky_solve_bonus_y(D, z)
    x1, x = cholesky_solve(LT, y)
    return x


if __name__ == '__main__':
    m = int(input("m="))
    set_epsilon(10 ** -m)
    A_init = [[1, 2.5, 3], [2.5, 8.25, 15.5], [3, 15.5, 43]]
    b = [12, 38, 68]
    # n = int(input("n="))
    # A_init = random_symmetric_matrix(n)
    # b = random_vector(n)

    A = cholesky_decomposition(A_init)
    print("A =", A)
    calculate_determinant(A)
    x_chol, y_chol = cholesky_solve(A, b)
    print("b = ", b)
    print("x_chol = ", x_chol)
    print("y_chol = ", y_chol)

    residual = np.dot(A, x_chol) - b
    norm_residual = np.linalg.norm(residual, ord=2)
    print("norm_x = ", norm_residual)

    residual = np.dot(A, y_chol) - b
    norm_residual = np.linalg.norm(residual, ord=2)
    print("norm_y = ", norm_residual)

    print("-----------Numpy-------------")

    L = np.linalg.cholesky(A_init)
    U = get_transpose(L)
    print(" L = ", L)
    print(" U = ", U)
    print("A(from numpy)", multiply_matrices(L, U))
    x = np.linalg.solve(A_init, b)
    print("x=", x)

    print("-----------Bonus 2-------------")

    x_bonus=solve_system_bonus(A_init,b)
    print("x bonus = ", x_bonus)
