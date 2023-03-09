# LEAGAN DAN ADRIAN - TABLAN ANDREI RAZVAN - TEMA 2
import numpy as np


def machine_epsilon():
    eps = np.finfo(float).eps
    return eps


def lower_triangular(A):
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if (i == j):
                L[i][j] = np.sqrt(max(A[i][i] - s, 0))
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - s))
    return L


def change_diagonal(L):
    n = len(L)
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i][j] = 1
    return L


def get_transpose(A):
    return np.transpose(A)


def get_diagonal_matrix(A):
    return np.diag(np.diag(A))


def multiply_matrices(A, B):
    return A.dot(B)


def cholesky_decomposition(A):
    n = len(A)
    L = [[0.0] * n for i in range(n)]
    D = [0.0] * n

    for i in range(n):
        for j in range(i):
            s = 0.0
            for k in range(j):
                s += L[i][k] * L[j][k] * D[k]
            L[i][j] = (A[i][j] - s) / D[j]
        s = 0.0
        for k in range(i):
            s += L[i][k] ** 2 * D[k]
        D[i] = A[i][i] - s
        L[i][i] = 1.0

    LT = [[L[j][i] for j in range(n)] for i in range(n)]

    B = [[0.0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(min(i, j) + 1):
                s += L[i][k] * D[k] * LT[k][j]
            B[i][j] = s

    return L, D, LT, B


if __name__ == '__main__':
    A = [[4, 6, 10], [6, 25, 19], [10, 19, 94]]

    '''
    L = np.linalg.cholesky(A)

    D=get_diagonal_matrix(L)
    LT=get_transpose(L)
    L=change_diagonal(L)
    print(A)
    print(D)
    print(L)
    print(LT)
    print(multiply_matrices(multiply_matrices(L,D),LT))
    '''

    L, D, LT, B = cholesky_decomposition(A)
    n = len(A)
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
    print("A= ",A)
    print("L = ", L)
    print("D = ", D)
    print("LT = ", LT)
    print("B = ", B)
    print("det A = ", det_A)