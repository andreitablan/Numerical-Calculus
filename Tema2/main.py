# LEAGAN DAN ADRIAN - TABLAN ANDREI RAZVAN - TEMA 2
import numpy as np


def machine_epsilon():
    eps = np.finfo(float).eps
    return eps


def lower_triangular(A):
    n = len(A)
    L = np.zeros((n, n))
    for j in range(n):
        for i in range(j, n):
            if i == j:
                L[i][j] = np.sqrt(A[i][j] - sum(L[i][k] ** 2 for k in range(j)))
            else:
                L[i][j] = (A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]

    return L


def change_diagonal(L):
    n=len(L)
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


def cholesky_with_diagonal_out(A):
    n = len(A)
    L = np.linalg.cholesky(A)
    d = np.diag(L)
    L_prime = np.identity(n) + L - np.diag(d)
    D = np.diag(d)
    print(L)
    print(np.matmul(L_prime, D))


def cholesky(A):
    L= np.linalg.cholesky(A)
    return L


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    A = np.array([[2., 2., 3.],
                  [2., 6., 5.],
                  [3., 5., 8.]])
    L = lower_triangular(A)
    LT = get_transpose(L)
    cholesky_with_diagonal_out(A)
    print("-------------------")

