# LEAGAN DAN ADRIAN - TABLAN ANDREI RAZVAN - TEMA 2
import numpy as np


def machine_epsilon():
    eps = np.finfo(float).eps
    return eps


def lower_triangular(A):
    n = A.shape[0]  # get the size of the matrix A
    L = np.zeros_like(A)  # create a matrix of zeros with the same shape as A

    # fill the lower triangular part of L with the corresponding entries of A
    for i in range(n):
        for j in range(i + 1):
            L[i, j] = A[i, j]
    for i in range(n):
        for j in range(n):
            if i==j :
                L[i, j] = 1
    print(L)
    print("a fost L aici")
    return L



def get_transpose(A):
    return np.transpose(A)


def get_diagonal_matrix(A):
    return np.diag(np.diag(A))


def multiply_matrices(A, B):
    return A.dot(B)


def cholesky(A):
    L= np.linalg.cholesky(A)
    return L


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    A = np.array([[1, 2, 3], [2, 1, 5], [3, 5, 1]])
    L = lower_triangular(A)
    D = get_diagonal_matrix(A)
    LT = get_transpose(L)
    print(LT)
    A1 = multiply_matrices(multiply_matrices(L, D), LT)
    #print(cholesky(A))
    print("-------------------")
    print(A1)
