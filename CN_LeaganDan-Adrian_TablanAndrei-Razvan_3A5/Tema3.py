# FII - 3A5 - Tema 3 - Leagan Dan & Tablan Andrei

import math
import random
import numpy as np
import scipy.linalg

global epsilon


def set_epsilon(number):
    global epsilon
    epsilon = number


def calculate_b(A, s):
    n = len(A)
    b = [0] * n
    for i in range(0, n):
        sum = 0
        for j in range(0, n):
            sum += s[j] * A[i][j]
        b[i] = sum
    return b


def create_identity_matrix(n):
    identity_matrix = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        identity_matrix[i][i] = 1
    return identity_matrix


def transpose_vector(v):
    return [[x] for x in v]


def u_multiply_u_t(u):
    n = len(u)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = u[i] * u[j]
    return result


def verify_divide(number):
    global epsilon
    if abs(number) <= epsilon:
        raise ValueError("CANNOT DIVIDE!")


def subtract_matrix(matrix1, matrix2):
    matrix1_rows = len(matrix1)
    matrix2_rows = len(matrix2)
    matrix1_col = len(matrix1[0])
    matrix2_col = len(matrix2[0])

    if matrix1_rows != matrix2_rows or matrix1_col != matrix2_col:
        return "ERROR: size must be the same"

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


def multiply_matrix_with_scalar(matrix, scalar):
    for line in matrix:
        for element in line:
            element = element * scalar
    return matrix


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


def multiply_matrices_bonus(R, Q):
    n = len(R)
    result = np.zeros((n,n))
    for i in range(n):
        for j in range(i, n):
            for k in range(n):
                result[i][j] += R[j][k] * Q[k][i]
    return result


def transpose_matrix(A):
    n_rows = len(A)
    n_cols = len(A[0])
    result = [[0] * n_rows for _ in range(n_cols)]
    for i in range(n_rows):
        for j in range(n_cols):
            result[j][i] = A[i][j]
    return result


def qr_householder(A, b):
    global epsilon
    n = len(A)
    I = create_identity_matrix(n)
    Q = create_identity_matrix(n)

    for r in range(0, n - 1):
        Pr = [[0.0] * n for i in range(n)]
        sigma = 0.0
        for i in range(r, n):
            sigma += A[i][r] ** 2
        if sigma <= epsilon:
            break
        k = math.sqrt(sigma)
        if A[r][r] > 0:
            k = -k
        Beta = sigma - k * A[r][r]
        u = [0.0] * n
        u[r] = A[r][r] - k
        for i in range(r + 1, n):
            u[i] = A[i][r]
        '''
        V = u_multiply_u_t(u)
        V = multiply_matrix_with_scalar(V, 1 / Beta)
        Pr = subtract_matrix(I, V)
        A = multiply_matrices(Pr, A)
        '''
        for j in range(r + 1, n):
            gama = 0.0
            sum = 0.0
            for i in range(r, n):
                sum += u[i] * A[i][j]
            verify_divide(Beta)
            gama = sum / Beta
            for i in range(r, n):
                A[i][j] = A[i][j] - gama * u[i]
        A[r][r] = k;
        for i in range(r + 1, n):
            A[i][r] = 0
        gama = 0.0
        sum = 0.0
        for i in range(r, n):
            sum += u[i] * b[i]
        verify_divide(Beta)
        gama = sum / Beta

        for i in range(r, n):
            b[i] = b[i] - gama * u[i]

        for j in range(n):
            gama = 0.0
            sum = 0.0
            for i in range(r, n):
                sum += u[i] * Q[i][j]
            verify_divide(Beta)
            gama = sum / Beta
            for i in range(r, n):
                Q[i][j] = Q[i][j] - gama * u[i]
    Q = transpose_matrix(Q)
    return Q, A, b


def solve_upper_triangular_system(R, b):
    n = len(R)
    x = [0] * n

    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += R[i][j] * x[j]
        verify_divide(R[i][i])
        x[i] = (b[i] - s) / R[i][i]

    return x


def solve_upper_triangular_system_q(R, Q_transpose, b):
    n = len(R)
    m = len(Q_transpose[0])
    x = [0] * m

    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += R[i][j] * x[j]
        verify_divide(R[i][i])
        x[i] = sum(Q_transpose[i][k] * b[k] for k in range(m)) - s / R[i][i]

    return x


def matrix_vector_multiply(matrix, vector):
    result = [0] * len(matrix)
    for i in range(len(matrix)):
        row = matrix[i]
        for j in range(len(vector)):
            result[i] += row[j] * vector[j]
    return result


def calculate_norm_vectors(x, y):
    n = len(x)
    sum = 0
    for i in range(n):
        sum += abs(x[i] - y[i]) ** 2
    norm = math.sqrt(sum)
    return norm


def calculate_norm_one_vector(x):
    n = len(x)
    sum = 0
    for i in range(n):
        sum += x[i] ** 2
    norm = math.sqrt(sum)
    return norm


def calculate_norm_system(A, x, b):
    n = len(A)
    sum = 0
    for i in range(n):
        s_line = 0
        for j in range(n):
            s_line += A[i][j] * x[j]
        z = s_line - b[i]
        sum += abs(z) ** 2
    norm = math.sqrt(sum)
    return norm


def calculate_norms(A_init, x_householder, x_qr, b, s):
    norm = calculate_norm_system(A_init, x_householder, b_init)
    print("|| A_init x_householder - b_init||2 : ", norm)
    if norm < epsilon:
        print("The norm is less than 10^(-6)")
    else:
        print("The norm is not less than 10^(-6)")
    norm = calculate_norm_system(A_init, x_qr, b_init)
    print("|| A_init x_qr - b_init||2 : ", norm)
    if norm < epsilon:
        print("The norm is less than 10^(-6)")
    else:
        print("The norm is not less than 10^(-6)")
    norm = calculate_norm_vectors(x_householder, s) / calculate_norm_one_vector(s)
    print("|| x_householder - s||2 / ||s||2: ", norm)
    if norm < epsilon:
        print("The norm is less than 10^(-6)")
    else:
        print("The norm is not less than 10^(-6)")
    norm = calculate_norm_vectors(x_qr, s) / calculate_norm_one_vector(s)
    print("|| x_qr - s||2 / ||s||2: ", norm)
    if norm < epsilon:
        print("The norm is less than 10^(-6)")
    else:
        print("The norm is not less than 10^(-6)")


def calculate_norm_two_matrices(A, B):
    sum = 0.0
    for i in range(len(A)):
        for j in range(len(A[0])):
            sum += abs(A[i][j] - B[i][j]) ** 2
    euclidean_norm = math.sqrt(sum)
    return euclidean_norm


def random_matrix(n):
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(random.uniform(0, 100))
        matrix.append(row)
    return matrix


def random_vector(n):
    s = [0.0] * n
    for i in range(n):
        s[i] = random.randint(1, 100)
    return s


def calculate_inverse(Q, A):
    Qt = np.array(transpose_matrix(Q))
    n = len(A)
    # check determinant
    A_inverse = np.zeros((n, n))
    for j in range(n):
        b = Qt[:, j]
        x_star = solve_upper_triangular_system(A, b)
        A_inverse[:, j] = x_star
    return A_inverse


def calculate_limit(A, s):
    epsilon = 10**-5
    iterations = 0
    maximum_iterations = 100000
    b = calculate_b(A, s)
    Ak = [row.copy() for row in A]
    while True:
        Q, R, b = qr_householder(Ak, b)
        Akp1 = multiply_matrices_bonus(R, Q)
        norm = calculate_norm_two_matrices(Akp1, Ak)
        if norm <= epsilon:
            print("ok")
            return Akp1
        Ak = [row.copy() for row in Akp1]
        if iterations > maximum_iterations:
            return Akp1
        iterations += 1


if __name__ == '__main__':
    set_epsilon(10 ** -3)
    A_init = [[0, 0, 4], [1, 2, 3], [0, 1, 2]]
    s = [3, 2, 1]
    #n = 100
    #A_init= random_matrix(n)
    #s = random_vector(n)

    A_init_2 = [row.copy() for row in A_init]
    b_init = calculate_b(A_init, s)

    Q_numpy, R_numpy = np.linalg.qr(A_init)
    Qb = np.matmul(Q_numpy.T, b_init)
    x_qr = np.linalg.solve(R_numpy, Qb)
    '''
    print("---Numpy---")
    print("---Q---")
    for line in Q_numpy:
        print(line)
    print("---R---")
    for line in R_numpy:
        print(line)
    print("---x qr---")
    print(x_qr)
    '''
    print("---CALCULAT---")
    Q, A, b = qr_householder(A_init, b_init)
    x_householder = solve_upper_triangular_system(A, b)
    '''
    print("---Q---")
    for line in Q:
        print(line)
    print("---R---")
    for line in A:
        print(line)
    print("---x householder---")
    print(x_householder)
    '''
    norm = calculate_norm_vectors(x_qr, x_householder)
    print("---norm---")
    print(norm)

    print("-----------4--------------")
    calculate_norms(A_init, x_householder, x_qr, b_init, s)

    print("-----------5--------------")
    A_inverse = calculate_inverse(Q, A)
    print("Identity", multiply_matrices(A_init,A_inverse))
    A_inverse_numpy = scipy.linalg.inv(A_init_2)
    print("Identity numpy", multiply_matrices(A_inverse_numpy,A_init_2))

    norm_matrices = calculate_norm_two_matrices(A_inverse, A_inverse_numpy)
    '''
    print("A_inverse: ")
    for line in A_inverse:
        print(line)
    print("A_inverse is right calculated: ", multiply_matrices(A_init, A_inverse))
    print("A_inverse_numpy: ", A_inverse_numpy)
    '''
    print("Matrices norm:", norm_matrices)

    print("----------BONUS-------------")
    m = 2
    lower_triangle = np.random.rand(m, m)
    symmetric_matrix = lower_triangle + lower_triangle.T
    print("A: ", symmetric_matrix)
    rand_vector = np.random.rand(m)
    print("Limit : ", calculate_limit(symmetric_matrix, rand_vector))