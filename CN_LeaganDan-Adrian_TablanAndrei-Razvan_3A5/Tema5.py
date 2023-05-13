# FII - 3A5 - Tema 5 - Leagan Dan & Tablan Andrei

import math
import random
import numpy as np


def verify_divide(number):
    if abs(number) <= 10 ** (-6):
        raise ValueError("CANNOT DIVIDE!")


def calculate_norm_vectors(x, y):
    n = len(x)
    sum1 = 0
    for i in range(n):
        sum1 += abs(x[i] - y[i])
    norm = math.sqrt(sum1)
    return norm


def read_matrix_only(file1):
    lines = []
    with open(file1, "r") as file_in:
        for line in file_in:
            lines.append(line)
    n = int(lines[0])
    a = [[] for i in range(n)]
    for line in lines:
        elements = line.split(",")
        if len(elements) > 1:
            number = float(elements[0].strip())
            i = int(elements[1].strip())
            j = int(elements[2].strip())
            my_list = [number, j]
            if len(a[i]) != 0:
                ok = False
                for k in range(0, len(a[i])):
                    element_list = a[i][k]
                    if element_list[1] == my_list[1]:
                        suma = element_list[0] + my_list[0]
                        a[i][k] = []
                        a[i][k] = [suma, j]
                        ok = True
                        break
                if ok is False:
                    a[i].append(my_list)
                new_list = sorted(a[i], key=lambda x: x[1])
                a[i] = []
                a[i] = [elem for elem in new_list]
            else:
                a[i].append(my_list)
    return n, a


def generate_random_matrix(n):
    a = [[] for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            value2 = random.choice([True, False])
            if value2:
                number = round(random.uniform(0.001, 1000), 2)
                new_element_1 = [number, j]
                a[i].append(new_element_1)
                if i != j:
                    new_element_2 = [number, i]
                    a[j].append(new_element_2)
    for i in range(n):
        new_list = sorted(a[i], key=lambda x: x[1])
        a[i] = []
        a[i] = [elem for elem in new_list]
    return a


# metoda puterii aproximare
# valorile proprii de modul maxim la toate matricile
def calculate_euclidian_norm(x):
    n = len(x)
    sum = 0
    for i in range(n):
        sum += abs(x[i]) ** 2
    norm = math.sqrt(sum)
    return norm


def generate_random_vector(dim):
    v = np.random.randn(dim)
    v = v / np.linalg.norm(v)
    return v


def multiply_matrix_vector(A, vector):
    vector_output = []
    p = len(A)
    n = len(A[0])
    for i in range(0, p):
        sum = 0.0
        for j in range(0, n):
            sum += A[i][j] * vector[j]
        vector_output.append(sum)
    return vector_output


def multiply_matrix_vector_normal(A, x):
    m, n = len(A), len(x)
    if n != len(A[0]):
        raise ValueError("Matrix and vector have incompatible dimensions")
    y = [0] * m
    for i in range(m):
        y[i] = sum(A[i][j] * x[j] for j in range(n))
    return y


def multiply_two_matrices(a, b):
    m1, n1 = len(a), len(a[0])
    m2, n2 = len(b), len(b[0])

    if n1 != m2:
        raise ValueError("Matrices cannot be multiplied")

    result = [[0] * n2 for _ in range(m1)]

    for i in range(m1):
        for j in range(n2):
            for k in range(n1):
                result[i][j] += a[i][k] * b[k][j]

    return result


def multiply_two_matrices_not_square(A, B):
    if len(A[0]) != len(B):
        print(len(A[0]), len(B))
        raise ValueError("Number of columns in A must be equal to number of rows in B")
    m, n = len(A), len(A[0])
    n, p = len(B), len(B[0])

    C = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def metoda_puterii(a):
    epsilon = 10 ** (-3)
    kmax = 1000000
    n = len(a)
    v = generate_random_vector(n)

    w = multiply_matrix_vector(a, v)
    w = np.array(w)
    lam = np.dot(w, v)
    k = 0

    v = (1 / np.linalg.norm(w)) * w
    w = multiply_matrix_vector(a, v)
    w = np.array(w)
    lam = np.dot(w, v)
    k += 1
    while np.linalg.norm(w - lam * v) > n * 10 ** (-9) and k <= kmax:
        v = (1 / np.linalg.norm(w)) * w
        w = multiply_matrix_vector(a, v)
        w = np.array(w)
        lam = np.dot(w, v)
        k += 1

    print(k)
    print(lam)
    print(v)


def verify_matrix(a, n):
    for i in range(n):
        for element in a[i]:
            number = element[0]
            j = element[1]
            ok = False
            for element_1 in a[j]:
                if element_1[1] == i:
                    ok = True
                    if abs(element_1[0] - number) > 10 ** (-6):
                        print(i, j, element_1[0], element[0])
                        return False
            if ok is False:
                print(i, j, element[0])
                return False
    return True


def verify_null_on_diagonals(a, n):
    for index in range(1, n):
        flag1 = 0
        for element_tuple in a[index]:
            if element_tuple[1] == index:
                if element_tuple[0] != 0:
                    flag1 = 1
        if flag1 == 0:
            return False
    return True


def matrix_rank(s):
    rank = len([x for x in s if abs(x) > 1e-12])
    return rank


def inverse(A):
    p = len(A)
    n = len(A[0])

    I = [[0 for x in range(p)] for y in range(p)]
    for i in range(p):
        I[i][i] = 1

    augmented_matrix = [row_A + row_I for row_A, row_I in zip(A, I)]

    def row_operation(matrix, i, j, factor):
        for k in range(len(matrix[0])):
            matrix[i][k] -= factor * matrix[j][k]

    # Perform row operations to transform the left-hand side of the augmented matrix into the identity matrix
    for i in range(p):
        element = augmented_matrix[i][i]
        for j in range(i + 1, p):
            factor = augmented_matrix[j][i] / element
            row_operation(augmented_matrix, j, i, factor)

        for j in range(i):
            factor = augmented_matrix[j][i] / element
            row_operation(augmented_matrix, j, i, factor)

        for j in range(n):
            augmented_matrix[i][j] /= element

    inverse_A = []
    for i in range(p):
        inverse_A.append(augmented_matrix[i][n:])
    return inverse_A


def vector_matrix_norm(x, A):
    m, n = len(A), len(A[0])
    if n != len(x):
        raise ValueError("Vector and matrix have incompatible dimensions")
    norm = 0
    for i in range(m):
        for j in range(n):
            norm += (A[i][j] - x[j]) ** 2

    return math.sqrt(norm)


def matrix_manhattan_norm(matrix):
    col_sums = [0] * len(matrix[0])
    for row in matrix:
        for i in range(len(row)):
            col_sums[i] += abs(row[i])
    return max(col_sums)


def matrix_euclidean_norm(A):
    p = len(A)
    n = len(A[0])
    norm = 0
    for i in range(p):
        for j in range(n):
            norm += A[i][j] ** 2
    return math.sqrt(norm)


def transpose(matrix):
    m = len(matrix)
    n = len(matrix[0])

    transpose_matrix = [[0] * m for _ in range(n)]

    for i in range(m):
        for j in range(n):
            transpose_matrix[j][i] = matrix[i][j]

    return transpose_matrix


def subtract_matrices(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        return "Error: Matrices must have the same dimensions."

    result = [[0 for j in range(len(matrix1[0]))] for i in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result[i][j] = matrix1[i][j] - matrix2[i][j]

    return result


def multiply_vector_with_matrix(x, A):
    m = len(A)
    n = len(A[0])
    p = len(x)

    result = [0] * m

    for i in range(m):
        for j in range(n):
            result[i] += A[i][j] * x[j]
    return result


def multiply_vectors(x, y):
    n = len(x)
    result = 0
    for i in range(n):
        result += x[i] * y[i]
    return result


def create_diagonal_matrix(s, p, n):
    if len(s) > n:
        raise ValueError("Length of s cannot be greater than n")
    S = np.zeros((p, n))
    for i in range(len(s)):
        S[i, i] = s[i]
    return S


def create_second_diagonal_matrix(s, p, n):
    if len(s) > n:
        raise ValueError("Length of s cannot be greater than n")
    S = np.zeros((n, p))
    for i in range(len(s)):
        S[i, i] = 1 / s[i]
    return S


def svd(p, n):
    A = np.random.rand(p, n)
    b = np.random.rand(p)
    U, s, VT = np.linalg.svd(A)
    S = create_second_diagonal_matrix(s, p, n)

    # S=np.diag([1/s_i if s_i != 0 else 0 for s_i in s])
    print("Valorile singulare ale matricei A: ")
    print(s)

    rank_A = matrix_rank(s)
    print("Matrix A rank: ", rank_A)

    V = transpose(VT)
    UT = transpose(U)

    print(multiply_two_matrices_not_square(S, UT))
    AI = multiply_two_matrices_not_square(V, multiply_two_matrices_not_square(S, UT))

    cond_A = matrix_euclidean_norm(A) * matrix_euclidean_norm(AI)
    print("Numarul de conditionare al matricei A: ", cond_A)
    print("Matrix AI = VSUT: ")
    print(AI)

    A_inv = np.linalg.pinv(A)
    print("A_inv")
    print(A_inv)

    xi = multiply_matrix_vector(AI, b)
    print(xi)

    norm2 = calculate_norm_vectors(b, multiply_matrix_vector(A, xi))
    print("Norm 2: ", norm2)

    AT = transpose(A)
    ATA = multiply_two_matrices_not_square(AT, A)
    ATA_inv = np.linalg.inv(ATA)
    AJ = multiply_two_matrices_not_square(ATA_inv, AT)
    print("Matrix AJ: ")
    print(AJ)
    A_prime = subtract_matrices(AI, AJ)
    norm1 = matrix_manhattan_norm(A_prime)
    print("Norm 1:", norm1)


if __name__ == '__main__':
    '''
    n512, a512 = read_matrix_only("sisteme/m_rar_sim_2023_512.txt")
    print("The matrix 512 has A=AT:", verify_matrix(a512, n512))
    n1024, a1024 = read_matrix_only("sisteme/m_rar_sim_2023_1024.txt")
    print("The matrix 1024 has A=AT:", verify_matrix(a1024, n1024))
    n2023, a2023 = read_matrix_only("sisteme/m_rar_sim_2023_2023.txt")
    print("The matrix 2023 has A=AT:", verify_matrix(a2023, n2023))
    a_generated = generate_random_matrix(3)
    a_generated=[[[131.89, 1]], [[131.89, 0]], [[518.78, 2]]]
    
    a_generated = a1024
    print("A generated", a_generated)
    print("Metoda Puterii")
    metoda_puterii(a_generated)
    '''
    p = 11
    n = 10
    svd(p, n)
