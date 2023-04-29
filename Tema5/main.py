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


def multiply_matrix_vector(matrix, vector):
    vector_output = []
    for line_list in matrix:
        sum = 0.0
        for element in line_list:
            sum += element[0] * vector[element[1]]
        vector_output.append(sum)

    return vector_output


def metoda_puterii(a):
    epsilon = 10 ** (-3)
    kmax = 1000000
    n = len(a_generated)
    v = generate_random_vector(n)

    w = multiply_matrix_vector(a, v)
    w = np.array(w)
    lam = w * v
    k = 0

    v = (1 / np.linalg.norm(w)) * w
    w = multiply_matrix_vector(a, v)
    w = np.array(w)
    lam = w * v
    k += 1
    while np.linalg.norm(w - lam * v) > n * 10 ** (-9) and k <= kmax:
        v = (1 / np.linalg.norm(w)) * w
        w = multiply_matrix_vector(a, v)
        w = np.array(w)
        lam = w * v
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


def svd(p,n):
    A = np.random.rand(p, n)
    b= np.random.rand(p)
    U, s, VT = np.linalg.svd(A)

    print("Valorile singulare ale matricei A: ")
    print(s)

    rank_A = np.linalg.matrix_rank(A)
    print("Matrix A rank: ", rank_A)

    cond_A = np.linalg.cond(A)
    print("Numărul de condiționare al matricei A: ", cond_A)

    AI = np.zeros((A.shape[0], A.shape[1]))
    AI[:A.shape[1], :A.shape[1]] = np.diag(s)

    V = VT.T
    AI = np.dot(V, np.dot(AI, U.T))

    print("Matrix AI = VSUT: ")
    print(AI)

    xi = np.dot(AI, b)
    print(xi)

    norm2=np.linalg.norm(b-np.dot(A,xi))
    print("Norm 1: ", norm2)

    AT = A.T
    ATA = np.dot(AT, A)
    ATA_inv = np.linalg.inv(ATA)
    AJ = np.dot(ATA_inv, AT)
    print("Matrix AJ: ")
    print(AJ)

    norm1=np.linalg.norm(AI-AJ,ord=1)
    print("Norm 1:", norm1)


if __name__ == '__main__':
    n512, a512 = read_matrix_only("sisteme/m_rar_sim_2023_512.txt")
    print("The matrix 512 has A=AT:", verify_matrix(a512, n512))
    n1024, a1024 = read_matrix_only("sisteme/m_rar_sim_2023_1024.txt")
    print("The matrix 1024 has A=AT:", verify_matrix(a1024, n1024))
    n2023, a2023 = read_matrix_only("sisteme/m_rar_sim_2023_2023.txt")
    print("The matrix 2023 has A=AT:", verify_matrix(a2023, n2023))
    a_generated = generate_random_matrix(10)
    print("Metoda Puterii")
    metoda_puterii(a_generated)
    p=10
    n=10
    svd(p,n)