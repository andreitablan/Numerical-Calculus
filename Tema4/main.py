import numpy as np
import math


def verify_divide(number):
    if abs(number) <= 10 ** (-6):
        raise ValueError("CANNOT DIVIDE!")


def calculate_norm_vectors(x, y):
    n = len(x)
    sum = 0
    for i in range(n):
        sum += abs(x[i] - y[i]) ** 2
    norm = math.sqrt(sum)
    return norm


def multiply_matrix_vector(matrix, vector):
    result = []
    n = len(matrix)
    for list_of_lists in matrix:
        dot_product = 0
        for list_ in list_of_lists:
            dot_product += list_[0] * vector[list_[1]]
        result.append(dot_product)
    return result


def verify_matrices_equality(matrix1, matrix2):
    n = len(matrix1)
    for i in range(n):
        for j in range(n):
            if abs(matrix1[i][j] - matrix2[i][j]) >= 10 ** (-6):
                return False
    return True


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
                        print(element_list[0], my_list[0])
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


def read_system(file1, file2):
    b = []
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
                        print(element_list[0], my_list[0])
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

    lines2 = []
    with open(file2, "r") as file_in:
        for line in file_in:
            lines2.append(line)
    for index in range(0, n):
        number = float(lines2[index])
        b.append(number)
    return n, a, b


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


def solve_gauss_seidel(a, b, n):
    epsilon = 1e-6
    k_max = 10000
    k = 0
    xc = [0.0 for i in range(n)]
    xp = [0.0 for i in range(n)]

    xp = [elem for elem in xc]
    for i in range(n):
        sum1 = 0.0
        sum2 = 0.0
        element_diagonal = 0.0
        for element_list in a[i]:
            if element_list[1] < i:
                sum1 += element_list[0] * xc[element_list[1]]
            elif element_list[1] == i:
                element_diagonal = element_list[0]
            else:
                sum2 += element_list[0] * xp[element_list[1]]
        verify_divide(element_diagonal)

        xc[i] = (b[i] - sum1 - sum1) / element_diagonal
    norm = calculate_norm_vectors(xc, xp)
    k += 1

    while epsilon <= norm <= 10 ** 8 and k <= k_max:
        xp = [elem for elem in xc]
        for i in range(n):
            sum1 = 0.0
            sum2 = 0.0
            element_diagonal = 0.0
            for element_tuple in a[i]:
                if element_tuple[1] < i:
                    sum1 += element_tuple[0] * xc[element_tuple[1]]
                elif element_tuple[1] == i:
                    element_diagonal = element_tuple[0]
                else:
                    sum2 += element_tuple[0] * xp[element_tuple[1]]
            verify_divide(element_diagonal)
            xc[i] = (b[i] - sum1 - sum1) / element_diagonal
        norm = calculate_norm_vectors(xc, xp)
        k += 1
    if norm < epsilon:
        return xc
    else:
        return xc


def add_matrices_bonus(matrix1, matrix2):
    n = len(matrix1)
    new_matrix = [[] for i in range(n)]
    for i in range(n):
        for list_ in matrix1[i]:
            print("hi")


if __name__ == '__main__':
    n, a, b = read_system("sis_liniar/a_5.txt", "sis_liniar/b_5.txt")
    print("The matrix has all elements on the diagonals !=0", verify_null_on_diagonals(a, n))
    if verify_null_on_diagonals(a, n):
        xc = solve_gauss_seidel(a, b, n)
        a_xc = multiply_matrix_vector(a, xc)
        norm = calculate_norm_vectors(a_xc, b)
        print(norm)
    else:
        print("It cannot be solved")
