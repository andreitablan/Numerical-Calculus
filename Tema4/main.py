import numpy as np
import math


def verify_divide(number):
    if abs(number) <= 10 ** (-6):
        raise ValueError("CANNOT DIVIDE!")


def normalize_array(x, y):
    maxim_x = max(x)
    maxim_y = max(y)
    if maxim_x == 0.0:
        maxim_x = maxim_y
    elif maxim_y == 0.0:
        maxim_y = maxim_x
    for element in x:
        verify_divide(maxim_x)
        element = element / maxim_x
    for element in y:
        verify_divide(maxim_y)
        element = element / maxim_y
    return x, y


def calculate_norm_vectors(x, y):
    n = len(x)
    sum1 = 0
    x, y = normalize_array(x, y)
    for i in range(n):
        sum1 += abs(x[i] - y[i])
    norm = math.sqrt(sum1)
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
        for list1 in matrix1[i]:
            for list2 in matrix2[i]:
                if list1[1] == list2[1]:
                    if abs(list1[0] - list2[0]) >= 10 ** (-6):
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
    norm = 0
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

        xc[i] = (b[i] - sum1 - sum2) / element_diagonal
    norm = calculate_norm_vectors(xc,xp)
    k += 1

    while epsilon <= norm <= 10 ** 8 and k <= k_max:
        xp = [elem for elem in xc]
        for i in range(0, n):
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
            xc[i] = (b[i] - sum1 - sum2) / element_diagonal
        norm = calculate_norm_vectors(xc, xp)
        k += 1
    if norm < epsilon:
        return xc
    else:
        return xp


def add_matrices_bonus(matrix1, matrix2):
    n = len(matrix1)
    new_matrix = [[] for i in range(n)]
    for i in range(n):
        for list_1 in matrix1[i]:
            flag = False
            for list_2 in matrix2[i]:
                if list_1[1] == list_2[1]:
                    flag = True
                    new_element = [list_1[0] + list_2[0], list_1[1]]
                    new_matrix[i].append(new_element)
            if flag is False:
                new_matrix[i].append(list_1)
    for i in range(n):
        for list_1 in matrix2[i]:
            flag = False
            for list_2 in new_matrix[i]:
                if list_1[1] == list_2[1]:
                    flag = True
            if flag is False:
                new_matrix[i].append(list_1)
    return new_matrix


if __name__ == '__main__':
    n, a, b = read_system("sis_liniar/a_5.txt", "sis_liniar/b_5.txt")
    print("The matrix has all elements on the diagonals !=0", verify_null_on_diagonals(a, n))
    if verify_null_on_diagonals(a, n):
        xc = solve_gauss_seidel(a, b, n)
        if xc == "DIVERGENTA":
            print("DIVERGENTA")
        else:
            a_xc = multiply_matrix_vector(a, xc)
            norm = calculate_norm_vectors(a_xc, b)
            print(norm)
    else:
        print("It cannot be solved")

    '''------BONUS-------'''
    n1, a1 = read_matrix_only("sis_liniar/a.txt")
    n2, a2 = read_matrix_only("sis_liniar/b.txt")
    n3, a3 = read_matrix_only("sis_liniar/aplusb.txt")

    bonus_matrix = add_matrices_bonus(a1, a2)
    if verify_matrices_equality(bonus_matrix, a3) is True:
        print("The matrices are equal")
    else:
        print("The matrices are not equal")
