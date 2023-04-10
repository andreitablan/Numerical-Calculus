import numpy as np
import math


def verify_divide(number):
    if abs(number) <= 10 ** (-6):
        raise ValueError("CANNOT DIVIDE!")

def calculate_norm_vectors(x, y):
    n = len(x)
    sum1 = 0
    # x, y = normalize_array(x, y)
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


if __name__ == '__main__':
    n1, a1 = read_matrix_only("sisteme/m_rar_sim_2023_512.txt")


