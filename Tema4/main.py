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
            my_tuple = (number, j)
            if len(a[i]) != 0:
                for tuple_from_list in a[i]:
                    if tuple_from_list[1] == my_tuple[1]:
                        new_tuple = (tuple_from_list[1] + my_tuple[1], j)
                        a[i].remove(tuple_from_list)
                        a[i].append(new_tuple)
                    else:
                        a[i].append(my_tuple)
            else:
                a[i].append(my_tuple)
            new_list = sorted(a[i], key=lambda x: x[1])
            a[i]=[]
            a[i] = new_list.copy()
    lines2 = []
    with open(file2, "r") as file_in:
        for line in file_in:
            lines2.append(line)
    for index in range(1, n):
        number = float(lines2[index])
        b.append(number)
    return n, a, b


def verify_null_on_diagonals(a, n):
    for index in range(1, n):
        flag1 = 0
        flag2 = 0

        for element_tuple in a[index]:
            if element_tuple[1] == index:
                flag1 = 1
            if element_tuple[1] + index == n:
                flag2 = 1
        if flag1 == 0 and flag2 == 0:
            return False
    return True


if __name__ == '__main__':
    n, a, b = read_system("sis_liniar/a_5.txt", "sis_liniar/b_5.txt")
    print("The matrix has all elements on the diagonals !=0", verify_null_on_diagonals(a, n))
