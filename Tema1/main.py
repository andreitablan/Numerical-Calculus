# FII - 3A5 - Leagan Dan & Tablan Andrei
import numpy as np


def ex_1(u):
    while (1 + u) != 1:
        u = u / 10
    return u


def ex_2a(x, y, z):
    return (x + y) + z == x + (y + z)


def multiply(a, b):
    return a * b


def verify(x, y, z):
    a = multiply(multiply(x, y), z)
    b = multiply(x, multiply(y, z)),
    return a, b, a == b


def ex_2b(u):
    x = u
    y = np.random.uniform(0.0, 10.2)
    z = np.random.uniform(0.0, 10.2)
    print("The numbers are: ", x, y, z)
    return verify(x, y, z)


if __name__ == '__main__':
    # ex 1
    u = ex_1(1)
    # print(u)

    # ex 2
    # print("The answer for (x+y)+z==x+(y+z) is ", ex_2a(1.0, u, u))
    print("The answers for (xy)z and x(yz) are:", ex_2b(u))
