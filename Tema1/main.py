# FII - 3A5 - Leagan Dan & Tablan Andrei
import numpy as np


def ex_1(u):
    while (1 + u) != 1:
        u = u / 10
    return u


if __name__ == '__main__':
    print(ex_1(1))