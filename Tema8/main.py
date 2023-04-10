import math
import numpy as np


def secant_method(f_derivative, x0, x, epsilon, max_iterations):
    k = 0
    delta = 0
    g_x = f_derivative(x)
    g_x0 = f_derivative(x0)
    delta = ((x - x0) * g_x) / (g_x - g_x0)
    if -epsilon <= (g_x - g_x0) <= epsilon:
        if abs(g_x) <= epsilon / 100:
            delta = 0
        else:
            delta = 10 ** (-5)
    x0 = x
    x = x - delta
    k += 1
    while k < max_iterations and delta <= 10 ** 8 and abs(delta >= epsilon):
        g_x = f_derivative(x)
        g_x0 = f_derivative(x0)
        delta = ((x - x0) * g_x) / (g_x - g_x0)
        if -epsilon <= (g_x - g_x0) <= epsilon:
            if abs(g_x) <= epsilon / 100:
                delta = 0
            else:
                delta = 10 ** (-5)
        x0 = x
        x = x - delta
        k += 1
    if abs(delta) >= epsilon:
        return x
    else:
        return "DIVERGENTA"


def f1(x):
    return 1 / 3 * x ** 3 - 2 * x ** 2 + 2 * x + 3


def f1_derivative(x):
    return x ** 2 - 4 * x + 2


def f2(x):
    return math.sin(x) + x ** 2


def f2_derivative(x):
    x1 = x - 0.1
    x2 = x + 0.1
    y1 = f2(x1)
    y2 = f2(x2)
    return (y2 - y1) / (x2 - x1)


def f3(x):
    return x**4 - 6*x**3 + 13*x - 12*x + 4


def f3_derivative(x):
    return 4*x**3 - 18*x**2 + 1


if __name__ == '__main__':
    x0 = 0
    x1 = -0.5
    epsilon = 10 ** -5
    max_iterations = 100
    x_min = secant_method(f1_derivative, x0, x1, epsilon, max_iterations)
    print(x_min)

    x0 = 3.0
    x1 = 4.0
    epsilon = 10 ** -5
    max_iterations = 100
    x_min = secant_method(f2_derivative, x0, x1, epsilon, max_iterations)
    print(x_min)

    x0 = 0.5
    x1 = 2.5
    epsilon = 10 ** -5
    max_iterations = 100
    x_min = secant_method(f3_derivative, x0, x1, epsilon, max_iterations)
    print(x_min)
