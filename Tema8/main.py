import math
import numpy as np


def secant_method(f, derivate, x0, x, epsilon, max_iterations):
    k = 0
    flag = 0
    delta = 0
    g_x = derivate(f, x)
    g_x0 = derivate(f, x0)
    delta = ((x - x0) * g_x) / (g_x - g_x0)
    if -epsilon <= (g_x - g_x0) <= epsilon:
        if abs(g_x) <= epsilon / 100:
            delta = 0
            flag = 1
        else:
            delta = 10 ** (-5)
    x0 = x
    x = x - delta
    k += 1
    if flag == 0:
        while k < max_iterations and abs(delta) <= 10 ** 8 and abs(delta) >= epsilon:
            g_x = derivate(f, x)
            g_x0 = derivate(f, x0)
            delta = ((x - x0) * g_x) / (g_x - g_x0)
            if -epsilon <= (g_x - g_x0) <= epsilon:
                if abs(g_x) <= epsilon / 100:
                    delta = 0
                    break
                else:
                    delta = 10 ** (-5)
            x0 = x
            x = x - delta
            k += 1
    if abs(delta) < epsilon:
        return x,k
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
    return x ** 4 - 6 * (x ** 3) + 13 * (x ** 2) - 12 * x + 4


def f3_derivative(x):
    return 4 * (x ** 3) - 18 * (x ** 2) + 26 * x - 12


def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h


def second_derivative(f, x, h=1e-5):
    return (derivative(f, x + h, h) - derivative(f, x, h)) / h


def derivative_1_1(f, x, h=1e-5):
    g1 = (3 * f(x) - 4 * f(x - h) + f(x - 2 * h)) / (2 * h)
    return g1


def derivative_1_2(f, x, h=1e-5):
    g2 = (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)
    return g2


def derivative_2(f, x, h=1e-05):
    F = (-f(x + 2 * h) + 16 * f(x + h) - 30 * f(x) + 16 * f(x - h) - f(x - 2 * h)) / (12 * (h ** 2))
    return F


if __name__ == '__main__':
    x0 = 3.0
    x1 = 4.0
    epsilon = 10 ** -4
    max_iterations = 1000
    x_min1,iterations1 = secant_method(f1,derivative_1_1,x0, x1, epsilon, max_iterations)
    x_min2,iterations2 = secant_method(f1,derivative_1_2,x0, x1, epsilon, max_iterations)
    print(x_min1, iterations1)
    print(x_min1, iterations2)
    if derivative_2(f1, x_min1)>0:
        print(x_min1, True)
    else:
        print(x_min1, False)
    if derivative_2(f1, x_min2)>0:
        print(x_min2, True)
    else:
        print(x_min2, False)
    x0 = 0
    x1 = -0.5
    epsilon = 10 ** -4
    max_iterations = 1000
    x_min1, iterations1 = secant_method(f2, derivative_1_1, x0, x1, epsilon, max_iterations)
    x_min2, iterations2 = secant_method(f2, derivative_1_2, x0, x1, epsilon, max_iterations)
    print(x_min1, iterations1)
    print(x_min1, iterations2)
    if derivative_2(f2, x_min1) > 0:
        print(x_min1, True)
    else:
        print(x_min1, False)
    if derivative_2(f2, x_min2) > 0:
        print(x_min2, True)
    else:
        print(x_min2, False)

    x0 = 1
    x1 = 2
    epsilon = 10 ** -4
    max_iterations = 1000
    x_min1, iterations1 = secant_method(f3, derivative_1_1, x0, x1, epsilon, max_iterations)
    x_min2, iterations2 = secant_method(f3, derivative_1_2, x0, x1, epsilon, max_iterations)
    print(x_min1, iterations1)
    print(x_min1, iterations2)
    if derivative_2(f3, x_min1) > 0:
        print(x_min1, True)
    else:
        print(x_min1, False)
    if derivative_2(f3, x_min2) > 0:
        print(x_min2, True)
    else:
        print(x_min2, False)
