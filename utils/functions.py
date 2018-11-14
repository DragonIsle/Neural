import numpy as np
import math


def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
    return sigmoid(x) * (1 - sigmoid(x))


def threshold(x):
    if x > 0:
        return 1
    else:
        return 0


def relu(x):
    return max(x, 0)


def softplus(x):
    return math.log1p(1 + np.exp(round(x, 3)))


def linear(x):
    return x


def linear_derivative(x):
    return 1


def quadratic(x):
    return x * x


def quadratic_der(x):
    return 2 * x
