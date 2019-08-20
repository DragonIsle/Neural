import math

import numpy as np


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
    return np.vectorize(math.log1p)(1 + np.exp(round(x, 3)))


def linear(x):
    return x


def linear_derivative(x):
    return 1


def quadratic(x):
    return x * x


def quadratic_der(x):
    return 2 * x


def softmax(x):
    """Только для векторов"""
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


def softmax_prime(x):
    """Только для векторов"""
    return softmax(x) * (1 - softmax(x))
