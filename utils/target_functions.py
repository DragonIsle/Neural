from math import log

import numpy as np


def j_quadratic(y_hat, y):
    """
    Оценивает значение квадратичной целевой функции.

    y - матрица правильных ответов (n, N)
    y_hat - матрица предсказаний (n, N)
    Возвращает значение J (число)
    """

    return 0.5 * np.mean((y_hat - y) ** 2)


def j_quadratic_derivative(y_hat, y):
    """
    Вычисляет матрицу частных производных целевой функции по каждому из предсказаний.
    y - матрица правильных ответов (n, N)
    y_hat - матрица предсказаний (n, N)
    """

    return (y_hat - y) / (len(y) * y.shape[1])


def j_cross_entropy(y_hat, y):
    return -1 * np.mean(y * np.vectorize(log)(y_hat) + (1 - y) * np.vectorize(log)(1 - y_hat))


def j_cross_entropy_derivative(y_hat, y):
    return (y_hat - y) / (y_hat * (1 - y_hat) * len(y) * y.shape[1])
