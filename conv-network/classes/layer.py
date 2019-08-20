import random

import numpy as np

from classes.neuron import Neuron


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.intermediate_sums = np.array([[]])
        self.intermediate_activations = np.array([[]])

    def process_input(self, input_matrix):
        """
        :param input_matrix: Матрица входов для слоя, (n, m)
        :return: Матрица активаций на выходе слоя
        """

        sums = np.array([neuron.summatory(input_matrix).flatten()
                         for neuron in self.neurons]).T
        activations = np.array([v.activation(sums[:, i])
                                for i, v in enumerate(self.neurons)]).T
        self.intermediate_sums = sums
        self.intermediate_activations = activations
        return activations

    def get_acts_by_sums_derivative(self):
        """
        :return: Матрица производных активационных функций по суммам для каждого нейрона, (n, m)
        """
        return np.array([v.derivative(self.intermediate_sums[:, i])
                         for i, v in enumerate(self.neurons)]).T

    def update_mini_batch(self, x, errors, learning_rate):
        """
        x - матрица размера (batch_size, m)
        errors - матрица ошибок (batch_size, N), N - число нейронов в слое
        learning_rate - константа скорости обучения
        """

        for i, v in enumerate(self.neurons):
            v.update_mini_batch(x, errors[:, [i]], learning_rate)

    def get_error(self, next_layer_errors, next_layer_weights):
        """
        Считает ошибку на данном слое сети
        next_layer_errors - ndarray размера (n, n_{l+1})
        weights - ndarray размера (n_{l+1}, n_l+1)
        :return: матрица ошибок (n, n_l)
        """

        sums = self.intermediate_sums
        sum_primes = np.array([v.derivative(sums[:, i])
                               for i, v in enumerate(self.neurons)])
        return (next_layer_weights[:, 1:].T.dot(next_layer_errors.T) * sum_primes).T

    def get_weights(self):
        """
        :return: Матрица весов для всего слоя, (n_l, n_{l-1})
        """
        return np.array([v.w.flatten() for i, v in enumerate(self.neurons)])

    @staticmethod
    def init_with_weights(weights, act_func, act_func_der):
        return Layer(
            [Neuron(weights_row.reshape(weights.shape[1], 1), act_func, act_func_der) for weights_row in weights])

    @staticmethod
    def layer_with_random_weights(neuron_count, weights_len, act_func, act_func_der):
        return Layer([Neuron(np.array([[random.uniform(-1.0, 1.0)] for j in range(weights_len + 1)]),
                             act_func, act_func_der) for i in range(neuron_count)])
