import numpy as np
from utils.visualization_utils import print_graph


class Network:

    def __init__(self, layers, target_function, target_function_derivative):
        self.layers = layers
        self.target_function = target_function
        self.target_function_derivative = target_function_derivative

    def process_input(self, x, y):
        """
        Выполняет один проход по сети для матрицы входных данных x.

        :param x: Вектор входов, (n, m),
        n - количество примеров
        m - количество переменных(равно числу нейронов в 1-м слое сети)
        :param y: Матрица правильных ответов, (n, N),
        N - Число нейронов в выходном слое
        :return: Значение целевой функции сети
        """

        return self.target_function(self.get_result_matrix(x), y)

    def get_result_matrix(self, x):
        """
        Возвращает матрицу активаций выходного слоя
        :param x: Матрица входов, (n, m)
        :return: Матрица выходных активаций, (n, N)
        """

        next_layer_input = np.append(np.ones((len(x), 1), dtype=int), x, axis=1)
        for layer in self.layers:
            activations = layer.process_input(next_layer_input)
            next_layer_input = np.append(np.ones((len(activations), 1), dtype=int), activations, axis=1)
        return next_layer_input[:, 1:]

    def sgd(self, x, y, batch_size, learning_rate, step_limit, eps=1e-6, visualize=False):
        """

        :param x: Матрица входов - (n, m), n - число примеров, m - число переменных
        :param y: Матрица правильных ответов - (n, N), n - число примеров,
        N - число нейронов выходного слоя
        :param batch_size: Размер батча, выбираемого для обучения
        :param learning_rate: Константа скорости обучения
        :param step_limit: Максимальное число шагов, которое может сделать алгоритм
        :param eps: Точность алгоритма
        :param visualize: Визуализировать ли изменение целевой функции
        :return: 1, если успешно сошлось, 0 если достигнут предел количества допустимых шагов
        """

        errors = 0
        step = 0
        steps = []
        target_func_results = []
        while (not errors) and (step < step_limit):
            batch_ids_arr = Network.get_batches(np.arange(len(y)), batch_size)
            init_target_func = self.process_input(x, y)
            for batch_ids in batch_ids_arr:
                x_b = x[batch_ids]
                y_b = y[batch_ids]
                self.update_mini_batch(x_b, y_b, learning_rate)
                step += 1
                if visualize:
                    steps.append(step)
                    target_func_results.append(self.process_input(x, y))
            res_target_func = self.process_input(x, y)
            errors += int(abs(init_target_func - res_target_func) < eps)
        if visualize:
            print_graph(steps, target_func_results)
        return errors

    def update_mini_batch(self, x, y, learning_rate):
        """
        Обновляет все веса сети для одного батча
        :param x: Матрица входов - (batch_size, m), batch_size - число примеров, m - число переменных
        :param y: Матрица правильных ответов - (batch_size, N), batch_size - число примеров,
        N - число нейронов выходного слоя
        :param learning_rate: Константа скорости обучения
        """
        result_matrix = self.get_result_matrix(x)
        last_layer = self.layers[-1]
        da_dz = last_layer.get_acts_by_sums_derivative()

        layer_error = self.target_function_derivative(result_matrix, y) * da_dz
        layer_weights = last_layer.get_weights()

        last_layer_id = len(self.layers) - 1
        reversed_layers = self.layers[::-1]
        for i, v in enumerate(reversed_layers):
            if i < last_layer_id:
                layer_input = reversed_layers[i + 1].intermediate_activations
            else:
                layer_input = x
            layer_input = np.append(np.ones((layer_input.shape[0], 1), dtype=int), layer_input, axis=1)

            v.update_mini_batch(layer_input, layer_error, learning_rate)
            if i < last_layer_id:
                layer_error = reversed_layers[i + 1].get_error(layer_error, layer_weights)
                layer_weights = reversed_layers[i + 1].get_weights()

    @staticmethod
    def get_batches(x, batch_size):
        """
        Делит массив x на батчи размера n
        :param x: Массив для деления
        :param batch_size: Размер батча
        :return: Массив батчей
        """
        np.random.shuffle(x)
        return np.array([x[i:i + batch_size] for i in range(0, len(x), batch_size)])

