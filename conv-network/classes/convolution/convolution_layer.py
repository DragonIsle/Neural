import numpy as np

from utils.file_utils import read_matrix_from_file
from .convolution_neuron import ConvolutionNeuron


class ConvolutionLayer:
    """
    Класс, представляющий слой сверточной сети(сверки или подвыборки в зависимости от нейронов)
    neurons = ядра свертки слоя
    """

    def __init__(self, neurons):
        self.neurons = neurons
        self.saved_maps = []

    def process_sign_maps(self, sign_maps_arr):
        """
        Применяет каждый нейрон слоя к входному массиву карт признаков

        :param sign_maps_arr: Входной массив карт признаков для каждого примера
        :return: Массив с новыми картами признаков для каждого примера
        """
        self.clear_convolution_results()
        self.saved_maps = np.array([self.process_maps_one_example(maps) for maps in sign_maps_arr])
        return self.saved_maps

    def process_maps_one_example(self, maps):
        res = []
        neurons_for_map = int(len(self.neurons) / len(maps))
        for i in range(len(maps)):
            start = i * neurons_for_map
            for neuron in self.neurons[start:start + neurons_for_map]:
                res.append(neuron.process_sign_map(maps[i]))
        return np.array(res)

    def get_errors(self, next_layer_errs, next_layer_weights):
        res = []
        for ex_id in range(len(next_layer_errs)):
            res.append(self.create_error_maps(next_layer_errs[ex_id], next_layer_weights, ex_id))
        return np.array(res)

    def create_error_maps(self, next_layer_errs, next_layer_weights, ex_id):
        return np.array([neuron.create_err_map(next_layer_errs[i], next_layer_weights[i], ex_id)
                         for i, neuron in enumerate(self.neurons)])

    def update_mini_batch(self, layer_input, errors, learning_rate):
        input_id = 0
        neurons_for_map = int(len(self.neurons) / layer_input.shape[1])

        for i, v in enumerate(self.neurons):
            v.update_mini_batch(layer_input[:, input_id], errors[:, i], learning_rate)
            if i == input_id * neurons_for_map + neurons_for_map:
                input_id += 1

    def get_weights(self):
        return np.array([neuron.kernel for neuron in self.neurons])

    def clear_convolution_results(self):
        for neuron in self.neurons:
            neuron.convolution_res = []

    @staticmethod
    def init_random_convolution_layer(neuron_count, kernel_size, subsampling_size=2):
        return ConvolutionLayer(np.array([ConvolutionNeuron(np.random.rand(kernel_size, kernel_size) - 0.5,
                                                            subsampling_size) for i in range(neuron_count)]))

    @staticmethod
    def init_with_weights_from_file(file_name, kernel_sie, subsampling_size):
        matrix = read_matrix_from_file(file_name)
        return ConvolutionLayer(np.array([ConvolutionNeuron(v.reshape(kernel_sie, kernel_sie), subsampling_size)
                                          for i, v in enumerate(matrix)]))
