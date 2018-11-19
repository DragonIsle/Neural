import skimage.measure as measure
import numpy as np
import scipy.signal as signal


class ConvolutionNeuron:
    """
    Нейрон, сворачивающий карту с помощью ядра.
    kernel - ядро для свертки.
    """

    def __init__(self, kernel, subsample_size=2):

        assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1, "Kernel must be squared and odd-sized!"

        self.kernel = kernel
        self.subsample_size = subsample_size
        self.convolution_res = []

    def process_sign_maps(self, sign_maps):
        """
        Сворачивает входной массив карт признаков ядром и возвращает новый массив.
        :param sign_maps: массив карт для свертки, одна карта разммера (n, m)
        :return: массив новых карт, полученный в процессе свертки,
         одна карта размера (n - kernel.x + 1, m - kernel.y + 1)
        """
        self.convolution_res = []
        return np.array([self.process_sign_map(sign_map) for sign_map in sign_maps])

    def process_sign_map(self, sign_map):
        """
        Сворачивает входную карту признаков ядром и возвращает новую карту.
        :param sign_map: карта для свертки, (n, m)
        :return: новая карта, полученный в процессе свертки, (n - kernel.x + 1, m - kernel.y + 1)
        """
        res_map = signal.convolve2d(sign_map, self.kernel, 'valid')
        self.convolution_res.append(res_map)
        return measure.block_reduce(res_map, (2, 2), np.max)

    def update_mini_batch(self, neuron_input, error, learning_rate):
        grad = np.zeros_like(self.kernel)
        for i in range(neuron_input.shape[0]):
            grad += self.get_grad_for_one_example(neuron_input[i], error[i])
        self.kernel -= grad * learning_rate

    def get_grad_for_one_example(self, neuron_input, error):
        grad = np.zeros_like(self.kernel)
        kernel_y = self.kernel.shape[0]
        kernel_x = self.kernel.shape[1]
        for y in range(error.shape[0]):
            for x in range(error.shape[1]):
                grad += neuron_input[y:y + kernel_y, x:x + kernel_x] * error[y, x]
        return grad / (error.shape[0] * error.shape[1])

    def create_err_map(self, next_layer_err, next_layer_weights, ex_id):
        err_on_subs = self.create_err_map_for_subs(next_layer_err, next_layer_weights)
        err_on_convs = self.create_err_map_for_conv(err_on_subs, ex_id)
        return err_on_convs

    @staticmethod
    def create_err_map_for_subs(next_layer_err, next_layer_weights):
        reversed_kernel = np.fliplr(np.flipud(next_layer_weights))
        return signal.convolve2d(next_layer_err, reversed_kernel, 'full')

    def create_err_map_for_conv(self, sub_layer_err, ex_id):
        input_map = np.array(self.convolution_res)[ex_id]
        max_values_primes_iter = iter(sub_layer_err.flatten())
        output = np.zeros_like(input_map)
        for y in range(sub_layer_err.shape[0]):
            for x in range(sub_layer_err.shape[1]):
                start_y = y * self.subsample_size
                start_x = x * self.subsample_size
                wind = input_map[start_y:start_y + self.subsample_size, start_x:start_x + self.subsample_size]
                (a, b) = np.unravel_index(np.argmax(wind, axis=None), wind.shape)
                oid_y = min(y * self.subsample_size + a, input_map.shape[0] - 1)
                oid_x = min(x * self.subsample_size + b, input_map.shape[1] - 1)
                output[oid_y, oid_x] = next(max_values_primes_iter)
        return output
