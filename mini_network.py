import sys

from classes.convolution.convolution_layer import ConvolutionLayer
from classes.convolution.convolutional_network import ConvolutionNetwork
from classes.layer import Layer
from classes.network import Network
from classes.neuron import *
from utils.file_utils import *
from utils.target_functions import *


def transform_input_imgs_to_data(imgs):
    data = np.array(list((map(lambda arr: arr.flatten(), imgs)))) / 255
    return abs(1 - data)


def run_conv_net():
    conv_layer1 = ConvolutionLayer.init_with_weights_from_file('resources/weights_kernels0', 5, 2)
    conv_layer2 = ConvolutionLayer.init_with_weights_from_file('resources/weights_kernels1', 5, 2)
    conv_layer3 = ConvolutionLayer.init_with_weights_from_file('resources/weights_kernels2', 3, 2)
    # conv_layer1 = ConvolutionLayer.init_random_convolution_layer(8, 5, 2)
    # conv_layer2 = ConvolutionLayer.init_random_convolution_layer(16, 5, 2)
    # conv_layer3 = ConvolutionLayer.init_random_convolution_layer(32, 3, 2)
    layer1 = Layer.layer_with_random_weights(100, 768, sigmoid, sigmoid_prime)
    layer2 = Layer.layer_with_random_weights(10, 100, sigmoid, sigmoid_prime)

    # network_conv = Network([layer1, layer2], j_cross_entropy, j_cross_entropy_derivative)
    network_conv = Network(
        [Layer.init_with_weights(read_matrix_from_file('resources/weights_conv' + str(i)), sigmoid, sigmoid_prime)
         for i in range(2)], j_cross_entropy, j_cross_entropy_derivative)

    network = ConvolutionNetwork([conv_layer1, conv_layer2, conv_layer3], network_conv)
    exs, ans = read_all_char_examples_with_answers('resources/digits', False, 64, 48)
    exs = abs(1 - exs / 255)

    print(network.sgd(exs, ans, 110, 1, 0.02, 10, 1e-9, visualize=False))

    # for i, l in enumerate(network.fully_connected_net.layers):
    #     save_matrix_to_file('resources/weights_conv' + str(i), l.get_weights())
    #
    # for i, l in enumerate(network.layers):
    #     save_matrix_to_file('resources/weights_kernels' + str(i), map(lambda x: x.flatten(), l.get_weights()))

    for i, l in enumerate(network.fully_connected_net.layers):
        save_matrix_to_file('resources/weights_conv_t' + str(i), l.get_weights())

    for i, l in enumerate(network.layers):
        save_matrix_to_file('resources/weights_kernels_t' + str(i), map(lambda x: x.flatten(), l.get_weights()))

    print(network.process_input(exs, ans))


def run_simple_net():
    network = Network(
        [Layer.init_with_weights(read_matrix_from_file('resources/weights' + str(i)), sigmoid, sigmoid_prime) for i in
         range(2)],
        j_cross_entropy,
        j_cross_entropy_derivative)

    # data = transform_input_imgs_to_data(read_char_images_from_dir('resources/test', False, 36, 27))
    data, answers = read_all_char_examples_with_answers('resources/digits', False)

    data = transform_input_imgs_to_data(data)
    print(network.sgd(data, answers, 110, 10, 10, eps=1e-7, visualize=True))

    for i, l in enumerate(network.layers):
        save_matrix_to_file('resources/weights' + str(i), l.get_weights())

    print(network.get_result_matrix(data))

    print(network.process_input(data, answers))


if __name__ == '__main__':
    run_conv_net()
    # run_simple_net()
    sys.exit(0)
