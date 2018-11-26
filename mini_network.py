import sys
from mnist import MNIST
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


def transform_labels_to_vectors(labels):
    res = []
    for label in labels:
        res.append(get_vector_from_label(label))
    return np.array(res)


def get_vector_from_label(label):
    res = np.zeros(10)
    res[label] = 1
    return res


def get_example_batch(batch_size):
    mndata = MNIST('resources/train/mnist')

    images, labels = mndata.load_training()
    images = np.array(images).reshape(len(images), 1, 28, 28) / 255
    answers = transform_labels_to_vectors(labels)
    return images[:batch_size], answers[:batch_size]


def run_conv_net():
    conv_layer1 = ConvolutionLayer.init_with_weights_from_file('resources/weights_kernels0', 5, 2)
    conv_layer2 = ConvolutionLayer.init_with_weights_from_file('resources/weights_kernels1', 5, 2)
    # conv_layer1 = ConvolutionLayer.init_random_convolution_layer(10, 5, 2)
    # conv_layer2 = ConvolutionLayer.init_random_convolution_layer(20, 5, 2)
    layer1 = Layer.layer_with_random_weights(50, 320, sigmoid, sigmoid_prime)
    layer2 = Layer.layer_with_random_weights(10, 50, sigmoid, sigmoid_prime)

    # network_conv = Network([layer1, layer2], j_cross_entropy, j_cross_entropy_derivative)
    network_conv = Network(
        [Layer.init_with_weights(read_matrix_from_file('resources/weights_conv' + str(i)), sigmoid, sigmoid_prime)
         for i in range(2)], j_cross_entropy, j_cross_entropy_derivative)

    network = ConvolutionNetwork([conv_layer1, conv_layer2], network_conv)

    images, answers = get_example_batch(1000)
    network.sgd(images, answers, 100, 1, 1, 100, 1e-9, visualize=True)

    # for i, l in enumerate(network.fully_connected_net.layers):
    #     save_matrix_to_file('resources/weights_conv' + str(i), l.get_weights())
    #
    # for i, l in enumerate(network.layers):
    #     save_matrix_to_file('resources/weights_kernels' + str(i), map(lambda x: x.flatten(), l.get_weights()))

    for i, l in enumerate(network.fully_connected_net.layers):
        save_matrix_to_file('resources/weights_conv_t' + str(i), l.get_weights())

    for i, l in enumerate(network.layers):
        save_matrix_to_file('resources/weights_kernels_t' + str(i), map(lambda x: x.flatten(), l.get_weights()))

    print(network.process_input(images, answers))


def run_on_test_data():
    conv_layer1 = ConvolutionLayer.init_with_weights_from_file('resources/weights_kernels_t0', 5, 2)
    conv_layer2 = ConvolutionLayer.init_with_weights_from_file('resources/weights_kernels_t1', 5, 2)
    network_conv = Network(
        [Layer.init_with_weights(read_matrix_from_file('resources/weights_conv_t' + str(i)), sigmoid, sigmoid_prime)
         for i in range(2)], target_func_for_tests, j_cross_entropy_derivative)
    network = ConvolutionNetwork([conv_layer1, conv_layer2], network_conv)

    mndata = MNIST('resources/train/mnist')

    images, labels = mndata.load_testing()
    images = np.array(images).reshape(len(images), 1, 28, 28) / 255
    answers = transform_labels_to_vectors(labels)
    print("\nQuality is {:.2f}%\n".format(network.process_input(images[:100], answers[:100]) * 100 / 100))


def run_simple_net():
    network = Network(
        [Layer.init_with_weights(read_matrix_from_file('resources/weights' + str(i)), sigmoid, sigmoid_prime) for i in
         range(2)],
        j_cross_entropy,
        j_cross_entropy_derivative)

    # data = transform_input_imgs_to_data(read_char_images_from_dir('resources/test', False, 36, 27))
    data, answers = read_all_char_examples_with_answers('resources/train', False)

    data = transform_input_imgs_to_data(data)
    print(network.sgd(data, answers, 110, 10, 10, eps=1e-7, visualize=True))

    for i, l in enumerate(network.layers):
        save_matrix_to_file('resources/weights' + str(i), l.get_weights())

    print(network.get_result_matrix(data))

    print(network.process_input(data, answers))


if __name__ == '__main__':
    import time
    t = time.time()
    run_conv_net()
    # run_simple_net()
    run_on_test_data()
    print(time.time() - t)
    sys.exit(0)
