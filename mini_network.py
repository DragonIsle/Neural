import sys

from classes.neuron import *
from utils.target_functions import *
from classes.layer import Layer
from classes.network import Network
from utils.file_utils import *

if __name__ == '__main__':

    layer1 = Layer.layer_with_random_weights(2, 2, quadratic, quadratic_der)

    layer2 = Layer.layer_with_random_weights(1, 2, linear, linear_derivative)

    network = Network([layer1, layer2], j_quadratic, j_quadratic_derivative)

    data, answers = read_inputs('elliptic_paraboloid')
    print(network.sgd(data, answers, 20, 0.0002, 1000, eps=1e-8, visualize=True))

    print(network.get_result_matrix(data))

    for layer in network.layers:
        print(layer.get_weights())

    print(network.process_input(data, answers))

    data, answers = ([[5, 6]], [[9.5]])

    print(network.process_input(data, answers))

    sys.exit(0)
