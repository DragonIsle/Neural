from utils.target_functions import *
from utils.visualization_utils import print_graph


class ConvolutionNetwork:
    def __init__(self, layers, fully_connected_net):
        self.layers = layers
        self.fully_connected_net = fully_connected_net

    def get_result_matrix(self, input_data):
        next_layer_input = input_data
        for layer in self.layers:
            next_layer_input = layer.process_sign_maps(next_layer_input)

        return self.fully_connected_net.get_result_matrix(self.transform_maps(next_layer_input))

    def process_input(self, input_data, y):
        next_layer_input = input_data
        for layer in self.layers:
            next_layer_input = layer.process_sign_maps(next_layer_input)

        return self.fully_connected_net.process_input(self.transform_maps(next_layer_input), y)

    def sgd(self, x, y, batch_size, learning_rate, learning_rate_conn, step_limit, eps=1e-6, visualize=False):
        errors = 0
        step = 0
        steps = []
        target_func_results = []
        while (not errors) and (step < step_limit):
            batch_ids_arr = self.fully_connected_net.get_batches(np.arange(len(y)), batch_size)
            # init_target_func = self.process_input(x, y)
            for batch_ids in batch_ids_arr:
                x_b = x[batch_ids]
                y_b = y[batch_ids]
                self.update_mini_batch(x_b, y_b, learning_rate, learning_rate_conn)
                step += 1
                if visualize:
                    steps.append(step)
                    target_func_results.append(self.process_input(x, y))
                if step == step_limit:
                    break
            # res_target_func = self.process_input(x, y)
            # errors += int(abs(init_target_func - res_target_func) < eps)
        if visualize:
            print_graph(steps, target_func_results)
        return errors

    def update_mini_batch(self, x, y, learning_rate, learning_rate_conn):
        self.process_input(x, y)
        transformed = self.transform_maps(self.layers[-1].saved_maps)
        errors = self.fully_connected_net.update_mini_batch(transformed, y, learning_rate_conn)
        last_conv_layer_err = self.get_last_convolution_layer_err \
            (errors, self.fully_connected_net.layers[0].get_weights())

        layer_err = last_conv_layer_err
        last_layer_id = len(self.layers) - 1
        reversed_layers = self.layers[::-1]
        layer_weights = self.layers[-1].get_weights()

        for i, v in enumerate(reversed_layers):
            if i < last_layer_id:
                layer_input = reversed_layers[i + 1].saved_maps
            else:
                layer_input = x
            v.update_mini_batch(layer_input, layer_err, learning_rate)
            if i < last_layer_id:
                layer_err = reversed_layers[i + 1].get_errors(layer_err, layer_weights)
                layer_weights = reversed_layers[i + 1].get_weights()

        return layer_err

    def get_last_convolution_layer_err(self, next_layer_errors, next_layer_weights):
        last_layer = self.layers[-1]
        last_layer_map_size = last_layer.saved_maps[0].shape[1:]

        err_sub_lauer = (next_layer_weights[:, 1:].T.dot(next_layer_errors.T)).T
        err_maps_sub_layer = ConvolutionNetwork.transform_err_arrays_to_maps \
            (err_sub_lauer, last_layer_map_size[0], last_layer_map_size[1])

        final_result = []
        for ex_id in range(len(next_layer_errors)):
            ex_result = []
            for i, neuron in enumerate(last_layer.neurons):
                ex_result.append(neuron.create_err_map_for_conv(err_maps_sub_layer[ex_id][i], ex_id))
            final_result.append(np.array(ex_result))
        return np.array(final_result)

    @staticmethod
    def transform_maps(maps_arr):
        return np.array([maps.flatten() for maps in maps_arr])

    @staticmethod
    def transform_err_arrays_to_maps(err_arrays, map_y, map_x):
        return np.array([ConvolutionNetwork.transform_err_array_to_maps(err_arr, map_y, map_x)
                         for err_arr in err_arrays])

    @staticmethod
    def transform_err_array_to_maps(err_arr, map_y, map_x):
        elems_count = map_y * map_x
        return np.array([err_arr[i:i + elems_count].reshape(map_y, map_x)
                         for i in range(int(len(err_arr) / elems_count))])
