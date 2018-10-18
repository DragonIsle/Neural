import numpy as np
import random


def read_inputs(file_name):
    f = open(file_name)

    data = np.zeros((1, 2))
    answers = np.zeros((1, 1))
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        x_i = np.array([[float(line.split(' ')[0]),  float(line.split(' ')[1])]])
        y_i = np.array([[float(line.split(' ')[2])]])
        data = np.append(data, x_i, axis=0)
        answers = np.append(answers, y_i, axis=0)
    f.close()
    return data[1:, :], answers[1:, :]


def read_weights_from_file(file_name, layer_number):
    f = open(file_name)
    weights = []
    while True:
        line = f.readline()[:-1]
        if len(line) == 0:
            break
        weights_line = line.split(' ')
        weights.append([float(v) for v in weights_line])
    f.close()
    return np.array(weights)


def append_weights_to_file(file_name, neuron_count, neuron_weights_count, range_from, range_to):
    f = open(file_name, 'a')
    for i in range(neuron_count):
        res_str = ''
        for j in range(neuron_weights_count):
            res_str += str(round(random.uniform(range_from, range_to), 4)) + ' '
        f.write(res_str[:-1] + '\n')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    print(read_weights_from_file('../resources/weights', 1))
