import os
import random

import cv2
import numpy as np


def read_matrix_from_file(file_name):
    f = open(file_name)
    matrix = []
    while True:
        line = f.readline()[:-1]
        if len(line) == 0:
            break
        matrix_row = line.split(' ')
        matrix.append([float(v) for v in matrix_row])
    f.close()
    return np.array(matrix)


def init_random_weights_in_file(file_name, neuron_count, neuron_weights_count, range_from, range_to):
    f = open(file_name, 'a')
    for i in range(neuron_count):
        res_str = ''
        for j in range(neuron_weights_count):
            res_str += str(round(random.uniform(range_from, range_to), 4)) + ' '
        f.write(res_str[:-1] + '\n')
    f.write('\n')
    f.close()


def save_matrix_to_file(file_name, matrix, mode='w'):
    f = open(file_name, mode)
    for row in matrix:
        f.write(np.array2string(row, max_line_width=100000, formatter={'float_kind': lambda x: "%.8f" % x})[1:-1] + '\n')
    f.write('\n')
    f.close()


def read_char_images_from_dir(directory_path, colorized, x, y):
    res_arr = []
    for filename in sorted(os.listdir(directory_path)):
        f_path = directory_path + '/' + filename
        if colorized:
            res_arr.append([cv2.resize(cv2.imread(f_path, cv2.IMREAD_COLOR), (x, y))])
        else:
            res_arr.append([cv2.resize(cv2.imread(f_path, cv2.IMREAD_GRAYSCALE), (x, y))])
    return np.array(res_arr)


def read_all_char_examples_with_answers(directory_path, colorized, x=36, y=27):
    dirs = os.listdir(directory_path)
    diag = np.zeros((len(dirs), len(dirs)), int)
    np.fill_diagonal(diag, 1)
    examples = []
    answers = []
    for i, directory in enumerate(sorted(dirs)):
        one_char_exs = read_char_images_from_dir \
            (directory_path + '/' + directory, colorized, x, y)
        examples.extend(one_char_exs)
        for _ in range(len(one_char_exs)):
            answers.append(diag[i])
    return np.array(examples), np.array(answers)


if __name__ == '__main__':
    exs, ans = read_all_char_examples_with_answers('../resources/letters', False)
    save_matrix_to_file('../resources/weights', np.array([[10, 12], [13, 14]]))
