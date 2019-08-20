import os

import matplotlib.pyplot as plt


def save(name='', fmt='png'):
    pwd = os.getcwd()
    path = '../resources/pictures/{}'.format(fmt)
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)
    plt.savefig('{}.{}'.format(name, fmt), fmt=fmt)
    os.chdir(pwd)
    plt.close()


def print_graph(x, y):
    fig = plt.figure()

    plt.plot(x, y)
    plt.ylabel("Целевая функция")
    plt.xlabel("Число шагов алгоритма")
    plt.ylim(0, 0.04)
    plt.show()
