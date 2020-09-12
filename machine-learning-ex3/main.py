""" Driver for exercise 3 in Coursera's machine learning``
"""
import os
import numpy as np
from matplotlib import pyplot
from scipy.io import loadmat


def visualize(X, figsize=(10, 10)):
    m, n = X.shape
    examples, features = X.shape
    ex_witdth = int(np.round(np.sqrt(n)))
    ex_height = features / ex_witdth
    rows = int(np.floor(np.sqrt(m)))
    cols = int(np.ceil(m / rows))

    figure, ax_array = pyplot.subplots(rows, cols, figsize=figsize)
    figure.subplots_adjust(wspace=0.025, hspace=0.025)
    ax_array = ax_array.ravel()
    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(ex_witdth, ex_witdth, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')


def part_one():
    INPUT_LAYER_SIZE = 400
    OUTPUT_LAYER_SIZE = 10
    data = loadmat(os.path.join('./data/ex3data1.mat'))
    X, y = data['X'], data['y'].ravel()
    rand_indicies = np.random.choice(y.size, 100, replace=False)
    sel = X[rand_indicies, :]
    visualize(sel)


def main():
    """Driver function
    """
    part_one()


if __name__ == '__main__':
    main()