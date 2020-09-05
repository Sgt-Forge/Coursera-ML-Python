""" Programming exercise two for Coursera's machine learning course.

Run main.py to see output for Week 3's programming exercise #2
"""
import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from typing import List

from sigmoid import sigmoid


def visualize(X: List[List[float]], y: List[int]) -> None:
    """Plot data.

    Generates scatter plot

    Args:
        X: A matrix of scores for exams 1 and 2 for each student
        y: Binary vector to track admittance for each student

    Returns:
        None
    """
    pos = y == 1
    neg = y == 0
    _fig = pyplot.figure()
    pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
    pyplot.xlabel('Exam Score 1')
    pyplot.ylabel('Exam Score 2')
    pyplot.legend(['Admitted', 'Rejected'])


def main():
    """Main driver function.

    Runs the sections for programming exercise two

    Returns:
        None
    """
    data = np.loadtxt(os.path.join('data/ex2data1.txt'), delimiter=',')
    X, y = data[:, 0:2], data[:, 2]
    visualize(X, y)
    pyplot.show()

    theta = np.array([1, 2])
    cross = np.dot(X, theta)
    print(cross)


if __name__ == '__main__':
    main()
