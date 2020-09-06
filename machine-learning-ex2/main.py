""" Programming exercise two for Coursera's machine learning course.

Run main.py to see output for Week 3's programming exercise #2
"""
import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from typing import List
from scipy import optimize

from sigmoid import sigmoid
from logistic_cost import cost_function


def print_section_header(section: str) -> None:
    """Prints a section header to STDOUT

    Args:
        section: Name of the section to print
    Returns:
        None
    """
    spacing = 50
    blank_space = ' ' * ((spacing - len(section)) // 2 - 1)
    print('='*spacing)
    print('{}{}'.format(blank_space, section))
    print('='*spacing)


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


def optimize_theta(cost_function, initial_theta, X, y):
    """Optimize theta parameters using a cost function and initial theta

    Args:
        cost_function: Cost function used to calculate error
        initial_theta: Starting values for theta parameters
        X: Input features
        y: Labels for training set

    Returns:

    """
    res = optimize.minimize(cost_function,
                            initial_theta,
                            (X, y),
                            jac=True,
                            method='TNC',
                            options={'maxiter': 400})

    return res


def part_one():
    """Drive function for part one of the exercise

    Visualize the data, compute cost and gradient and learn optimal theta
    paramaters

    Returns:
        None
    """
    print_section_header('Section 1')
    data = np.loadtxt(os.path.join('data/ex2data1.txt'), delimiter=',')
    X, y = data[:, 0:2], data[:, 2]
    visualize(X, y)
    pyplot.show()

    m = y.size
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    theta = np.array([-24, 0.2, 0.2])
    cost, gradient = cost_function(theta, X, y)
    print("Cost:\n\t{:.3f}".format(cost))
    print("Gradient:\n\t{:.3f}, {:.3f}, {:.3f}".format(*gradient))
    optimized = optimize_theta(cost_function, theta, X, y)
    optimized_cost = optimized.fun
    optimized_theta = optimized.x
    print('Optimized cost:\n\t{:.3f}'.format(optimized_cost))
    print('Optimized theta:\n\t{:.3f}, {:.3f}, {:.3f}'.
          format(*optimized_theta))


def main():
    """Main driver function.

    Runs the sections for programming exercise two

    Returns:
        None
    """
    part_one()


if __name__ == '__main__':
    main()
