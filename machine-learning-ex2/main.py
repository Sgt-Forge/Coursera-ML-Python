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
from regularized_cost_function import regularized_cost_function


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
    pyplot.plot(X[pos, 0], X[pos, 1], 'k+', lw=2, ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
    pyplot.xlabel('Exam Score 1')
    pyplot.ylabel('Exam Score 2')
    pyplot.legend(['Admitted', 'Rejected'])


def optimize_theta(cost_function, initial_theta, X, y,
                   options={'maxiter': 400}):
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
                            options=options)

    return res


def plot_decision_boundary(theta, X, y):
    """Plot data and draw decision boundary with given theta parameters.

    Generates scatter plot with decision boundary.

    Args:
        visualize: Plotting function to create a scatter plot
        theta: Theta parameters for the decision boundary
        X: A matrix of scores for exams 1 and 2 for each student
        y: Binary vector to track admittance for each student

    Returns:
        None
    """
    visualize(X[:, 1:3], y)
    '''
    If you want to figure out _how_ to plot the decision boundary, you have to
    understand the following links:
    https://statinfer.com/203-5-2-decision-boundary-logistic-regression/
    https://en.wikipedia.org/wiki/Logistic_regression
    Basically we have to plot the line when our probability is 0.5. You can
    recover the theta paremeters from the equation by calculating the odds of
    classifying 0.5 (yes, the literal definition of odds: {p / 1-p} )
    '''
    X_points = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    y_points = (-1 / theta[2]) * (theta[1] * X_points + theta[0])
    pyplot.plot(X_points, y_points)


def predict(theta, X):
    """Make predictions for test set with trained theta parameters

    Args:
        theta: Trained theta parameters
        X: Test set

    Returns:
        array-like of predictions
    """
    predictions = sigmoid(X.dot(theta)) >= 0.5

    return predictions


def map_features(X1, X2):
    """Maps two features to a 6 degree polynomial feature set

    Args:
        X: initial feature set without bias feature

    Returns:
        Mapped feature set with added bias feature
    """
    degree = 6
    if X1.ndim > 0:
        mapped_features = [np.ones(X1.shape[0])]
    else:
        mapped_features = [(np.ones(1))]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            mapped_features.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(mapped_features, axis=1)
    else:
        return np.array(mapped_features, dtype=object)


def plot_non_linear_boundary(theta, X, y):
    visualize(X, y)
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((u.size, v.size))
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            z[i, j] = np.dot(map_features(ui, vj), theta)

    z = z.T

    pyplot.contour(u, v, z, levels=[0], linewidths=2, colors='g')
    pyplot.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens',
                    alpha=0.4)


def part_one():
    """Driver function for part one of the exercise

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
    plot_decision_boundary(optimized_theta, X, y)
    pyplot.show()
    test_scores = np.array([1, 45, 85])
    probability = sigmoid(test_scores.dot(optimized_theta))
    print('Probability for student with scores 45 and 85:\n\t{:.3f}'.
          format(probability))
    print('Expected value: 0.775 +/- 0.002')
    predictions = predict(optimized_theta, X)
    print('Training accuracy:\n\t{:.3f}'.
          format(np.mean(predictions == y) * 100))
    print('Expected accuracy: 89.00%')


def part_two():
    """Driver function for part two of the exercise

    Visualize the data, compute regularized cost and gradient, and learn
    optimal theta parameters

    Returns:
        None
    """
    print_section_header('Section 2')
    data = np.loadtxt(os.path.join('data/ex2data2.txt'), delimiter=',')
    X, y = data[:, 0:2], data[:, 2]
    visualize(X, y)
    pyplot.show()
    X_mapped = map_features(X[:, 0], X[:, 1])
    m = y.size
    theta = np.zeros(X_mapped.shape[1])
    cost, gradient = regularized_cost_function(theta, X_mapped, y, 1)
    print("Cost:\n\t{:.3f}".format(cost))
    print('Gradient:\n\t{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.
          format(*gradient))
    theta = np.ones(X_mapped.shape[1])
    cost, gradient = regularized_cost_function(theta, X_mapped, y, 10)
    print('Set initial thetas to 1, and lambda to 10')
    print("Cost:\n\t{:.3f}".format(cost))
    print('Gradient:\n\t{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.
          format(*gradient))
    optimized = optimize_theta(cost_function, theta, X_mapped, y,
                               options={'maxiter': 100})
    optimized_cost = optimized.fun
    optimized_theta = optimized.x
    plot_non_linear_boundary(optimized_theta, X, y)
    pyplot.show()


def main():
    """Main driver function.

    Runs the sections for programming exercise two

    Returns:
        None
    """
    part_one()
    part_two()


if __name__ == '__main__':
    main()
