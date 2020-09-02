import os
import numpy as np 
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from time import sleep

def warm_up_exercise():
    A = np.eye(5)

    return A

def plot_data(x, y):
    figure = pyplot.figure()
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Profit in $10,000s')
    pyplot.xlabel('Population of City in 10,000s')
    pyplot.show()

def computeCost(X, y, theta):
    m = y.size
    J = (1/(2*m)) * sum( (np.matmul(X, theta) - y)**2 )
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    adjusted_theta = theta

    for i in range(0, num_iters):
        hypothesis = np.matmul(X, adjusted_theta)
        error = hypothesis - y
        gradient = np.matmul(error, X)
        adjusted_theta -= (alpha / m) * gradient

    return adjusted_theta

def main():
    data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m = y.size

    plot_data(X, y)

    X = np.stack([np.ones(m), X], axis=1)
    iterations = 1500
    alpha = 0.01
    theta = np.array([0.0, 0.0])
    print(computeCost(X, y, theta))
    print(computeCost(X, y, np.array([-1, 2])))
    print(gradientDescent(X, y, np.array([0.0, 0.0]), alpha, iterations))


if __name__ == '__main__':
    main()