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

def compute_cost(X, y, theta):
    m = y.shape[0]
    J = (1/(2*m)) * np.sum( np.square((np.dot(X, theta) - y)) )
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    adjusted_theta = theta.copy()

    for i in range(0, num_iters):
        adjusted_theta -= (alpha / m) * (np.dot(X, adjusted_theta) - y).dot(X)

    return adjusted_theta

def predict(X, theta):
    pred = np.dot(X, theta)
    print("For population =", X[1]*10000, ' we predict a profit of {:.2f}\n'.format(pred*10000))

def visualize_cost(X, y, trained_theta):
    # Step over theta0 values in range -10,10 with 100 steps
    # theta0_vals => (100 x 1)
    theta0_vals = np.linspace(-10,10,100)
    # Step over theta1 values in range -4 4 with 100 steps
    # thteta1_vals => (100 x 1) 
    theta1_vals = np.linspace(-1,4,100)
    # Create a matrix of costs at different values of theta0 and theta1
    # J_vals => (100 x 100)
    J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))
    
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            J_vals[i, j] = compute_cost(X, y, [theta0, theta1])
    J_vals = J_vals.T

    figure = pyplot.figure(figsize=(12, 5))
    # First parameter controls position in sub plot
    # projection controls 3d-ness
    axis = figure.add_subplot(121, projection='3d')
    # cmap='viridis' makes it colorful
    axis.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
    pyplot.xlabel('theta0')
    pyplot.ylabel('theta1')
    axis.set_zlabel('Cost J')
    pyplot.title('Cost at different thetas')

    axis = pyplot.subplot(122)
    # Levels controls number and positions of the contour lines, should be int or array-like object
    pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
    axis.set_xlabel('theta0')
    axis.set_ylabel('theta1')
    pyplot.plot(trained_theta[0], trained_theta[1], 'ro', ms=10, lw=2)
    pyplot.title('Minimum value of cost J')

def normalizeFeatures(X):
    # X => (m x n) : m = num. examples, n = num. features
    X_norm = X.copy()
    m = X_norm.shape[0]
    # X.shape => (m x n)
    # np.zeros(X.shape[1]) => np.zeros(n) => (n x 1)
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    for feature in range(X.shape[1]):
        mu[feature] = np.mean(X[:, feature])
        sigma[feature] = np.std(X[:, feature])
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

def part1():
    print('='*50)
    print('\tBegin Part 1')
    print('='*50)
    data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m = y.size

    plot_data(X, y)
    pyplot.show()

    X = np.stack([np.ones(m), X], axis=1)
    iterations = 1500
    alpha = 0.01
    theta = np.array([0.0, 0.0])
    print(compute_cost(X, y, theta))
    print(compute_cost(X, y, np.array([-1, 2])))
    theta = gradient_descent(X, y, theta, alpha, iterations)

    plot_data(X[:, 1], y)
    pyplot.plot(X[:, 1], np.dot(X, theta), '-')
    pyplot.legend(['Training data', 'Linear regression'])
    pyplot.show()
    predict([1, 3.5], theta)
    predict([1, 7], theta)

    visualize_cost(X, y, theta)
    pyplot.show()

def part3():
    print('='*50)
    print('\tBegin Part 3 (Optional)')
    print('='*50)
    data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
    # Grabs all rows and all columns from [start, 2) exclusive of index 2
    X = data[:, :2]
    # Grabs all rows and column at index 2
    y = data[:, 2]
    m = y.shape[0]
    print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:,1]', 'y'))
    for i in range(10):
        print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))

    X_norm, mu, sigma = normalizeFeatures(X)
    print('Computed mean:\t', mu)
    print('Computed STD:\t', sigma)
    # Concatenate a vector of 1s at the beginning of X_norm
    # Axis controls whether you concatenate by row or col, row is 0, col is 1
    X = np.concatenate([np.ones((m,1)), X_norm], axis=1)
    theta = np.array([0.0, 0.0, 0.0])
    cost = compute_cost(X, y, theta)
    print(cost)
    iterations = 400
    alpha = 0.1
    theta = gradient_descent(X, y, theta, alpha, iterations)
    print(theta)
    cost = compute_cost(X, y, theta)
    test_case = [1, 1650, 3]
    test_case[1:3] = (test_case[1:3] - mu) / sigma
    cost = np.dot(test_case,theta)
    print(cost)

def main():
    # part1()
    part3()



if __name__ == '__main__':
    main()