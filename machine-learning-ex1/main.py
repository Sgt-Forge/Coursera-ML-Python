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

def main():
    data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m = y.size

    plot_data(X, y)


if __name__ == '__main__':
    main()