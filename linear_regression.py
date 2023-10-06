import numpy as np
import random2 as rd


def get_slope_intercept(X, y):
    """
    Calculate the slope and the intercept of the linear regression:
    y = mx + b,  m is the slope and b is the intercept.
    
    """
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    slope_numerator = X_mean * y_mean - np.mean(X * y)
    slope_demoninator = X_mean ** 2 - np.mean(X ** 2)

    slope = slope_numerator / slope_demoninator

    intercept = y_mean - slope * X_mean

    return slope, intercept


def generate_linear_data(x_range: tuple, slope=1, intercept=1, variance=3):
    """
    Generate data with linear shape, according to the formula of linear regression: y = mx + b.
    In addition, the y is summed with a variance to simulate a real world relation.
    """

    X = np.arange(x_range[0], x_range[1] + 1)

    y = np.array([(slope * x + intercept) + rd.randint(-variance, variance) for x in X])

    return X, y


def add_outlier(y, position=5, variance=50):
    """
    add an outlier to the data set in a given position and variance.
    """
    y[position] = y[position] + variance

    return y


def mean_absolute_error(y, y_predict):
    return sum(np.abs(y - y_predict)) / len(y) if y is not None else None
