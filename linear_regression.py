import numpy as np
import random2 as rd

def get_slope_intercept(X, y):
    """
    Calculate the slope and the intercept of the linear regression:
    y = mx + b,  m is the slope and b is the intercept.
    
    """
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    m_numerator = X_mean * y_mean - np.mean(X*y)
    m_demoninator = X_mean ** 2 - np.mean(X**2)
    
    m = m_numerator / m_demoninator

    b = y_mean - m * X_mean
    
    return m, b


def generate_linear_data(x_range: tuple, m=1, b=1, variance=3):
    
    X = np.arange(x_range[0], x_range[1] + 1)
    
    y = np.array([(m*x + b) + rd.randint(-variance, variance) for x in X])
    
    return X, y
    




