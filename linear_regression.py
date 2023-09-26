import numpy as np

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)



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
    
    return m



