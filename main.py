import matplotlib.pyplot as plt
import numpy as np
import linear_regression as lr

x_range = (-20, 30)
slope = -4
intercept = 50
variance = 5

X, y = lr.generate_linear_data(x_range, slope=slope, intercept=intercept, variance=variance)


for index in range(1, len(X) + 1):
    
    partial_y = y[:index]
    partial_X = X[:index]
    
    m, b = lr.get_slope_intercept(partial_X, partial_y)
    # print(m, b)
    # print(partial_X, partial_y)
    y_line = [m*x + b for x in X]
    mae = lr.mean_absolute_error(y, y_line)
    
    plt.xlim((x_range[0] - 10, x_range[1] + 10))
    plt.ylim((np.min(y) -25, np.max(y) + 10))

    plt.scatter(X[index:], y[index:])
    plt.scatter(X[:index], partial_y)
    plt.plot(X, y_line, color='orange')
    plt.annotate(f"mae = {mae:.2f} ", (x_range[0] -9, np.min(y) - 5))
    
    input('Pressione Enter para continuar')

    
    plt.show()


