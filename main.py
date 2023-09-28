import matplotlib.pyplot as plt
import numpy as np
import linear_regression as lr

x_range = (-20, 30)
slope = -3
intercept = 50
variance = 10

X, y = lr.generate_linear_data(x_range, slope, intercept, variance)


for index in range(1, len(X) + 1):
    
    partial_y = y[:index]
    partial_X = X[:index]
    
    m, b = lr.get_slope_intercept(partial_X, partial_y)
    y_line = [m*x + b for x in X]
    mae = lr.mean_absolute_error(y, y_line)
    
    plt.xlim((x_range[0] - 30, x_range[1] + 5))
    plt.ylim((np.min(y) -40, np.max(y) + 10))

    plt.scatter(X[index:], y[index:])
    plt.scatter(X[:index], partial_y)
    plt.plot(X, y_line, color='orange')
    plt.annotate(f"mae = {mae:.2f} \nModel: y = {m:.2f}*x + {b:.2f} \nReal: y = {slope}*x + {intercept}", (x_range[0] - 27, np.min(y) - 35))
    
    input('Pressione Enter para continuar')
    
    plt.show()
