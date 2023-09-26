import matplotlib.pyplot as plt
import numpy as np
import linear_regression as lr


X, y = lr.generate_linear_data(x_range=(-10, 10), m=-4, b=10, variance=10)


for index in range(1, len(X) + 1):
    
    
    plt.xlim((-20, 40))
    plt.ylim((-150, 100))
    
    m, b = lr.get_slope_intercept(X[:index], y[:index])
    print(m, b)
    y_line = [m*x + b for x in X]
    
    plt.scatter(X[index:], y[index:])
    plt.scatter(X[:index], y[:index])
    plt.plot(X, y_line, color='orange')
    plt.annotate("error=0.03", (-15, -140))
    
    input()

    
    plt.show()


