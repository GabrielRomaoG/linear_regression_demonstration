import matplotlib.pyplot as plt
import numpy as np
import linear_regression as lr

x_range = (-20, 30)
slope = -3
intercept = 50
variance = 10

X, y = lr.generate_linear_data(x_range, slope, intercept, variance)

fig, (ax_main, ax_annotations) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [6, 1]})

for index in range(1, len(X) + 1):
    ax_main.clear()
    ax_annotations.clear()

    partial_y = y[:index]
    partial_X = X[:index]

    m, b = lr.get_slope_intercept(partial_X, partial_y)
    y_line = [m * x + b for x in X]
    mae = lr.mean_absolute_error(y, y_line)

    ax_main.set_xlim((x_range[0] - 10, x_range[1] + 10))
    ax_main.set_ylim((np.min(y) - 10, np.max(y) + 10))

    ax_main.scatter(X[index:], y[index:])
    ax_main.scatter(X[:index], partial_y)
    ax_main.plot(X, y_line, color='orange')

    ax_annotations.text(0, 0,
                        f"mae = {mae:.2f} \nModel: y = {m:.2f}*x + {b:.2f} \nReal: y = {slope}*x + {intercept}",
                        fontsize=12, color='black')

    ax_annotations.axis('off')
    plt.subplots_adjust(hspace=0.25)

    plt.pause(0.5)
