import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import linear_regression as lr

plt.style.use("dark_background")

plt.rcParams.update(
    {
        "axes.facecolor": "#171721",
        "figure.facecolor": "#171721",
    }
)

index = 0

def main():
    x_range = (-20, 30)
    slope = -3
    intercept = 50
    variance = 15

    X, y = lr.generate_linear_data(x_range, slope, intercept, variance)

    fig, (ax_main, ax_annotations) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [6, 1]}
    )

    def next_iteration(event):
        global index
        index += 1
        
        ax_main.clear()
        ax_annotations.clear()

        partial_y = y[:index]
        partial_X = X[:index]

        m, b = lr.get_slope_intercept(partial_X, partial_y)
        y_line = [m * x + b for x in X]
        mae = lr.mean_absolute_error(y, y_line)

        ax_main.set_xlim((x_range[0] - 20, x_range[1] + 20))
        ax_main.set_ylim((np.min(y) - 20, np.max(y) + 20))

        ax_main.scatter(X[index:], y[index:], color="#5F68DE")
        ax_main.scatter(partial_X, partial_y, color="#e65239")
        ax_main.plot(X, y_line, color="orange")

        ax_annotations.text(
            0,
            0,
            f"mae = {mae:.2f} \nModel: y = {m:.2f}*x + {b:.2f} \nReal: y = {slope}*x + {intercept}",
            fontsize=14,
            color="white",
        )

        ax_annotations.axis("off")
        plt.subplots_adjust(hspace=0.25)

        plt.draw()

    next_button_ax = plt.axes([0.7, 0.01, 0.2, 0.05])  # Define the button's position and size

    next_button = Button(next_button_ax, 'Next Iteration', color='lightblue', hovercolor='skyblue')
    next_button.label.set_color('black')  # Set the text color to black

    next_button.on_clicked(next_iteration)

    plt.show()

if __name__ == '__main__':
    main()