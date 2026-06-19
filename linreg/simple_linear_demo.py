import numpy as np
import matplotlib.pyplot as plt

from gradient_step import gradient_step


def make_data():
    rng = np.random.default_rng(12)
    x = rng.uniform(-4, 4, 60)
    y = 4.5 + 2.75 * x + rng.normal(0, 1.15, 60)
    return x, y


def fit_model(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    theta = np.zeros(2)
    history = [theta.copy()]

    for _ in range(80):
        theta = gradient_step(theta, X, y, 0.04)
        history.append(theta.copy())

    return X, theta, np.array(history)


def loss_value(X, y, theta):
    error = X @ theta - y
    return np.mean(error ** 2) / 2.0


def make_contour(X, y, history):
    intercept_values = np.linspace(history[:, 0].min() - 2, history[:, 0].max() + 2, 120)
    slope_values = np.linspace(history[:, 1].min() - 1.5, history[:, 1].max() + 1.5, 120)
    grid = np.zeros((len(slope_values), len(intercept_values)))

    for row, slope in enumerate(slope_values):
        for col, intercept in enumerate(intercept_values):
            grid[row, col] = loss_value(X, y, np.array([intercept, slope]))

    return intercept_values, slope_values, grid


def plot_results(x, y, theta, history, X):
    intercept_values, slope_values, grid = make_contour(X, y, history)
    mesh_x, mesh_y = np.meshgrid(intercept_values, slope_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(x, y, color="steelblue", alpha=0.8, label="fake data")
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = theta[0] + theta[1] * x_line
    axes[0].plot(x_line, y_line, color="crimson", linewidth=2.5, label="learned line")
    axes[0].set_title("Simple linear regression")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(frameon=False)

    contour = axes[1].contour(mesh_x, mesh_y, grid, levels=25, cmap="viridis")
    axes[1].clabel(contour, inline=True, fontsize=7, fmt="%.1f")
    axes[1].plot(history[:, 0], history[:, 1], marker="o", markersize=3, color="crimson")
    axes[1].scatter(history[0, 0], history[0, 1], color="black", s=40, label="start")
    axes[1].scatter(history[-1, 0], history[-1, 1], color="green", s=50, label="finish")
    axes[1].set_title("Gradient steps")
    axes[1].set_xlabel("intercept")
    axes[1].set_ylabel("slope")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    plt.show()


def main():
    x, y = make_data()
    X, theta, history = fit_model(x, y)
    plot_results(x, y, theta, history, X)


if __name__ == "__main__":
    main()