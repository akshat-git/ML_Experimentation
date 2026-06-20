import numpy as np
import matplotlib.pyplot as plt

from gradient_step import binary_cross_entropy, gradient_step, sigmoid
from mock_data import make_binary_data


def fit_model(X, y):
    X = np.column_stack([np.ones(len(X)), X[:, 0]])
    theta = np.zeros(2)
    history = [theta.copy()]

    for _ in range(70):
        theta = gradient_step(theta, X, y, 0.08)
        history.append(theta.copy())

    return X, theta, np.array(history)


def make_loss_grid(X, y, history):
    intercepts = np.linspace(history[:, 0].min() - 2.0, history[:, 0].max() + 2.0, 120)
    slopes = np.linspace(history[:, 1].min() - 2.0, history[:, 1].max() + 2.0, 120)
    grid = np.zeros((len(slopes), len(intercepts)))

    for row, slope in enumerate(slopes):
        for col, intercept in enumerate(intercepts):
            grid[row, col] = binary_cross_entropy(X, y, np.array([intercept, slope]))

    return intercepts, slopes, grid


def plot_results(x, y, X, theta, history):
    intercepts, slopes, grid = make_loss_grid(X, y, history)
    mesh_x, mesh_y = np.meshgrid(intercepts, slopes)
    x_line = np.linspace(x.min(), x.max(), 200)
    line_X = np.column_stack([np.ones(len(x_line)), x_line])
    probabilities = sigmoid(line_X @ theta)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(x[y == 0], y[y == 0], color="#457b9d", label="class A")
    axes[0].scatter(x[y == 1], y[y == 1], color="#e63946", label="class B")
    axes[0].plot(x_line, probabilities, color="#1d3557", linewidth=2.5, label="predicted probability")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_title("Binary classifier")
    axes[0].set_xlabel("feature x")
    axes[0].set_ylabel("class / probability")
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
    data, labels = make_binary_data(seed=4, size=60)
    x = data[:, 0]
    X, theta, history = fit_model(data, labels)
    plot_results(x, labels, X, theta, history)


if __name__ == "__main__":
    main()