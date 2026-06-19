import numpy as np
import matplotlib.pyplot as plt

from gradient_step import gradient_step


def make_data():
    rng = np.random.default_rng(22)
    base = rng.normal(0, 2.0, 85)
    x1 = base + rng.normal(0, 0.3, 85)
    x2 = 0.9 * base + rng.normal(0, 0.3, 85)
    X = np.column_stack([x1, x2])
    y = 0.5 + 4.2 * x1 + 4.0 * x2 + rng.normal(0, 1.8, 85)
    X = X - X.mean(axis=0)
    y = y - y.mean()
    return X, y


def fit_model(X, y):
    theta = np.zeros(2)
    history = [theta.copy()]

    for _ in range(90):
        theta = gradient_step(theta, X, y, 0.06, l2_strength=10.0)
        history.append(theta.copy())

    return theta, np.array(history)


def loss_value(X, y, theta):
    error = X @ theta - y
    mse = np.mean(error ** 2) / 2.0
    ridge = 10.0 * np.sum(theta[1:] ** 2) / (2.0 * len(y))
    return mse + ridge


def make_contour(X, y, history):
    w1_values = np.linspace(history[:, 0].min() - 1.8, history[:, 0].max() + 1.8, 120)
    w2_values = np.linspace(history[:, 1].min() - 1.8, history[:, 1].max() + 1.8, 120)
    grid = np.zeros((len(w2_values), len(w1_values)))

    for row, w2 in enumerate(w2_values):
        for col, w1 in enumerate(w1_values):
            grid[row, col] = loss_value(X, y, np.array([w1, w2]))

    return w1_values, w2_values, grid


def plot_results(X, y, theta, history):
    w1_values, w2_values, grid = make_contour(X, y, history)
    mesh_x, mesh_y = np.meshgrid(w1_values, w2_values)
    predictions = X @ theta

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y, predictions, color="slateblue", alpha=0.8)
    low = min(y.min(), predictions.min())
    high = max(y.max(), predictions.max())
    axes[0].plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.5)
    axes[0].set_title("Ridge regression")
    axes[0].set_xlabel("actual y")
    axes[0].set_ylabel("predicted y")
    axes[0].text(0.03, 0.97, f"final norm: {np.linalg.norm(theta):.3f}", transform=axes[0].transAxes, va="top")

    contour = axes[1].contour(mesh_x, mesh_y, grid, levels=25, cmap="cividis")
    axes[1].clabel(contour, inline=True, fontsize=7, fmt="%.1f")
    axes[1].plot(history[:, 0], history[:, 1], marker="o", markersize=3, color="purple")
    axes[1].scatter(history[0, 0], history[0, 1], color="black", s=40, label="start")
    axes[1].scatter(history[-1, 0], history[-1, 1], color="green", s=50, label="finish")
    axes[1].set_title("Gradient steps with L2")
    axes[1].set_xlabel("weight 1")
    axes[1].set_ylabel("weight 2")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    plt.show()


def main():
    X, y = make_data()
    theta, history = fit_model(X, y)
    plot_results(X, y, theta, history)


if __name__ == "__main__":
    main()