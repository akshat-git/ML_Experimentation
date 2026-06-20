import numpy as np
import matplotlib.pyplot as plt

from gradient_step import binary_cross_entropy, gradient_step, sigmoid
from mock_data import make_admission_data


def fit_model(X, y):
    means = X.mean(axis=0)
    scales = X.std(axis=0)
    scales[scales == 0] = 1.0
    Xs = (X - means) / scales
    Xd = np.column_stack([np.ones(len(Xs)), Xs])

    theta = np.zeros(3)
    history = [theta.copy()]
    losses = [binary_cross_entropy(Xd, y, theta)]

    for _ in range(90):
        theta = gradient_step(theta, Xd, y, 0.12)
        history.append(theta.copy())
        losses.append(binary_cross_entropy(Xd, y, theta))

    return means, scales, Xd, theta, np.array(history), np.array(losses)


def plot_results(X, y, means, scales, theta, history, losses):
    Xs = (X - means) / scales
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    score_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 120)

    boundary_scores = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
    boundary_gpa = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    boundary_x, boundary_y = np.meshgrid(boundary_gpa, boundary_scores)
    grid = np.column_stack([boundary_x.ravel(), boundary_y.ravel()])
    grid_scaled = (grid - means) / scales
    grid_X = np.column_stack([np.ones(len(grid_scaled)), grid_scaled])
    probs = sigmoid(grid_X @ theta).reshape(boundary_x.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X[y == 0, 0], X[y == 0, 1], color="#457b9d", label="rejected")
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1], color="#e63946", label="admitted")
    axes[0].contour(boundary_x, boundary_y, probs, levels=[0.5], colors=["#1d3557"], linewidths=2)
    axes[0].set_title("Admission predictor")
    axes[0].set_xlabel("GPA")
    axes[0].set_ylabel("test score")
    axes[0].legend(frameon=False)

    axes[1].plot(losses, color="#2a9d8f", linewidth=2)
    axes[1].set_title("Loss across gradient steps")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("binary cross-entropy")
    axes[1].text(0.03, 0.95, f"final loss: {losses[-1]:.3f}", transform=axes[1].transAxes, va="top")

    fig.tight_layout()
    plt.show()


def main():
    X, y = make_admission_data(seed=9, size=80)
    means, scales, Xd, theta, history, losses = fit_model(X, y)
    plot_results(X, y, means, scales, theta, history, losses)


if __name__ == "__main__":
    main()