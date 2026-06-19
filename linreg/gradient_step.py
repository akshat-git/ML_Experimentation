import numpy as np


def gradient_step(theta, X, y, learning_rate, l2_strength=0.0):
    count = len(y)
    predictions = X @ theta
    error = predictions - y
    gradient = (X.T @ error) / count

    if l2_strength:
        penalty = theta.copy()
        penalty[0] = 0.0
        gradient = gradient + (l2_strength / count) * penalty

    return theta - learning_rate * gradient