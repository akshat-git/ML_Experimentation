import numpy as np


def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(X, y, theta, l2_strength=0.0):
	probabilities = sigmoid(X @ theta)
	probabilities = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
	loss = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))

	if l2_strength:
		loss += (l2_strength / (2.0 * len(y))) * np.sum(theta[1:] ** 2)

	return loss


def gradient_step(theta, X, y, learning_rate, l2_strength=0.0):
	count = len(y)
	probabilities = sigmoid(X @ theta)
	gradient = (X.T @ (probabilities - y)) / count

	if l2_strength:
		penalty = theta.copy()
		penalty[0] = 0.0
		gradient = gradient + (l2_strength / count) * penalty

	return theta - learning_rate * gradient
