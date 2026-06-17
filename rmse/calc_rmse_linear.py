
import math


def calc_rmse_linear(slope, intercept, points):

	predicted_values = [(slope * x) + intercept for x, _ in points]
	actual_values = [y for _, y in points]
	squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(actual_values, predicted_values)]
	return math.sqrt(sum(squared_errors) / len(squared_errors))