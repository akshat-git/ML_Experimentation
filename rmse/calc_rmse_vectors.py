import math
def calculate_rmse(actual_prices, predicted_prices):
	squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(actual_prices, predicted_prices)]
	return math.sqrt(sum(squared_errors) / len(squared_errors))
