import math
import matplotlib.pyplot as plt
from calc_rmse_vectors import calculate_rmse

def plot_predictions_and_residuals(actual_prices, predicted_prices):
	indices = list(range(1, len(actual_prices) + 1))
	residuals = [actual - predicted for actual, predicted in zip(actual_prices, predicted_prices)]

	fig, (ax_prices, ax_residuals) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

	# Scatter points only, with dashed error lines between each actual and predicted value.
	ax_prices.scatter(indices, actual_prices, s=70, color="navy", label="Actual prices", zorder=3)
	ax_prices.scatter(indices, predicted_prices, s=70, color="crimson", label="Predicted prices", zorder=3)
	ax_prices.set_title("Housing Prices vs. Predictions")
	ax_prices.set_ylabel("Price ($)")
	ax_prices.set_xticks(indices)
	ax_prices.set_xticklabels([f"P{i}" for i in indices])
	ax_prices.grid(True, alpha=0.3)
	ax_prices.legend()

	for index, actual, predicted in zip(indices, actual_prices, predicted_prices):
		ax_prices.vlines(index, min(actual, predicted), max(actual, predicted), colors="gray", linestyles="dashed", alpha=0.8, linewidth=2)

	ax_residuals.axhline(0, color="black", linewidth=1)
	ax_residuals.bar(indices, residuals, color="steelblue", width=0.55)
	ax_residuals.set_title("Residuals (Actual - Predicted)")
	ax_residuals.set_xlabel("Sample")
	ax_residuals.set_ylabel("Residual ($)")
	ax_residuals.set_xticks(indices)
	ax_residuals.set_xticklabels([f"P{i}" for i in indices])
	ax_residuals.grid(True, axis="y", alpha=0.3)

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	# Mock housing data: actual sale prices vs. model predictions
	actual_prices = [250000, 310000, 180000, 420000, 515000]
	predicted_prices = [240000, 305000, 190000, 400000, 500000]

	rmse = calculate_rmse(actual_prices, predicted_prices)
	print(f"RMSE: {rmse:.2f}")
	plot_predictions_and_residuals(actual_prices, predicted_prices)


