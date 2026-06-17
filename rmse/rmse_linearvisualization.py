import matplotlib.pyplot as plt

from calc_rmse_linear import calc_rmse_linear


def plot_line_and_residuals(slope, intercept, points):
	x_values = [x for x, _ in points]
	actual_values = [y for _, y in points]
	predicted_values = [(slope * x) + intercept for x in x_values]
	residuals = [actual - predicted for actual, predicted in zip(actual_values, predicted_values)]
	plot_padding = 150

	fig, (ax_line, ax_residuals) = plt.subplots(
		2,
		1,
		figsize=(11, 9),
		gridspec_kw={"height_ratios": [2, 1]},
	)

	# Show the straight line against the actual points.
	ax_line.plot(x_values, predicted_values, color="crimson", linewidth=2.5, label=f"Line: y = {slope}x + {intercept}")
	ax_line.scatter(x_values, actual_values, color="navy", s=70, zorder=3, label="Housing points")

	for x_value, actual, predicted in zip(x_values, actual_values, predicted_values):
		ax_line.vlines(x_value, min(actual, predicted), max(actual, predicted), colors="gray", linestyles="dashed", alpha=0.75)

	ax_line.set_title("Line Fit Against Fake Housing Data")
	ax_line.set_xlabel("House size (sq ft)")
	ax_line.set_ylabel("Price ($)")
	ax_line.set_xlim(min(x_values) - plot_padding, max(x_values) + plot_padding)
	ax_line.grid(True, alpha=0.3)
	ax_line.legend()

	ax_residuals.axhline(0, color="black", linewidth=1)
	ax_residuals.bar(x_values, residuals, width=120, color="steelblue", edgecolor="white", align="center")
	ax_residuals.set_title("Residual Bars (Actual - Predicted)")
	ax_residuals.set_xlabel("House size (sq ft)")
	ax_residuals.set_ylabel("Residual ($)")
	ax_residuals.set_xlim(min(x_values) - plot_padding, max(x_values) + plot_padding)
	ax_residuals.set_xticks(x_values)
	ax_residuals.grid(True, axis="y", alpha=0.3)

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	# Fake housing dataset: (square feet, sale price)
	points = [
		(700, 145000),
		(900, 140000),
		(1100, 190000),
		(1300, 185000),
		(1500, 240000),
		(1700, 225000),
		(1900, 280000),
		(2100, 260000),
	]

	# A simple hand-picked line to test the function.
	slope = 110.0
	intercept = 60000.0

	rmse = calc_rmse_linear(slope, intercept, points)
	print(f"RMSE: {rmse:.2f}")
	plot_line_and_residuals(slope, intercept, points)
