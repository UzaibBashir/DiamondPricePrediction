"""Regression metrics, plots, and cross-validation helpers."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score


def evaluate_regression_model(model_name, actual_values, predicted_values):
    """Return standard regression metrics for one model."""
    return {
        "Model": model_name,
        "R2": r2_score(actual_values, predicted_values),
        "RMSE": float(mean_squared_error(actual_values, predicted_values) ** 0.5),
        "MAE": mean_absolute_error(actual_values, predicted_values),
        "MAPE": mean_absolute_percentage_error(actual_values, predicted_values),
    }


def plot_actual_vs_predicted(actual_values, predicted_values, model_name, axis=None):
    """Plot actual values against model predictions."""
    if axis is None:
        _, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(actual_values, predicted_values, alpha=0.3, color="steelblue", s=8)
    lower_bound = min(actual_values.min(), predicted_values.min())
    upper_bound = max(actual_values.max(), predicted_values.max())
    axis.plot([lower_bound, upper_bound], [lower_bound, upper_bound], "k--", linewidth=1, label="Perfect")
    axis.set_xlabel("Actual")
    axis.set_ylabel("Predicted")
    axis.set_title(f"Actual vs Predicted - {model_name}")
    axis.legend()
    return axis


def plot_residuals(actual_values, predicted_values, model_name, axis=None):
    """Plot prediction residuals."""
    if axis is None:
        _, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(predicted_values, actual_values - predicted_values, alpha=0.3, color="salmon", s=8)
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Residual")
    axis.set_title(f"Residuals - {model_name}")
    return axis


def cross_validate_regression_model(model, features, target, folds=5, scoring="r2"):
    """Run cross-validation and return the fold scores."""
    scores = cross_val_score(model, features, target, cv=folds, scoring=scoring)
    print(f"  CV {scoring}: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_model_results(results):
    """Sort model metric dictionaries by descending R2."""
    return pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)