# plotting.py
"""Plotting functions for the regression experiment."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Using seaborn for nicer plots

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs. Predicted", save_path=None):
    """
    Creates a scatter plot of actual vs predicted values with a diagonal line.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the plot. If None, plot is shown.
    """
    plt.figure(figsize=(8, 8))
    # Ensure input are numpy arrays for easier handling
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Scatter plot
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)

    # Diagonal line (y=x)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')

    # Labels and Title
    plt.xlabel("Actual Airflow Rate")
    plt.ylabel("Predicted Airflow Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Ensure equal scaling for x and y axes
    plt.tight_layout()

    if save_path:
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.close() # Close plot window if saving
    else:
        plt.show()
