# plotting.py
"""Plotting functions for the regression experiment."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Using seaborn for nicer plots
import os

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs. Predicted", save_path=None,
                             scatter_kwargs=None, add_jitter=False, jitter_strength=0.005): # Added scatter_kwargs and jitter
    """
    Creates a scatter plot of actual vs predicted values with a diagonal line.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the plot. If None, plot is shown.
        scatter_kwargs (dict, optional): Additional keyword arguments for sns.scatterplot.
        add_jitter (bool): Whether to add small random noise to points for better visibility.
        jitter_strength (float): Magnitude of jitter to add (relative to data range if not careful).
                                 For your airflow rates (1.6-2.4), 0.005 might be reasonable.
    """
    plt.figure(figsize=(8, 8))
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    # --- Prepare data for plotting (with optional jitter) ---
    x_plot = y_true_np.copy()
    y_plot = y_pred_np.copy()

    if add_jitter:
        # Add a small amount of random noise to x and y coordinates
        # Adjust jitter_strength based on the scale of your data
        x_jitter = np.random.normal(0, jitter_strength, size=x_plot.shape)
        y_jitter = np.random.normal(0, jitter_strength, size=y_plot.shape)
        x_plot += x_jitter
        y_plot += y_jitter
        print(f"  Note: Jitter (strength={jitter_strength}) applied to scatter plot for visualization.")


    # --- Define default scatter plot arguments and update with user's ---
    default_scatter_args = {'s': 50, 'alpha': 0.7, 'edgecolors': 'k', 'linewidths': 0.5}
    if scatter_kwargs:
        default_scatter_args.update(scatter_kwargs)

    # Scatter plot using Seaborn for potentially nicer aesthetics
    sns.scatterplot(x=x_plot, y=y_plot, **default_scatter_args)

    # Diagonal line (y=x) - use original unjittered data for min/max
    min_val = min(np.min(y_true_np), np.min(y_pred_np))
    max_val = max(np.max(y_true_np), np.max(y_pred_np))
    # Extend line slightly for better visual appeal
    plot_margin = (max_val - min_val) * 0.05 
    line_min = min_val - plot_margin
    line_max = max_val + plot_margin
    plt.plot([line_min, line_max], [line_min, line_max], 'r--', lw=2, label='Ideal (y=x)')

    # Labels and Title
    plt.xlabel("Actual Airflow Rate")
    plt.ylabel("Predicted Airflow Rate")
    plt.title(title)
    plt.legend(loc='upper left')
    plt.grid(True)
    # Set axis limits to be slightly wider than the data range for clarity
    plt.xlim(line_min, line_max)
    plt.ylim(line_min, line_max)
    # plt.axis('equal') # 'equal' can sometimes make plots too small if ranges differ significantly,
                       # but good for y=x. Setting explicit limits above is often better.
    plt.gca().set_aspect('equal', adjustable='box') # Better way to enforce square plot with data limits
    plt.tight_layout()

    if save_path:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close() # Also close after showing interactively

def plot_loss_curve(loss_curve, title, save_dir, filename="mlp_loss_curve.png"):
    # ... (this function seems fine as is) ...
    plt.figure(figsize=(10, 5))
    plt.plot(loss_curve)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=150)
        print(f"MLP loss curve saved to: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()