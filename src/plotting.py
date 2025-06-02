# plotting.py
"""Plotting functions for the regression experiment."""

import config

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Using seaborn for nicer plots
import os
from sklearn.model_selection import learning_curve # For learning curves

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs. Predicted", save_path=None,
                             scatter_kwargs=None, add_jitter=False, jitter_strength=0.005,
                             material_labels=None, material_colors=None, legend_title="Material"): # New arguments
    """
    Creates a scatter plot of actual vs predicted values with a diagonal line.
    Can color points by material_labels if provided.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the plot. If None, plot is shown.
        scatter_kwargs (dict, optional): Additional keyword arguments for sns.scatterplot.
        add_jitter (bool): Whether to add small random noise to points for better visibility.
        jitter_strength (float): Magnitude of jitter to add.
        material_labels (array-like, optional): Labels for each data point indicating material type.
                                                Same length as y_true and y_pred.
        material_colors (dict, optional): Dictionary mapping material labels to colors.
                                          e.g., {'gypsum': 'blue', 'brick_cladding': 'green'}
        legend_title (str): Title for the material legend.
    """
    plt.figure(figsize=(9, 9)) # Slightly larger for legend
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    x_plot = y_true_np.copy()
    y_plot = y_pred_np.copy()

    if add_jitter:
        x_jitter = np.random.normal(0, jitter_strength, size=x_plot.shape)
        y_jitter = np.random.normal(0, jitter_strength, size=y_plot.shape)
        x_plot += x_jitter
        y_plot += y_jitter
        # print(f"  Note: Jitter (strength={jitter_strength}) applied.")

    default_scatter_args = {'s': 60, 'alpha': 0.7, 'edgecolors': 'k', 'linewidths': 0.5} # Default size increased
    if scatter_kwargs:
        default_scatter_args.update(scatter_kwargs)

    hue_arg = None
    palette_arg = None
    plot_data = pd.DataFrame({'Actual': x_plot, 'Predicted': y_plot})

    if material_labels is not None:
        if len(material_labels) == len(x_plot):
            plot_data[legend_title] = material_labels
            hue_arg = legend_title
            if material_colors:
                palette_arg = material_colors
        else:
            print("Warning: material_labels length mismatch. Plotting without material colors.")
            material_labels = None # Disable material coloring

    sns.scatterplot(data=plot_data, x='Actual', y='Predicted', hue=hue_arg, palette=palette_arg, **default_scatter_args)

    min_val_data = min(np.min(y_true_np), np.min(y_pred_np))
    max_val_data = max(np.max(y_true_np), np.max(y_pred_np))
    plot_margin = (max_val_data - min_val_data) * 0.05
    line_min_val = min_val_data - plot_margin
    line_max_val = max_val_data + plot_margin
    plt.plot([line_min_val, line_max_val], [line_min_val, line_max_val], 'r--', lw=2, label='Ideal (y=x)')

    plt.xlabel("Actual Airflow Rate")
    plt.ylabel("Predicted Airflow Rate")
    plt.title(title)
    if hue_arg: # If materials are plotted with hue
        plt.legend(loc='upper left', title=legend_title)
    else:
        plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlim(line_min_val, line_max_val)
    plt.ylim(line_min_val, line_max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_loss_curve(loss_curve_data, title, save_dir, filename="mlp_loss_curve.png"):
    """Plots the training loss curve for an MLP."""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_curve_data)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path_lc = os.path.join(save_dir, filename) # Renamed variable
        plt.savefig(save_path_lc, dpi=150)
        print(f"MLP loss curve saved to: {save_path_lc}")
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_learning_curves_custom(estimator, title, X, y, cv=None, n_jobs=None, 
                                train_sizes=np.linspace(.1, 1.0, 5), scoring='r2',
                                save_path=None):
    """
    Generates a plot of the test and training learning curve.

    Parameters:
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
    train_sizes : array-like, shape (n_ticks,), dtype (float or int)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the chosen validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature scorer(estimator, X, y).
    save_path : str, optional
        Path to save the plot. If None, plot is shown.
    """
    plt.figure(figsize=(10,6)) # Consistent figure size
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel(f"Score ({scoring if scoring else 'Default Score'})")

    # Calculate learning curve scores
    # Added shuffle=True and random_state for reproducibility if cv is KFold-like
    current_random_state = getattr(config, 'RANDOM_STATE', 42) if 'config' in globals() else 42 # Get from config if available
    
    # Make cv a KFold if it's an int, to enable shuffle
    if isinstance(cv, int):
        from sklearn.model_selection import KFold # Local import if needed
        cv = KFold(n_splits=cv, shuffle=True, random_state=current_random_state)


    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring,
        shuffle=True, random_state=current_random_state # Add shuffle and random_state here too
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1) # Cross-validation scores
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid(True)

    # Plot training scores
    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label="Training score")

    # Plot cross-validation scores
    plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    
    # Adjust y-limits based on typical R2 range, or make more adaptive if using other scores
    if scoring == 'r2':
        min_y_limit = np.min(test_scores_mean) - 0.1 # Start slightly below min CV score
        plt.ylim(max(-1.05, min_y_limit if np.isfinite(min_y_limit) else -0.1) , 1.05) # R2 can be negative
    else: # For scores like neg_mean_squared_error, adjust accordingly
        pass # May need custom y-limits for other scoring types

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(save_path, dpi=150)
        print(f"  Learning curve saved to: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()