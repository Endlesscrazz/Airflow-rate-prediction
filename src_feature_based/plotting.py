# src_feature_based/plotting.py
"""
Plotting functions for the feature-based regression experiments.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupKFold

# Import the central config file
from src_feature_based import config as cfg

def plot_actual_vs_predicted(y_true, y_pred, title, save_path, material_labels=None):
    """Creates a scatter plot of actual vs predicted values."""
    plt.figure(figsize=(8, 8))
    
    plot_data = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
    hue_arg = None
    if material_labels is not None:
        plot_data['Material'] = material_labels
        hue_arg = 'Material'

    sns.scatterplot(data=plot_data, x='True', y='Predicted', hue=hue_arg, alpha=0.7, edgecolor='k')
    
    min_val = min(y_true.min(), y_pred.min()) - 0.1
    max_val = max(y_true.max(), y_pred.max()) + 0.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel("True Airflow Rate (L/min)")
    plt.ylabel("Predicted Airflow Rate (L/min)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to: {save_path}")
    plt.close()

def plot_learning_curves(estimator, title, X, y, groups, save_path):
    """Generates and saves a plot of the learning curve."""
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("R² Score")
    
    cv_splitter = GroupKFold(n_splits=cfg.CV_FOLDS)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, groups=groups, cv=cv_splitter,
        n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 7), scoring='r2'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.ylim(-0.1, 1.05) # Typical range for R2 score
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved learning curve to: {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, title, save_path):
    """Plots feature importance for tree-based models like XGBoost and RandomForest."""
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {type(model).__name__} does not have feature_importances_. Skipping plot.")
        return
        
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, max(6, len(feature_names) // 2)))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved feature importance plot to: {save_path}")
    plt.close()