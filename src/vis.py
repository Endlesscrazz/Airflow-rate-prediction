# vis.py
"""Functions for visualizing feature analysis results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math

# Helper function to ensure directory exists
def _ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

# --- MODIFIED: Accepts full save_path ---
def plot_feature_vs_target(df, feature_columns, target_column, save_path=None):
    """Generates scatter plots for each feature against the target variable."""
    print(f"\n--- Generating Scatter Plots: Features vs. {target_column} ---")
    
    num_features = len(feature_columns)
    if num_features == 0: return
        
    ncols = min(3, num_features)
    nrows = math.ceil(num_features / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    axes = axes.flatten() 

    valid_feature_count = 0
    for i, feature in enumerate(feature_columns):
        if feature not in df.columns or target_column not in df.columns:
            print(f"Warning: Skipping scatter plot for '{feature}' due to missing column(s).")
            continue
            
        ax = axes[i]
        sns.scatterplot(data=df, x=feature, y=target_column, ax=ax, alpha=0.7)
        ax.set_title(f'{target_column} vs. {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel(target_column)
        ax.grid(True)
        valid_feature_count += 1

    # Hide any unused subplots
    for j in range(valid_feature_count, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    if save_path:
        _ensure_dir(save_path) # Ensure directory exists
        try:
            plt.savefig(save_path)
            print(f"Scatter plots saved to: {save_path}")
        except Exception as e:
            print(f"Error saving scatter plots to {save_path}: {e}")
            
    # plt.show() # Optional: Comment out if running non-interactively

# --- MODIFIED: Accepts full save_path ---
def plot_feature_distributions_by_target(df, feature_columns, target_column, save_path=None):
    """Generates box plots for each feature, grouped by the target variable."""
    print(f"\n--- Generating Box Plots: Feature Distributions by {target_column} ---")
    
    num_features = len(feature_columns)
    if num_features == 0: return
        
    ncols = min(3, num_features)
    nrows = math.ceil(num_features / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    axes = axes.flatten()

    valid_feature_count = 0
    for i, feature in enumerate(feature_columns):
        if feature not in df.columns or target_column not in df.columns:
            print(f"Warning: Skipping box plot for '{feature}' due to missing column(s).")
            continue

        ax = axes[i]
        df_copy = df.copy()
        # Convert target to category for proper grouping/labeling, handle potential NaNs first
        if df_copy[target_column].isnull().any():
             print(f"Warning: NaN values found in target '{target_column}' for boxplot grouping. Dropping.")
             df_copy.dropna(subset=[target_column], inplace=True)
        df_copy[target_column] = df_copy[target_column].astype('category') 
        
        sns.boxplot(data=df_copy, x=target_column, y=feature, ax=ax)
        ax.set_title(f'{feature} Distribution by {target_column}')
        ax.set_xlabel(target_column)
        ax.set_ylabel(feature)
        ax.grid(True, axis='y') 
        valid_feature_count += 1

    for j in range(valid_feature_count, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    if save_path:
        _ensure_dir(save_path)
        try:
            plt.savefig(save_path)
            print(f"Box plots saved to: {save_path}")
        except Exception as e:
            print(f"Error saving box plots to {save_path}: {e}")

    # plt.show() # Optional

# --- MODIFIED: Accepts full save_path ---
def plot_correlation_matrix(df, columns, save_path=None):
    """Generates a heatmap of the correlation matrix for the specified columns."""
    print("\n--- Generating Feature Correlation Matrix Heatmap ---")
    
    valid_columns = [col for col in columns if col in df.columns]
    if len(valid_columns) < 2:
        print("Warning: Need at least two valid columns for correlation matrix.")
        return
        
    correlation_matrix = df[valid_columns].corr()
    
    plt.figure(figsize=(max(8, len(valid_columns)*0.8), max(6, len(valid_columns)*0.8)))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1) # Set vmin/vmax
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() # Adjust layout after setting rotations

    if save_path:
        _ensure_dir(save_path)
        try:
            plt.savefig(save_path)
            print(f"Correlation matrix saved to: {save_path}")
        except Exception as e:
             print(f"Error saving correlation matrix to {save_path}: {e}")
             
    plt.show() # Optional