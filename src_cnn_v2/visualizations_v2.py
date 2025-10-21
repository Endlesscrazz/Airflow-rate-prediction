# src_cnn_v2/visualizations_v2.py
"""
Contains functions to generate standard, professional plots for analyzing
model training and performance for the V2 pipeline.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg

# --- Plotting Configuration ---
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def plot_learning_curves(log_df, save_dir):
    """
    Plots Train vs. Validation learning curves for both Loss and MAE.
    This is the primary tool for diagnosing overfitting and underfitting.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle('Model Learning Curves', fontsize=18)

    # --- Subplot 1: Loss (MSE in log-space) ---
    ax1.plot(log_df['epoch'], log_df['train_loss'], label='Training Loss', color='blue', lw=2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(log_df['epoch'], log_df['val_rmse'], label='Validation RMSE', color='orange', lw=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss (MSE)', color='blue')
    ax1_twin.set_ylabel('Validation RMSE (L/min)', color='orange')
    ax1.set_title('Loss vs. Validation RMSE')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines + lines2, labels + labels2, loc='upper right')

    # --- Subplot 2: Mean Absolute Error (MAE) ---
    ax2.plot(log_df['epoch'], log_df['train_mae'], label='Training MAE', color='blue', lw=2)
    ax2.plot(log_df['epoch'], log_df['val_mae'], label='Validation MAE', color='green', lw=2, linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error (L/min)')
    ax2.set_title('Training vs. Validation MAE')
    ax2.legend(loc='upper right')

    save_path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(save_path)
    print(f"Saved learning curves plot to: {save_path}")
    plt.close()

def plot_predictions_vs_true(preds_df, save_dir):
    """Plots a scatter plot of predicted vs. true values for the test set."""
    true_vals = preds_df['airflow_rate']
    pred_vals = preds_df['predicted_airflow']
    
    mae = (preds_df['absolute_error']).mean()
    rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
    ss_res = np.sum((true_vals - pred_vals)**2)
    ss_tot = np.sum((true_vals - np.mean(true_vals))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    plt.figure(figsize=(9, 9))
    plt.scatter(true_vals, pred_vals, alpha=0.7, edgecolors='k', s=50)
    
    lims = [
        min(min(true_vals), min(pred_vals)) * 0.95,
        max(max(true_vals), max(pred_vals)) * 1.05
    ]
    plt.plot(lims, lims, 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('True Airflow Rate (L/min)', fontsize=12)
    plt.ylabel('Predicted Airflow Rate (L/min)', fontsize=12)
    plt.title(f'Test Set: True vs. Predicted \n {cfg.EXPERIMENT_NAME} \nR² =  {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}' , fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(lims)
    plt.ylim(lims)
    
    save_path = os.path.join(save_dir, "test_set_predictions_vs_true.png")
    plt.savefig(save_path)
    print(f"Saved prediction scatter plot to: {save_path}")
    plt.close()
    
def plot_error_distribution(preds_df, save_dir):
    """Plots a histogram and KDE of the prediction errors."""
    errors = preds_df['absolute_error']
    
    plt.figure(figsize=(12, 7))
    sns.histplot(preds_df['absolute_error'], kde=True, bins=15)
    plt.axvline(errors.mean(), color='r', linestyle='--', label=f'Mean Error: {errors.mean():.3f}')
    plt.title('Distribution of Absolute Prediction Errors on Test Set', fontsize=14)
    plt.xlabel('Absolute Error (L/min)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    
    save_path = os.path.join(save_dir, "test_set_error_distribution.png")
    plt.savefig(save_path)
    print(f"Saved error distribution plot to: {save_path}")
    plt.close()

def main():
    print(f"--- Generating V2 Visualizations ---")
    print(f"Experiment: {cfg.EXPERIMENT_NAME} | Version: {cfg.EXPERIMENT_VERSION}")
    
    # --- Use the flexible, versioned results directory ---
    RESULTS_DIR = cfg.EXPERIMENT_RESULTS_DIR
    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    test_report_path = os.path.join(RESULTS_DIR, "test_set_report.xlsx")
    
    # All plots will be saved in the same results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Reading from and saving plots to: {RESULTS_DIR}")
    
    # --- Generate Plots ---
    try:
        log_df = pd.read_csv(log_path)
        plot_learning_curves(log_df, RESULTS_DIR)
    except FileNotFoundError:
        print(f"Warning: Training log not found at '{log_path}'. Skipping learning curve plot.")
        
    try:
        test_preds_df = pd.read_excel(test_report_path, sheet_name='Predictions')
        plot_predictions_vs_true(test_preds_df, RESULTS_DIR)
        plot_error_distribution(test_preds_df, RESULTS_DIR)
    except FileNotFoundError:
        print(f"Warning: Test predictions report not found at '{test_report_path}'. Skipping test set plots.")
    except Exception as e:
        print(f"An error occurred while reading the Excel report: {e}")

    print("\n--- Visualization Complete ---")

if __name__ == "__main__":
    main()

# python src_cnn_v2/visualizations_v2.py