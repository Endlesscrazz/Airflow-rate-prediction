# src_cnn_v3/predict_v3.py
import os
import sys
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v3 import config_v3 as cfg
from src_cnn_v3.dataset_utils_v3 import HybridDataset
from src_cnn_v3.models_v3 import HybridCropRegressor
from src_cnn_v3.logging_utils_v3 import log_section

# --- VISUALIZATION FUNCTIONS ---
sns.set_theme(style="whitegrid")

def plot_training_history(log_path, save_dir):
    """Plots Training Loss and Train vs Validation MAE."""
    if not os.path.exists(log_path):
        print(f"Warning: Training log not found at {log_path}")
        return

    print("Generating training history plots...")
    df = pd.read_csv(log_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Training History: {cfg.EXPERIMENT_NAME}', fontsize=16)

    # Plot 1: Loss
    sns.lineplot(data=df, x='epoch', y='train_loss', ax=ax1, color='blue', label='Train Loss')
    ax1.set_title("Training Loss Convergence")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Plot 2: MAE Comparison
    sns.lineplot(data=df, x='epoch', y='train_mae', ax=ax2, color='blue', label='Train MAE')
    sns.lineplot(data=df, x='epoch', y='val_mae', ax=ax2, color='orange', label='Validation MAE')
    
    # Mark Best Epoch
    best_row = df.loc[df['val_mae'].idxmin()]
    ax2.axvline(best_row['epoch'], color='green', linestyle='--', alpha=0.7, 
                label=f"Best Epoch ({int(best_row['epoch'])})")
    
    ax2.set_title("Train vs Validation MAE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (L/min)")
    ax2.legend()
    ax2.grid(True)

    save_path = os.path.join(save_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tolerance_curve(df, save_dir):
    """Plots the Cumulative Distribution Function (CDF) of the absolute error."""
    errors = np.sort(df['absolute_error'].values)
    n = len(errors)
    y = np.arange(1, n+1) / n * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(errors, y, linewidth=3, color='green')
    
    # Reference Lines
    acc_05 = (df['absolute_error'] < 0.5).mean() * 100
    acc_10 = (df['absolute_error'] < 1.0).mean() * 100
    
    plt.axvline(0.5, color='orange', linestyle='--', alpha=0.7)
    plt.axvline(1.0, color='blue', linestyle='--', alpha=0.7)
    
    plt.text(0.55, 5, f'Within ±0.5 L/min: {acc_05:.1f}%', color='orange', fontweight='bold')
    plt.text(1.05, 15, f'Within ±1.0 L/min: {acc_10:.1f}%', color='blue', fontweight='bold')

    plt.xlabel('Absolute Error (L/min)')
    plt.ylabel('Percentage of Test Samples (%)')
    plt.title(f'Model Success Rate (Error Tolerance)\n{cfg.EXPERIMENT_NAME}')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    
    limit_x = min(np.percentile(errors, 98), 5.0)
    plt.xlim(0, limit_x)
    plt.ylim(0, 105)
    
    plt.savefig(os.path.join(save_dir, "test_tolerance_curve.png"))
    plt.close()

def generate_evaluation_plots(df, save_dir):
    """Wrapper to generate all test set plots."""
    print("Generating evaluation plots...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Scatter Plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='airflow_rate', y='predicted_airflow', alpha=0.6, edgecolor='k')
    lims = [0, max(df['airflow_rate'].max(), df['predicted_airflow'].max()) * 1.05]
    plt.plot(lims, lims, 'r--', lw=2, label='Ideal')
    plt.xlabel('True Airflow (L/min)')
    plt.ylabel('Predicted Airflow (L/min)')
    plt.title(f'True vs Predicted: {cfg.EXPERIMENT_NAME}')
    plt.legend()
    #plt.axis('equal')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.savefig(os.path.join(save_dir, "test_scatter.png"))
    plt.close()

    # 2. Residual Plot
    residuals = df['airflow_rate'] - df['predicted_airflow']
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['airflow_rate'], y=residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('True Airflow (L/min)')
    plt.ylabel('Residual (True - Predicted)')
    plt.title('Residual Analysis')
    plt.savefig(os.path.join(save_dir, "test_residuals.png"))
    plt.close()

    # 3. Error Dist
    plt.figure(figsize=(10, 6))
    sns.histplot(df['absolute_error'], kde=True, bins=20)
    plt.xlabel('Absolute Error (L/min)')
    plt.title('Distribution of Prediction Errors')
    plt.savefig(os.path.join(save_dir, "test_error_dist.png"))
    plt.close()
    
    # 4. Tolerance Curve (New)
    plot_tolerance_curve(df, save_dir)

# --- MAIN EVALUATION LOGIC ---

def main():
    print(f"--- V3 Evaluation: {cfg.EXPERIMENT_NAME} ---")
    
    # 1. Load Artifacts
    model_path = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        sys.exit(f"FATAL: Model not found at {model_path}")
        
    print(f"Loading checkpoint from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=cfg.DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=cfg.DEVICE)

    scaler = checkpoint['scaler'] 
    
    # 2. Load Test Data
    test_meta_path = os.path.join(cfg.DATASET_DIR, "test_metadata.csv")
    df_test = pd.read_csv(test_meta_path)
    
    test_ds = HybridDataset(df_test, cfg.DATASET_DIR, scaler)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    
    # 3. Load Model
    num_features = scaler.mean_.shape[0]
    model = HybridCropRegressor(num_tabular_features=num_features).to(cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']} (Val MAE: {checkpoint['val_mae']:.4f})")

    # 4. Inference Loop
    all_preds = []
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for video, features, _ in test_loader:
            video = video.to(cfg.DEVICE)
            features = features.to(cfg.DEVICE)
            
            output = model(video, features)
            output = torch.clamp(output, 0.0, 1.0) # Clamp normalized range
            
            # Unscale to L/min
            output_real = output * cfg.MAX_FLOW_RATE
            all_preds.extend(output_real.cpu().numpy())

    # 5. Metrics Calculation
    results_df = df_test.copy()
    results_df['predicted_airflow'] = all_preds
    results_df['absolute_error'] = np.abs(results_df['predicted_airflow'] - results_df['airflow_rate'])
    results_df['squared_error'] = results_df['absolute_error'] ** 2
    
    mask = results_df['airflow_rate'] > 0.1
    results_df['percent_error'] = 0.0
    results_df.loc[mask, 'percent_error'] = (
        results_df.loc[mask, 'absolute_error'] / results_df.loc[mask, 'airflow_rate']
    ) * 100

    metrics = {
        "MAE (L/min)": results_df['absolute_error'].mean(),
        "RMSE (L/min)": np.sqrt(results_df['squared_error'].mean()),
        "R2 Score": 1 - (results_df['squared_error'].sum() / ((results_df['airflow_rate'] - results_df['airflow_rate'].mean())**2).sum()),
        
        # --- NEW LAYMAN METRICS ---
        "NMAE % (Normalized)": (results_df['absolute_error'].mean() / cfg.MAX_FLOW_RATE) * 100,
        "Accuracy (±0.5 L/min)": (results_df['absolute_error'] < 0.5).mean() * 100,
        "Accuracy (±1.0 L/min)": (results_df['absolute_error'] < 1.0).mean() * 100
    }

    # 6. Save Reports
    report_path = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "test_report.xlsx")
    results_df.to_excel(report_path, index=False)
    
    log_section("Test Set Evaluation", metrics)
    
    print("\n" + "="*40)
    print("TEST SET RESULTS")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:<25}: {v:.4f}")
    print("="*40)

    # 7. Generate ALL Plots
    # A. Training History
    log_path = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "training_log.csv")
    plot_training_history(log_path, cfg.EXPERIMENT_RESULTS_DIR)
    
    # B. Test Evaluation Plots (Includes Tolerance Curve)
    try:
        generate_evaluation_plots(results_df, cfg.EXPERIMENT_RESULTS_DIR)
        print(f"Plots saved to: {cfg.EXPERIMENT_RESULTS_DIR}")
    except Exception as e:
        print(f"Warning: Plot generation failed: {e}")

if __name__ == "__main__":
    main()

# python src_cnn_v3/predict_v3.py