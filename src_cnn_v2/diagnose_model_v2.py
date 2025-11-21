# src_cnn_v2/diagnose_model_v2.py
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
import joblib

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.dataset_utils_v2 import CroppedSequenceDataset
from src_cnn_v2.models_v2 import SimpleCropRegressor

# --- Configuration ---
sns.set_theme(style="whitegrid")
PLOT_DIR = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "diagnostic_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def run_inference(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for seq, delta_t, targets_scaled in dataloader:
            seq = seq.to(device)
            delta_t = delta_t.to(device)
            
            outputs_scaled = model(seq, delta_t)
            
            # De-normalize
            preds = (outputs_scaled * cfg.MAX_FLOW_RATE).cpu().numpy()
            targets = (targets_scaled * cfg.MAX_FLOW_RATE).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    return np.array(all_preds).clip(min=0), np.array(all_targets)

def plot_diagnostic(df, split_name):
    """Generates diagnostic plots for a specific data split."""
    true_vals = df['airflow_rate']
    pred_vals = df['predicted_airflow']
    residuals = true_vals - pred_vals

    # 1. True vs Predicted Scatter
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=true_vals, y=pred_vals, alpha=0.6, edgecolor='k')
    
    # Perfect prediction line
    lims = [0, max(true_vals.max(), pred_vals.max()) * 1.05]
    plt.plot(lims, lims, 'r--', lw=2, label='Perfect Prediction')
    
    plt.title(f'{split_name} Set: True vs. Predicted', fontsize=14)
    plt.xlabel('True Airflow (L/min)')
    plt.ylabel('Predicted Airflow (L/min)')
    plt.legend()
    plt.xlim(lims)
    plt.ylim(lims)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{split_name}_true_vs_pred.png"))
    plt.close()

    # 2. Residual Plot (True Value vs Error) - Helps spot saturation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=true_vals, y=residuals, alpha=0.6, edgecolor='k')
    plt.axhline(0, color='r', linestyle='--')
    plt.title(f'{split_name} Set: Residuals (Bias Analysis)', fontsize=14)
    plt.xlabel('True Airflow (L/min)')
    plt.ylabel('Residual (True - Predicted)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{split_name}_residuals.png"))
    plt.close()

def main():
    print(f"--- Running Model Diagnostics ---")
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    print(f"Saving plots to: {PLOT_DIR}")

    # 1. Load Model and Scaler
    scaler_path = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "scaler_v2.pkl")
    model_path = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "best_model_v2.pth")

    if not os.path.exists(model_path):
        sys.exit("Model not found. Run training first.")

    scaler = joblib.load(scaler_path)
    
    model = SimpleCropRegressor(
        lstm_hidden_size=cfg.INITIAL_PARAMS['lstm_hidden_size'],
        lstm_layers=cfg.INITIAL_PARAMS['lstm_layers'],
        dropout=cfg.INITIAL_PARAMS['dropout_rate']
    ).to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    
    # 2. Prepare Transforms (Instance Norm is handled in Dataset, so standard norm is identity here usually)
    # But we respect config just in case
    norm_params = cfg.NORM_CONSTANTS[1]
    data_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])

    # --- DIAGNOSE TRAINING SET ---
    print("\n... Processing Training Set (this may take time) ...")
    train_df = pd.read_csv(cfg.TRAIN_METADATA_PATH)
    
    # OPTIONAL: Sample the training set if it's too huge (e.g., take 2000 samples)
    # train_df = train_df.sample(n=min(len(train_df), 2000), random_state=42) 
    
    train_df_scaled = train_df.copy()
    train_df_scaled['delta_T'] = scaler.transform(train_df_scaled[['delta_T']])
    
    train_ds = CroppedSequenceDataset(train_df_scaled, cfg.DATASET_DIR, transform=data_transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE*2, shuffle=False, num_workers=4)

    train_preds, train_targets = run_inference(model, train_loader, cfg.DEVICE)
    
    train_df['predicted_airflow'] = train_preds
    train_df['absolute_error'] = (train_df['airflow_rate'] - train_df['predicted_airflow']).abs()
    
    plot_diagnostic(train_df, "Training")
    
    # Save Training Report
    train_report_path = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "training_set_report.xlsx")
    train_df.to_excel(train_report_path, index=False)
    print(f"Training diagnosis saved to: {train_report_path}")

    # --- DIAGNOSE VALIDATION SET ---
    print("\n... Processing Validation Set ...")
    val_df = pd.read_csv(cfg.VAL_METADATA_PATH)
    val_df_scaled = val_df.copy()
    val_df_scaled['delta_T'] = scaler.transform(val_df_scaled[['delta_T']])

    val_ds = CroppedSequenceDataset(val_df_scaled, cfg.DATASET_DIR, transform=data_transform)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE*2, shuffle=False, num_workers=4)

    val_preds, val_targets = run_inference(model, val_loader, cfg.DEVICE)

    val_df['predicted_airflow'] = val_preds
    val_df['absolute_error'] = (val_df['airflow_rate'] - val_df['predicted_airflow']).abs()

    plot_diagnostic(val_df, "Validation")
    print("Validation diagnosis plots created.")

    print(f"\n--- Diagnosis Complete. Check {PLOT_DIR} ---")

if __name__ == "__main__":
    main()

# python -m src_cnn_v2.diagnose_model_v2