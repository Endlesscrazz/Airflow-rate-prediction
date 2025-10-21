# src_cnn_v2/predict_v2.py
"""
Performs final evaluation on the TEST SET using the best trained V2 model.
This script is now fully config-driven to support versioned experiments.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import joblib

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.dataset_utils_v2 import CroppedSequenceDataset
from src_cnn_v2.models_v2 import SimpleCropRegressor
from src_cnn_v2.logging_utils_v2 import log_experiment_details

def main():
    print(f"--- V2 Model Final Evaluation on Test Set ---")
    print(f"Experiment: {cfg.EXPERIMENT_NAME} | Version: {cfg.EXPERIMENT_VERSION}")
    print(f"  - Using data split from random seed: {cfg.RANDOM_STATE}")

    # --- Setup Paths using the flexible config structure ---
    DATASET_DIR = cfg.DATASET_DIR
    # --- FIX: Use the versioned metadata path from the config ---
    TEST_METADATA_PATH = cfg.TEST_METADATA_PATH
    
    RESULTS_DIR = cfg.EXPERIMENT_RESULTS_DIR
    MODEL_PATH = os.path.join(RESULTS_DIR, "best_model_v2.pth")
    SCALER_PATH = os.path.join(RESULTS_DIR, "scaler_v2.pkl")
    test_report_path = os.path.join(RESULTS_DIR, "test_set_report.xlsx")

    # --- Sanity Checks ---
    for path in [MODEL_PATH, SCALER_PATH, TEST_METADATA_PATH]:
        if not os.path.exists(path):
            sys.exit(f"FATAL: Required file not found at '{path}'. Please run previous pipeline steps.")

    # --- Load Data and Artifacts ---
    test_df_orig = pd.read_csv(TEST_METADATA_PATH)
    test_df_scaled = test_df_orig.copy()
    print(f"Loaded {len(test_df_scaled)} test samples from: {DATASET_DIR}")

    scaler = joblib.load(SCALER_PATH)
    test_df_scaled['delta_T'] = scaler.transform(test_df_scaled[['delta_T']])

    # --- Create Dataset and DataLoader ---
    norm_params = cfg.NORM_CONSTANTS[1]
    test_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    
    test_dataset = CroppedSequenceDataset(test_df_scaled, DATASET_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Load Model ---
    model = SimpleCropRegressor(
        lstm_hidden_size=cfg.INITIAL_PARAMS['lstm_hidden_size'],
        lstm_layers=cfg.INITIAL_PARAMS['lstm_layers'],
        dropout=cfg.INITIAL_PARAMS['dropout_rate']
    ).to(cfg.DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.DEVICE))
    model.eval()
    print("Successfully loaded trained model and scaler.")

    # --- Perform Inference ---
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for seq, delta_t, targets_scaled in test_loader:
            seq, delta_t = seq.to(cfg.DEVICE), delta_t.to(cfg.DEVICE)
            outputs_scaled = model(seq, delta_t)
            all_targets.extend((targets_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())
            all_outputs.extend((outputs_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())

    all_outputs = np.array(all_outputs).clip(min=0)
    all_targets = np.array(all_targets)

    # --- Generate Comprehensive Report ---
    report_df = test_df_orig.copy()
    report_df['predicted_airflow'] = np.array(all_outputs).clip(min=0)
    report_df['absolute_error'] = (report_df['predicted_airflow'] - report_df['airflow_rate']).abs()
    report_df['percent_error'] = (report_df['absolute_error'] / report_df['airflow_rate']).abs().replace(np.inf, 0) * 100
    report_cols = ['original_sample_id', 'video_id', 'hole_id', 'source_dataset_key', 'airflow_rate', 'predicted_airflow', 'absolute_error', 'percent_error', 'delta_T']
    # Filter for columns that actually exist in the dataframe
    report_cols_exist = [col for col in report_cols if col in report_df.columns]
    report_df = report_df[report_cols_exist]

    # --- Calculate Summary Metrics ---
    test_mae = report_df['absolute_error'].mean()
    test_rmse = np.sqrt((report_df['absolute_error']**2).mean())
    ss_res = np.sum(report_df['absolute_error']**2)
    ss_tot = np.sum((report_df['airflow_rate'] - report_df['airflow_rate'].mean())**2)
    test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    nonzero_mask = report_df['airflow_rate'] > 1e-6
    test_mape = report_df.loc[nonzero_mask, 'percent_error'].mean()
    acc_10 = (report_df.loc[nonzero_mask, 'percent_error'] < 10).mean() * 100
    acc_25 = (report_df.loc[nonzero_mask, 'percent_error'] < 25).mean() * 100

    summary_metrics = {
        "Metric": ["R-squared (R²)", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", 
                   "Mean Absolute Percentage Error (MAPE)", "Accuracy @ 10%", "Accuracy @ 25%"],
        "Value": [test_r2, test_mae, test_rmse, test_mape, acc_10, acc_25],
        "Unit": ["-", "L/min", "L/min", "%", "%", "%"]
    }
    summary_df = pd.DataFrame(summary_metrics)
    
    # --- Save Report and Log Results ---
    with pd.ExcelWriter(test_report_path, engine='openpyxl') as writer:
        report_df.to_excel(writer, sheet_name='Predictions', index=False, float_format='%.4f')
        summary_df.to_excel(writer, sheet_name='Performance Summary', index=False)
    
    print(f"\nComprehensive test set report and summary saved to: {test_report_path}")

    final_test_metrics = summary_df.set_index('Metric')['Value'].to_dict()
    log_filepath = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "experiment_summary.txt")
    log_experiment_details(log_filepath, "Final Test Set Performance", final_test_metrics)
    
    print("\n--- FINAL TEST SET PERFORMANCE SUMMARY (Console) ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
    
# python src_cnn_v2/predict_v2.py