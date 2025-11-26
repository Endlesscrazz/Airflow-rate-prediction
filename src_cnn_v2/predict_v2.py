# src_cnn_v2/predict_v2.py
"""
Performs final evaluation on the TEST SET using the best trained V2 model.
UPDATED: Supports --seed argument to find models in _SEED_XX folders.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import joblib
import argparse

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.dataset_utils_v2 import CroppedSequenceDataset
from src_cnn_v2.models_v2 import SimpleCropRegressor
from src_cnn_v2.logging_utils_v2 import log_experiment_details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE, 
                        help="Random seed used for training (to locate the correct result folder)")
    args = parser.parse_args()

    print(f"--- V2 Model Final Evaluation on Test Set ---")
    print(f"Experiment: {cfg.EXPERIMENT_NAME} | Version: {cfg.EXPERIMENT_VERSION}")
    print(f"Target Seed: {args.seed}")

    # --- 1. Determine Results Directory ---
    # Check if the _SEED_XX folder exists (created by ensemble train script)
    seed_specific_dir = f"{cfg.EXPERIMENT_RESULTS_DIR}_SEED_{args.seed}"
    
    if os.path.exists(seed_specific_dir):
        RESULTS_DIR = seed_specific_dir
        print(f"Found seed-specific results directory: {RESULTS_DIR}")
    else:
        # Fallback to base directory
        RESULTS_DIR = cfg.EXPERIMENT_RESULTS_DIR
        print(f"Seed directory not found. checking base directory: {RESULTS_DIR}")

    DATASET_DIR = cfg.DATASET_DIR
    TEST_METADATA_PATH = cfg.TEST_METADATA_PATH
    
    MODEL_PATH = os.path.join(RESULTS_DIR, "best_model_v2.pth")
    SCALER_PATH = os.path.join(RESULTS_DIR, "scaler_v2.pkl")
    test_report_path = os.path.join(RESULTS_DIR, "test_set_report.xlsx")

    # --- Sanity Checks ---
    if not os.path.exists(MODEL_PATH):
        print(f"\nFATAL ERROR: Model file not found at: {MODEL_PATH}")
        print(f"Contents of directory {RESULTS_DIR}:")
        try:
            print(os.listdir(RESULTS_DIR))
        except FileNotFoundError:
            print("  (Directory does not exist)")
        sys.exit(1)

    if not os.path.exists(TEST_METADATA_PATH):
        sys.exit(f"FATAL: Metadata file not found at '{TEST_METADATA_PATH}'.")

    # --- Load Data ---
    test_df_orig = pd.read_csv(TEST_METADATA_PATH)
    test_df_scaled = test_df_orig.copy()
    print(f"Loaded {len(test_df_scaled)} test samples from: {DATASET_DIR}")

    # --- Handle Scaling ---
    if cfg.ENABLE_PER_FOLD_SCALING:
        if not os.path.exists(SCALER_PATH):
            sys.exit(f"FATAL: Scaler enabled in config but not found at '{SCALER_PATH}'.")
        print("Loading and applying scaler to Delta T...")
        scaler = joblib.load(SCALER_PATH)
        test_df_scaled['delta_T'] = scaler.transform(test_df_scaled[['delta_T']])
    else:
        print("Per-fold scaling is DISABLED. Using RAW Delta T values.")

    # --- Dataset & Loader ---
    test_dataset = CroppedSequenceDataset(test_df_scaled, DATASET_DIR, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Load Model ---
    model = SimpleCropRegressor(
        lstm_hidden_size=cfg.INITIAL_PARAMS['lstm_hidden_size'],
        lstm_layers=cfg.INITIAL_PARAMS['lstm_layers'],
        dropout=cfg.INITIAL_PARAMS['dropout_rate']
    ).to(cfg.DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.DEVICE))
    model.eval()
    print("Successfully loaded trained model.")

    # --- Inference ---
    all_targets, all_outputs = [], []
    
    with torch.no_grad():
        for seq, delta_t, targets_scaled in test_loader:
            seq, delta_t = seq.to(cfg.DEVICE), delta_t.to(cfg.DEVICE)
            
            with torch.amp.autocast('cuda'):
                outputs_scaled = model(seq, delta_t)
                outputs_scaled = outputs_scaled.clamp(0, 1) # Clamp predictions

            all_targets.extend((targets_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())
            all_outputs.extend((outputs_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())

    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)

    # --- Generate Report ---
    report_df = test_df_orig.copy()
    report_df['predicted_airflow'] = all_outputs
    report_df['absolute_error'] = (report_df['predicted_airflow'] - report_df['airflow_rate']).abs()
    
    report_df['percent_error'] = 0.0
    nonzero_mask = report_df['airflow_rate'] > 1e-6
    report_df.loc[nonzero_mask, 'percent_error'] = (
        report_df.loc[nonzero_mask, 'absolute_error'] / report_df.loc[nonzero_mask, 'airflow_rate']
    ).abs() * 100

    # Metrics
    test_mae = report_df['absolute_error'].mean()
    test_rmse = np.sqrt((report_df['absolute_error']**2).mean())
    ss_res = np.sum((report_df['airflow_rate'] - report_df['predicted_airflow'])**2)
    ss_tot = np.sum((report_df['airflow_rate'] - report_df['airflow_rate'].mean())**2)
    test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
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
    
    # Save
    with pd.ExcelWriter(test_report_path, engine='openpyxl') as writer:
        report_df.to_excel(writer, sheet_name='Predictions', index=False, float_format='%.4f')
        summary_df.to_excel(writer, sheet_name='Performance Summary', index=False)
    
    print(f"\nComprehensive test set report and summary saved to: {test_report_path}")
    print("\n--- FINAL TEST SET PERFORMANCE SUMMARY ---")
    print(summary_df.to_string(index=False))

    # Log
    final_test_metrics = summary_df.set_index('Metric')['Value'].to_dict()
    log_filepath = os.path.join(RESULTS_DIR, "experiment_summary.txt")
    log_experiment_details(log_filepath, "Final Test Set Performance", final_test_metrics)

if __name__ == "__main__":
    main()
    
# python src_cnn_v2/predict_v2.py --seed 42