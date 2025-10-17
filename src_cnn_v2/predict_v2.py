# src_cnn_v2/predict_v2.py
"""
Performs inference on the test set using the best trained V2 model.
Loads the saved model, scaler, and test data to generate predictions
and calculate a comprehensive set of performance metrics.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import joblib
import math

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.dataset_utils_v2 import CroppedSequenceDataset
from src_cnn_v2.models_v2 import SimpleCropRegressor
from src_cnn_v2.visualizations_v2 import plot_predictions_vs_true, plot_error_distribution,plot_diagnostic_distributions

def main():
    print("--- V2 Model Inference on Test Set ---")

    # --- Setup Paths ---
    CNN_DATASET_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.V2_DATASET_PARAMS["OUTPUT_SUBDIR"])
    # --- FIX: Point to the correct metadata file for the test set ---
    TEST_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "test_metadata_v2.csv")
    
    MODEL_DIR = os.path.join(cfg.OUTPUT_DIR, "trained_models")
    MODEL_PATH = os.path.join(MODEL_DIR, "best_model_v2.pth")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler_v2.pkl")

    # --- Sanity Checks ---
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"FATAL: Model file not found at '{MODEL_PATH}'. Please run train_v2.py first.")
    if not os.path.exists(SCALER_PATH):
        sys.exit(f"FATAL: Scaler file not found at '{SCALER_PATH}'. Please run train_v2.py first.")
    if not os.path.exists(TEST_METADATA_PATH):
        sys.exit(f"FATAL: Test metadata file not found at '{TEST_METADATA_PATH}'. Please run create_dataset_v2.py first.")

    # --- Load Data and Artifacts ---
    test_df_orig = pd.read_csv(TEST_METADATA_PATH)
    test_df_scaled = test_df_orig.copy() # Use a new df for scaled data
    print(f"Loaded {len(test_df_scaled)} test samples.")

    scaler = joblib.load(SCALER_PATH)
    test_df_scaled['delta_T'] = scaler.transform(test_df_scaled[['delta_T']])

    # --- Create Dataset and DataLoader ---
    norm_params = cfg.NORM_CONSTANTS[1]
    test_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    
    test_dataset = CroppedSequenceDataset(test_df_scaled, CNN_DATASET_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Load Model ---
    model = SimpleCropRegressor().to(cfg.DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.DEVICE))
    model.eval()
    print("Successfully loaded trained model and scaler.")

    # --- Perform Inference ---
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for seq, delta_t, targets in test_loader:
            seq, delta_t = seq.to(cfg.DEVICE), delta_t.to(cfg.DEVICE)
            
            outputs = model(seq, delta_t)
            
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(torch.expm1(outputs).cpu().numpy())

    all_outputs = np.array(all_outputs).clip(min=0)
    all_targets = np.array(all_targets)

    # --- Calculate Comprehensive Metrics ---
    mae = np.mean(np.abs(all_targets - all_outputs))
    rmse = np.sqrt(np.mean((all_targets - all_outputs)**2))
    
    ss_res = np.sum((all_targets - all_outputs)**2)
    ss_tot = np.sum((all_targets - np.mean(all_targets))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    nonzero_mask = all_targets > 1e-6
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((all_targets[nonzero_mask] - all_outputs[nonzero_mask]) / all_targets[nonzero_mask])) * 100
        acc_10 = np.mean(np.abs((all_targets[nonzero_mask] - all_outputs[nonzero_mask]) / all_targets[nonzero_mask]) < 0.10) * 100
        acc_25 = np.mean(np.abs((all_targets[nonzero_mask] - all_outputs[nonzero_mask]) / all_targets[nonzero_mask]) < 0.25) * 100
    else:
        mape, acc_10, acc_25 = float('nan'), float('nan'), float('nan')

    # --- Print Detailed Report ---
    print("\n--- DETAILED PREDICTION REPORT ---")
    print(f"{'Sample ID':<55} {'True':>8} {'Predicted':>10} {'Abs Error':>12}")
    print("-" * 97)
    
    results = []
    for i in range(len(all_targets)):
        # Use 'original_sample_id' if it exists, otherwise fall back to 'sample_id'
        sample_id_col = 'original_sample_id' if 'original_sample_id' in test_df_orig.columns else 'sample_id'
        sample_id = test_df_orig.iloc[i][sample_id_col]
        
        true_val = all_targets[i]
        pred_val = all_outputs[i]
        abs_err = abs(true_val - pred_val)
        results.append((sample_id, true_val, pred_val, abs_err))
    
    results.sort(key=lambda x: x[3], reverse=True)
    
    for sample_id, true_val, pred_val, abs_err in results:
        print(f"{sample_id:<55} {true_val:>8.4f} {pred_val:>10.4f} {abs_err:>12.4f}")
        
    print("\n--- FINAL TEST SET PERFORMANCE SUMMARY ---")
    print(f"  R-squared (R²):   {r2:.4f}")
    print(f"  MAE:              {mae:.4f} (L/min)")
    print(f"  RMSE:             {rmse:.4f} (L/min)")
    print(f"  MAPE:             {mape:.2f}%")
    print(f"  Accuracy @ 10%:   {acc_10:.2f}%")
    print(f"  Accuracy @ 25%:   {acc_25:.2f}%")
    print("-" * 42)

     # --- GENERATE PLOTS FOR DIAGNOSIS ---
    print("\n--- Generating Diagnostic Plots ---")
    plot_save_dir = os.path.join(cfg.OUTPUT_DIR, "visualizations_local_predict")
    os.makedirs(plot_save_dir, exist_ok=True)
    
    # Create a DataFrame of the results
    preds_df = pd.DataFrame({'true_airflow': all_targets, 'predicted_airflow': all_outputs})

    # Generate the plots
    plot_predictions_vs_true(preds_df, plot_save_dir, title="Local Predict Test Set")
    plot_error_distribution(preds_df, plot_save_dir, title="Local Predict Test Set")
    plot_diagnostic_distributions(test_df_orig, test_df_scaled, all_outputs, plot_save_dir)

if __name__ == "__main__":
    main()
# python src_cnn_v2/predict_v2.py