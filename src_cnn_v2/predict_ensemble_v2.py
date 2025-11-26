# src_cnn_v2/predict_ensemble.py
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.dataset_utils_v2 import CroppedSequenceDataset
from src_cnn_v2.models_v2 import SimpleCropRegressor

# --- CONFIGURATION ---
# List the paths to your 3 best models here
MODEL_PATHS = [
    "Output_CNN-LSTM/hardyboard_all_dataset_v2/iter-27-cs15-maksym-replica-hp-tuned-overpred-1.6_SEED_42/best_model_v2.pth",
    "Output_CNN-LSTM/hardyboard_all_dataset_v2/iter-27-cs15-maksym-replica-hp-tuned-overpred-1.6_SEED_43/best_model_v2.pth",
    #"Output_CNN-LSTM/hardyboard_all_dataset_v2/iter-27-cs15-maksym-replica-hp-tuned-overpred-2.3-rs42_SEED_44/best_model_v2.pth"
]
OUTPUT_FILE = "ensemble_test_report.xlsx"


def load_model(path):
    model = SimpleCropRegressor(
        lstm_hidden_size=cfg.INITIAL_PARAMS['lstm_hidden_size'],
        lstm_layers=cfg.INITIAL_PARAMS['lstm_layers'],
        dropout=cfg.INITIAL_PARAMS['dropout_rate']
    ).to(cfg.DEVICE)
    model.load_state_dict(torch.load(path, map_location=cfg.DEVICE))
    model.eval()
    return model

def main():
    print(f"--- Running Ensemble Prediction with {len(MODEL_PATHS)} Models ---")
    
    # 1. Load Data (Using the standard Test Split)
    test_df_orig = pd.read_csv(cfg.TEST_METADATA_PATH)
    test_dataset = CroppedSequenceDataset(test_df_orig, cfg.DATASET_DIR, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    
    all_targets = []
    ensemble_predictions = np.zeros(len(test_df_orig))

    # 2. Loop through each model
    for model_path in MODEL_PATHS:
        print(f"Processing model: {os.path.basename(os.path.dirname(model_path))}")
        model = load_model(model_path)
        
        model_preds = []
        current_targets = []
        
        with torch.no_grad():
            for seq, delta_t, targets_scaled in test_loader:
                seq, delta_t = seq.to(cfg.DEVICE), delta_t.to(cfg.DEVICE)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(seq, delta_t)
                    outputs = outputs.clamp(0, 1) # Clamp per model
                
                # Inverse scale
                preds_real = (outputs * cfg.MAX_FLOW_RATE).cpu().numpy()
                model_preds.extend(preds_real)
                
                if len(all_targets) == 0: # Only collect targets once
                    current_targets.extend((targets_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())

        # Add to ensemble sum
        ensemble_predictions += np.array(model_preds)
        if len(all_targets) == 0:
            all_targets = np.array(current_targets)

    # 3. Average
    ensemble_predictions /= len(MODEL_PATHS)
    
    # 4. Generate Report
    report_df = test_df_orig.copy()
    report_df['predicted_airflow'] = ensemble_predictions
    report_df['absolute_error'] = (report_df['predicted_airflow'] - report_df['airflow_rate']).abs()
    
    mae = report_df['absolute_error'].mean()
    rmse = np.sqrt((report_df['absolute_error']**2).mean())
    
    ss_res = np.sum((report_df['airflow_rate'] - report_df['predicted_airflow'])**2)
    ss_tot = np.sum((report_df['airflow_rate'] - report_df['airflow_rate'].mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    print("\n--- ENSEMBLE RESULTS ---")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    report_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved ensemble report to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

# python src_cnn_v2/predict_ensemble_v2.py