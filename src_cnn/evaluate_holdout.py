# src_cnn/evaluate_holdout.py
"""
Evaluates the final trained "Ultimate Hybrid" model on the unseen hold-out set.
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import r2_score, mean_squared_error

from src_cnn.cnn_utils import AirflowSequenceDataset
from src_cnn.cnn_models import UltimateHybridRegressor # <-- Use the correct model

# --- Configuration ---
CNN_DATASET_DIR = "cnn_dataset/dataset_cnn-lstm-all-split-holes" # <-- Use the correct dataset
HOLDOUT_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "holdout_metadata.csv")
MODEL_PATH = "trained_models_hybrid_final/final_model.pth" # <-- Correct model path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

# Define the feature sets to be loaded by the Dataset
CONTEXT_FEATURES = [
    'delta_T_log',
    'material_brick_cladding',
    'material_gypsum'
]
DYNAMIC_FEATURES = [
    'hotspot_area_log',
    'hotspot_avg_temp_change_rate_initial_norm',
    'overall_std_deltaT',
    'temp_max_overall_initial',
    'temp_std_avg_initial'
]

def main():
    print("--- Evaluating Final Hybrid Model on Hold-Out Set ---")
    
    # 1. Load Hold-Out Data
    holdout_df = pd.read_csv(HOLDOUT_METADATA_PATH)
    
    # Apply the same normalization as used in training (NO augmentation)
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Pass the feature lists to the Dataset
    holdout_dataset = AirflowSequenceDataset(
        holdout_df, 
        CNN_DATASET_DIR, 
        context_feature_cols=CONTEXT_FEATURES,
        dynamic_feature_cols=DYNAMIC_FEATURES,
        transform=val_transform
    )
    holdout_loader = DataLoader(holdout_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load the Trained Model
    model = UltimateHybridRegressor(
        num_context_features=len(CONTEXT_FEATURES),
        num_dynamic_features=len(DYNAMIC_FEATURES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. Get Predictions
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for seq, context, dynamic, targets in tqdm(holdout_loader, desc="Predicting"):
            seq, context, dynamic = seq.to(DEVICE), context.to(DEVICE), dynamic.to(DEVICE)
            outputs = model(seq, context, dynamic)
            all_targets.extend(targets.numpy())
            all_outputs.extend(outputs.cpu().numpy())

    # 4. Calculate and Print Metrics
    r2 = r2_score(all_targets, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
    
    print("\n--- Hold-Out Set Performance ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # 5. Generate and Save Visualization
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_outputs, alpha=0.6, edgecolors='k')
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("True Airflow Rate")
    plt.ylabel("Predicted Airflow Rate")
    plt.title(f"Hold-Out Set: True vs. Predicted (R² = {r2:.4f})")
    plt.legend()
    plt.grid(True)
    
    plot_save_path = "holdout_results_hybrid_plot.png"
    plt.savefig(plot_save_path)
    print(f"\nSaved results plot to: {plot_save_path}")
    plt.show()

if __name__ == "__main__":
    main()

# python -m src_cnn.evaluate_holdout