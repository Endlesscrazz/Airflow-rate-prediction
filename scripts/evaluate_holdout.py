# scripts/evaluate_holdout.py
"""
Evaluates a final trained model on the unseen hold-out set.
This script is made flexible to evaluate any specified model on any specified dataset.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import r2_score, mean_squared_error
import argparse
import joblib

# --- Import project modules ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn.cnn_utils import AirflowSequenceDataset
from src_cnn.cnn_models import UltimateHybridRegressor, SimplifiedCnnAvgRegressor
from src_cnn import config as cfg

def main():
    # 1. Setup & Parse Arguments
    parser = argparse.ArgumentParser(description="Evaluate a final model on the hold-out dataset.")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'avg'],
                        help="The type of model to evaluate ('lstm' or 'avg').")
    # --- START OF FIX ---
    parser.add_argument("--dataset_dir", type=str, required=True, # Make this required again
                        help="Path to the root directory of the dataset used for training.")
    # --- END OF FIX ---
    parser.add_argument("--in_channels", type=int, required=True,
                        help="The number of input channels the model was trained with (1, 2, or 3).")
    parser.add_argument("--optuna_tuned", action='store_true',
                        help="Specify if loading a model that was tuned with Optuna.")
    args = parser.parse_args()

    # 2. Configure Paths and Logging
    CNN_DATASET_DIR = args.dataset_dir
    HOLDOUT_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "holdout_metadata.csv") 
    
    model_name_tag = f"{args.model_type}_{args.in_channels}ch"
    if args.optuna_tuned: model_name_tag += "_optuna"
    if cfg.ENABLE_PER_FOLD_SCALING: model_name_tag += f"_{cfg.SCALER_KIND}scaled"
        
    MODEL_PATH = f"trained_models_final/final_model_{model_name_tag}.pth"
    SCALER_PATH = f"trained_models_final/final_scaler_{model_name_tag}.pkl"
    
    print("--- Evaluating Final Model on Hold-Out Set ---")
    print(f"Model Type: {args.model_type.upper()} | Channels: {args.in_channels} | Tuned: {args.optuna_tuned}")
    print(f"Dataset: {CNN_DATASET_DIR}")
    print(f"Loading model from: {MODEL_PATH}")

    # ... (The rest of the script is correct and does not need to be changed) ...
    if not os.path.isfile(HOLDOUT_METADATA_PATH):
        print(f"FATAL ERROR: Hold-out metadata not found at '{HOLDOUT_METADATA_PATH}'")
        sys.exit(1)
    holdout_df = pd.read_csv(HOLDOUT_METADATA_PATH)

    if cfg.ENABLE_PER_FOLD_SCALING:
        print(f"\nApplying saved '{cfg.SCALER_KIND}' scaler to hold-out data...")
        if not os.path.exists(SCALER_PATH):
            print(f"FATAL ERROR: Saved scaler not found at {SCALER_PATH}")
            print("Please run the final training script first to generate the scaler.")
            sys.exit(1)
        
        scaler = joblib.load(SCALER_PATH)
        numeric_cols = [col for col in (cfg.CONTEXT_FEATURES + cfg.DYNAMIC_FEATURES) if col in holdout_df.columns]
        
        # Use the loaded scaler to TRANSFORM the hold-out data. DO NOT FIT.
        holdout_df[numeric_cols] = scaler.transform(holdout_df[numeric_cols])
        print("Hold-out data successfully scaled.")
    
    # 4. Data Transforms and Loaders
    norm_params = cfg.NORM_CONSTANTS.get(args.in_channels)
    if not norm_params: raise ValueError(f"Normalization constants not defined for {args.in_channels} channels.")
    
    val_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    
    holdout_dataset = AirflowSequenceDataset(holdout_df, CNN_DATASET_DIR, cfg.CONTEXT_FEATURES, cfg.DYNAMIC_FEATURES, val_transform, is_train=False)
    holdout_loader = DataLoader(holdout_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    if args.model_type == 'lstm':
        if args.optuna_tuned:
            model = UltimateHybridRegressor(
                num_context_features=len(cfg.CONTEXT_FEATURES),
                num_dynamic_features=len(cfg.DYNAMIC_FEATURES),
                cnn_in_channels=args.in_channels,
                lstm_hidden_size=cfg.OPTUNA_PARAMS['lstm_hidden_size'],
                lstm_layers=cfg.OPTUNA_PARAMS['lstm_layers']
            )
            model.head[2] = torch.nn.Dropout(cfg.OPTUNA_PARAMS['dropout_rate'])
        else:
            model = UltimateHybridRegressor(
                num_context_features=len(cfg.CONTEXT_FEATURES),
                num_dynamic_features=len(cfg.DYNAMIC_FEATURES),
                cnn_in_channels=args.in_channels
            )
    elif args.model_type == 'avg':
        model = SimplifiedCnnAvgRegressor(
            num_context_features=len(cfg.CONTEXT_FEATURES),
            num_dynamic_features=len(cfg.DYNAMIC_FEATURES),
            cnn_in_channels=args.in_channels
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    if not os.path.isfile(MODEL_PATH):
        print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'")
        sys.exit(1)
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)
    model.eval()

    all_targets, all_outputs = [], []
    with torch.no_grad():
        for seq, context, dynamic, targets in tqdm(holdout_loader, desc="Predicting"):
            seq, context, dynamic = seq.to(cfg.DEVICE), context.to(cfg.DEVICE), dynamic.to(cfg.DEVICE)
            outputs = model(seq, context, dynamic)
            all_targets.extend(targets.numpy())
            all_outputs.extend(outputs.cpu().numpy())

    r2 = r2_score(all_targets, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
    
    print("\n--- Hold-Out Set Performance ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_outputs, alpha=0.6, edgecolors='k')
    min_val = min(min(all_targets), min(all_outputs))
    max_val = max(max(all_targets), max(all_outputs))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("True Airflow Rate")
    plt.ylabel("Predicted Airflow Rate")
    plt.title(f"Hold-Out Set: True vs. Predicted (R² = {r2:.4f})")
    plt.legend()
    plt.grid(True)
    
    plot_save_dir = "holdout_results"
    os.makedirs(plot_save_dir, exist_ok=True)
    plot_save_path = os.path.join(plot_save_dir, f"holdout_plot_{model_name_tag}.png")
    plt.savefig(plot_save_path)
    print(f"\nSaved results plot to: {plot_save_path}")

if __name__ == "__main__":
    main()

"""
python -m scripts.evaluate_holdout \
    --model_type "lstm" \
    --dataset_dir "CNN_dataset/dataset_2ch_thermal_masked_f10s" \
    --in_channels 2 \
    --optuna_tuned
"""