# src_cnn_v2/hyperparam_search_v2.py
"""
Hyperparameter search for the V2 (bottom-up) pipeline using Optuna.
This script runs multiple training trials on the fixed train/validation split
to find the best set of hyperparameters for the SimpleCropRegressor model.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import argparse
import numpy as np
import pandas as pd
from torchvision import transforms

# --- FIX: Add missing imports ---
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
# --- END FIX ---

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import V2 components
from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.dataset_utils_v2 import CroppedSequenceDataset
from src_cnn_v2.models_v2 import SimpleCropRegressor
from src_cnn_v2.train_v2 import train_one_epoch, evaluate # Re-use train/eval functions

def objective(trial):
    """
    The Optuna objective function. A 'trial' represents one set of hyperparameters.
    """
    # 1. Define the Hyperparameter Search Space
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])

    # --- Data Loading (using the fixed V2 splits) ---
    CNN_DATASET_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.V2_DATASET_PARAMS["OUTPUT_SUBDIR"])
    TRAIN_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "train_metadata_v2.csv")
    VAL_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "val_metadata_v2.csv")

    try:
        train_df_orig = pd.read_csv(TRAIN_METADATA_PATH)
        val_df_orig = pd.read_csv(VAL_METADATA_PATH)
    except FileNotFoundError as e:
        raise optuna.exceptions.TrialPruned(f"Pruning trial due to missing data file: {e}")

    train_df, val_df = train_df_orig.copy(), val_df_orig.copy()

    # Apply scaling consistently
    scaler = RobustScaler() if cfg.SCALER_KIND == "robust" else StandardScaler()
    train_df['delta_T'] = scaler.fit_transform(train_df[['delta_T']])
    val_df['delta_T'] = scaler.transform(val_df[['delta_T']])

    # --- Data Transforms & Loaders ---
    norm_params = cfg.NORM_CONSTANTS[1]
    train_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    val_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    
    train_dataset = CroppedSequenceDataset(train_df, CNN_DATASET_DIR, transform=train_transform)
    val_dataset = CroppedSequenceDataset(val_df, CNN_DATASET_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model Initialization with Trial Hyperparameters ---
    model = SimpleCropRegressor(
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout_rate
    ).to(cfg.DEVICE)
    
    criterion = nn.MSELoss()
    
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Training Loop with Pruning ---
    best_val_mae = float('inf') 
    for epoch in range(cfg.NUM_EPOCHS_CV):
        train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_mae, val_rmse, val_r2 = evaluate(model, val_loader, cfg.DEVICE)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae

        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search for the V2 pipeline.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials to run.")
    parser.add_argument("--study_name", type=str, default="simple-crop-regressor-tuning", help="Name for the Optuna study.")
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10, n_min_trials=5)
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        pruner=pruner
    )
    
    study.optimize(objective, n_trials=args.n_trials)

    print("\n--- OPTUNA SEARCH COMPLETE ---")
    print(f"Study name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Best Validation MAE): {trial.value:.4f}")
    
    print("  Best Hyperparameters (copy to config_v2.py):")
    print("INITIAL_PARAMS = {")
    # A bit more logic to format the output nicely
    params = trial.params
    # Manually map optuna names to config names if they differ
    params_for_config = {
        'lr': params.get('lr'),
        'weight_decay': params.get('weight_decay'),
        'dropout_rate': params.get('dropout_rate'),
        'lstm_hidden_size': params.get('lstm_hidden_size'),
        'lstm_layers': params.get('lstm_layers'),
        'optimizer': params.get('optimizer')
    }
    for key, value in params_for_config.items():
        if isinstance(value, str):
            print(f"    '{key}': '{value}',")
        else:
            print(f"    '{key}': {value},")
    print("}")