# src_cnn_v2/hyperparam_search_v2.py
"""
Hyperparameter search for the V2 pipeline. This script is self-contained
to avoid import path issues and includes all necessary training logic.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler

# Add project root to path for imports BEFORE any local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import V2 components
from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.dataset_utils_v2 import CroppedSequenceDataset
from src_cnn_v2.models_v2 import SimpleCropRegressor
from src_cnn_v2.logging_utils_v2 import log_experiment_details


# --- Self-Contained Training and Evaluation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets_orig_scale, all_outputs_orig_scale = [], []
    for seq, delta_t, targets_scaled in dataloader:
        seq, delta_t, targets_scaled = seq.to(device), delta_t.to(device), targets_scaled.to(device)
        optimizer.zero_grad()
        outputs_scaled = model(seq, delta_t)
        loss = criterion(outputs_scaled, targets_scaled)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * seq.size(0)
        all_targets_orig_scale.extend((targets_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())
        all_outputs_orig_scale.extend((outputs_scaled * cfg.MAX_FLOW_RATE).detach().cpu().numpy())
    avg_train_loss = running_loss / len(dataloader.dataset)
    train_mae = np.mean(np.abs(np.array(all_targets_orig_scale) - np.array(all_outputs_orig_scale)))
    return avg_train_loss, train_mae

def evaluate(model, dataloader, device):
    model.eval()
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for seq, delta_t, targets_scaled in dataloader:
            seq, delta_t = seq.to(device), delta_t.to(device)
            outputs_scaled = model(seq, delta_t)
            all_targets.extend((targets_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())
            all_outputs.extend((outputs_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())
    all_outputs = np.array(all_outputs).clip(min=0)
    all_targets = np.array(all_targets)
    mae = np.mean(np.abs(all_targets - all_outputs))
    rmse = np.sqrt(np.mean((all_targets - all_outputs)**2))
    r2 = 1 - (np.sum((all_targets - all_outputs)**2) / np.sum((all_targets - np.mean(all_targets))**2)) if np.sum((all_targets - np.mean(all_targets))**2) > 0 else 0.0
    return mae, rmse, r2

# --- Optuna Objective Function ---

def objective(trial):
    """
    The Optuna objective function. A 'trial' represents one set of hyperparameters.
    """
    print(f"\n--- Starting Optuna Trial #{trial.number} ---")

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])

    DATASET_DIR = cfg.DATASET_DIR
    TRAIN_METADATA_PATH = cfg.TRAIN_METADATA_PATH
    VAL_METADATA_PATH = cfg.VAL_METADATA_PATH

    print(f"  - Using Dataset Directory: {DATASET_DIR}")
    print(f"  - Reading Train Metadata: {os.path.basename(TRAIN_METADATA_PATH)}")
    
    try:
        train_df_orig = pd.read_csv(TRAIN_METADATA_PATH)
        val_df_orig = pd.read_csv(VAL_METADATA_PATH)
    except FileNotFoundError as e:
        print(f"  - ERROR: Metadata file not found. Pruning trial. Details: {e}")
        raise optuna.exceptions.TrialPruned(f"Pruning trial due to missing data file: {e}")

    train_df, val_df = train_df_orig.copy(), val_df_orig.copy()
    scaler = RobustScaler() if cfg.SCALER_KIND == "robust" else StandardScaler()
    train_df['delta_T'] = scaler.fit_transform(train_df[['delta_T']])
    val_df['delta_T'] = scaler.transform(val_df[['delta_T']])

    norm_params = cfg.NORM_CONSTANTS[1]
    train_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    val_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    
    train_dataset = CroppedSequenceDataset(train_df, DATASET_DIR, transform=train_transform)
    val_dataset = CroppedSequenceDataset(val_df, DATASET_DIR, transform=val_transform)
    
    tuning_batch_size = cfg.BATCH_SIZE * 2
    train_loader = DataLoader(train_dataset, batch_size=tuning_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=tuning_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleCropRegressor(lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers, dropout=dropout_rate).to(cfg.DEVICE)
    criterion = nn.MSELoss()
    
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_mae = float('inf') 
    num_tuning_epochs = 40 
    for epoch in range(num_tuning_epochs):
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_mae, val_rmse, val_r2 = evaluate(model, val_loader, cfg.DEVICE)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae

        trial.report(val_mae, epoch)
        if trial.should_prune():
            print(f"  - Trial pruned at epoch {epoch+1} with Val MAE: {val_mae:.4f}")
            raise optuna.exceptions.TrialPruned()

    print(f"  - Trial #{trial.number} finished. Best Val MAE: {best_val_mae:.4f}")
    return best_val_mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search for the V2 pipeline.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials to run.")
    parser.add_argument("--study_name", type=str, default="simple-crop-regressor-tuning", help="Name for the Optuna study.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage database URL. (e.g., 'sqlite:///study.db')")
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10, n_min_trials=5)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner
    )
    
    n_existing_trials = len(study.trials)
    n_trials_to_run = max(0, args.n_trials - n_existing_trials)
    
    if n_trials_to_run > 0:
        print(f"Study '{args.study_name}' loaded. {n_existing_trials} trials already exist.")
        print(f"Running {n_trials_to_run} new trials to reach the goal of {args.n_trials}.")
        study.optimize(objective, n_trials=n_trials_to_run)
    else:
        print(f"Study '{args.study_name}' already has {n_existing_trials} trials. Goal of {args.n_trials} is met. No new trials will be run.")

    print("\n--- OPTUNA SEARCH COMPLETE ---")
    
    log_filepath = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "experiment_summary.txt")
    os.makedirs(cfg.EXPERIMENT_RESULTS_DIR, exist_ok=True)
    
    best_trial = study.best_trial
    tuning_summary = {
        "Study Name": study.study_name,
        "Total Number of Trials in Study": len(study.trials),
        "Best Trial Number": best_trial.number,
        "Best Validation MAE": best_trial.value,
        "Best Hyperparameters": best_trial.params
    }
    log_experiment_details(log_filepath, "Hyperparameter Search Results", tuning_summary)
    print("\nBest hyperparameters have been logged.")

    print(f"\nStudy name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")
    print("\nBest trial:")
    print(f"  Value (Best Validation MAE): {best_trial.value:.4f}")
    print("  Best Hyperparameters (copy to config_v2.py):")
    print("INITIAL_PARAMS = {")
    params = best_trial.params
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