# src_cnn_v2/hyperparam_search_v2.py
"""
Hyperparameter optimization script for the V2 pipeline using Optuna. (Version 2 - Robust)

This script automates the search for the best model hyperparameters.
- It saves progress to a SQLite database, allowing the search to be resumed if interrupted.
- It uses a Median Pruner to stop unpromising trials early.

How to Run:
  python src_cnn_v2/hyperparam_search_v2.py
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import optuna

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2.models_v2 import SimpleCropRegressor
from src_cnn_v2.dataset_utils_v2 import CroppedSequenceDataset
from src_cnn_v2 import config_v2 as cfg

# --- Training and evaluation functions (unchanged) ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    for seq, delta_t, targets_scaled in dataloader:
        seq, delta_t, targets_scaled = seq.to(device), delta_t.to(device), targets_scaled.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs_scaled = model(seq, delta_t)
            loss = criterion(outputs_scaled, targets_scaled)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * seq.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for seq, delta_t, targets_scaled in dataloader:
            seq, delta_t = seq.to(device), delta_t.to(device)
            with torch.cuda.amp.autocast():
                outputs_scaled = model(seq, delta_t)
            all_targets.extend((targets_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())
            all_outputs.extend((outputs_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())
    all_outputs = np.array(all_outputs).clip(min=0)
    all_targets = np.array(all_targets)
    mae = np.mean(np.abs(all_targets - all_outputs))
    return mae

# --- The Core Optuna Objective Function (unchanged) ---
def objective(trial: optuna.Trial):
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.6),
        'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [64, 128, 256]),
        'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW']),
    }
    train_df_orig = pd.read_csv(cfg.TRAIN_METADATA_PATH)
    val_df_orig = pd.read_csv(cfg.VAL_METADATA_PATH)
    train_df, val_df = train_df_orig.copy(), val_df_orig.copy()
    if cfg.ENABLE_PER_FOLD_SCALING:
        scaler_tabular = RobustScaler() if cfg.SCALER_KIND == "robust" else StandardScaler()
        train_df['delta_T'] = scaler_tabular.fit_transform(train_df[['delta_T']])
        val_df['delta_T'] = scaler_tabular.transform(val_df[['delta_T']])
    train_dataset = CroppedSequenceDataset(train_df, cfg.DATASET_DIR)
    val_dataset = CroppedSequenceDataset(val_df, cfg.DATASET_DIR)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    model = SimpleCropRegressor(
        lstm_hidden_size=params['lstm_hidden_size'],
        lstm_layers=params['lstm_layers'],
        dropout=params['dropout_rate']
    ).to(cfg.DEVICE)
    if params['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.SmoothL1Loss()
    scaler_amp = torch.cuda.amp.GradScaler()
    best_val_mae = float('inf')
    for epoch in range(cfg.NUM_EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE, scaler_amp)
        val_mae = evaluate(model, val_loader, cfg.DEVICE)
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if val_mae < best_val_mae:
            best_val_mae = val_mae
    return best_val_mae


def main():
    print(f"--- Starting Hyperparameter Search for: {cfg.EXPERIMENT_NAME} ---")
  
    os.makedirs(cfg.EXPERIMENT_RESULTS_DIR, exist_ok=True)
    storage_name = f"sqlite:///{os.path.join(cfg.EXPERIMENT_RESULTS_DIR, 'hyperparam_search.db')}"
    study_name = f"{cfg.EXPERIMENT_NAME}_{cfg.EXPERIMENT_VERSION}"

    print(f"  - Using storage: {storage_name}")
    print(f"  - Study name: {study_name}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True, 
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    

    try:
        study.optimize(
            objective, 
            n_trials=100,
            timeout=3600 * 6
        )
    except KeyboardInterrupt:
        print("--- Search interrupted by user. ---")

    print("\n--- Hyperparameter Search Complete ---")
    print(f"  - Number of finished trials in this study: {len(study.trials)}")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    print(f"  - Pruned trials: {len(pruned_trials)}")
    print(f"  - Complete trials: {len(complete_trials)}")

    best_trial = study.best_trial
    print(f"\n  - Best trial finished with value (MAE): {best_trial.value:.6f}")
    
    print("\n  - Best hyperparameters found (copy to config_v2.py):")
    print("INITIAL_PARAMS = {")
    for key, value in best_trial.params.items():
        if isinstance(value, str):
            print(f"    '{key}': '{value}',")
        else:
            print(f"    '{key}': {value},")
    print("}")

if __name__ == "__main__":
    main()