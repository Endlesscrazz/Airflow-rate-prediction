# src_cnn_v3/tune_v3.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import optuna
from optuna.trial import TrialState
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v3 import config_v3 as cfg
from src_cnn_v3.dataset_utils_v3 import HybridDataset
from src_cnn_v3.models_v3 import HybridCropRegressor

# --- CONFIGURATION ---
N_TRIALS = 100  # Number of experiments to run
EPOCHS_PER_TRIAL = 30 # Keep low for tuning, we just need to see convergence behavior

def asymmetric_loss(preds, targets, penalty):
    diff = preds - targets
    loss = torch.where(torch.abs(diff) < 1, 0.5 * diff ** 2, torch.abs(diff) - 0.5)
    weights = torch.where(diff > 0, penalty, 1.0)
    return (loss * weights).mean()

def objective(trial):
    # 1. Hyperparameter Search Space
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [64, 128, 256]),
        'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
        'over_pred_penalty': trial.suggest_float('over_pred_penalty', 1.0, 3.0)
    }

    # 2. Load Data (Using config paths)
    # Note: loading inside objective is safer for multiprocess, though slightly slower
    df_train = pd.read_csv(os.path.join(cfg.DATASET_DIR, "train_metadata.csv"))
    df_val = pd.read_csv(os.path.join(cfg.DATASET_DIR, "val_metadata.csv"))
    
    # 3. Fit Scaler
    feat_cols = ['delta_T']
    if cfg.USE_HANDCRAFTED_FEATURES:
        feat_cols += ['feat_area', 'feat_aspect', 'feat_extent']
    
    scaler = StandardScaler()
    scaler.fit(df_train[feat_cols].values)
    
    # 4. Datasets & Loaders
    train_ds = HybridDataset(df_train, cfg.DATASET_DIR, scaler)
    val_ds = HybridDataset(df_val, cfg.DATASET_DIR, scaler)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    # 5. Model
    model = HybridCropRegressor(
        num_tabular_features=len(feat_cols),
        lstm_hidden=params['lstm_hidden_size'],
        lstm_layers=params['lstm_layers'],
        dropout=params['dropout_rate']
    ).to(cfg.DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scaler_amp = torch.amp.GradScaler('cuda')

    # 6. Training Loop (Mini)
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for video, features, target in train_loader:
            video, features, target = video.to(cfg.DEVICE), features.to(cfg.DEVICE), target.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                preds = model(video, features)
                loss = asymmetric_loss(preds, target, params['over_pred_penalty'])
            
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
        # Validation
        model.eval()
        val_mae_accum = 0.0
        with torch.no_grad():
            for video, features, target in val_loader:
                video, features = video.to(cfg.DEVICE), features.to(cfg.DEVICE)
                with torch.amp.autocast('cuda'):
                    preds = model(video, features)
                    preds = torch.clamp(preds, 0, 1)
                
                # Unscale to calculate Real MAE
                real_preds = (preds * cfg.MAX_FLOW_RATE).cpu().numpy()
                real_targets = (target * cfg.MAX_FLOW_RATE).numpy()
                val_mae_accum += np.mean(np.abs(real_targets - real_preds)) * video.size(0)
                
        val_mae = val_mae_accum / len(val_ds)
        
        # Report to Optuna
        trial.report(val_mae, epoch)
        
        # Pruning (Stop bad trials early)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_mae

def main():
    print(f"--- V3 Hyperparameter Tuning: {cfg.EXPERIMENT_NAME} ---")
    print(f"Max Flow: {cfg.MAX_FLOW_RATE}")
    
    # Create study
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    print(f"Starting {N_TRIALS} trials...")
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("Tuning interrupted by user.")

    print("\n" + "="*40)
    print("BEST TRIAL RESULTS")
    print("="*40)
    print(f"Value (Validation MAE): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best params to file
    save_path = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "best_hyperparams.txt")
    os.makedirs(cfg.EXPERIMENT_RESULTS_DIR, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(str(study.best_params))
    print(f"\nSaved best parameters to: {save_path}")

if __name__ == "__main__":
    main()