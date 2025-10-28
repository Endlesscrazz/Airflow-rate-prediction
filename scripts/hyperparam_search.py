# scripts/hyperparam_search.py
"""
Hyperparameter search for the CNN-based models using Optuna.
This script runs multiple training trials to find the best set of hyperparameters.
All configurations are imported from src_cnn.config.
"""
import os
import sys
import torch
import torch.optim as optim
import optuna
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from torchvision import transforms
import itertools

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn.cnn_models import UltimateHybridRegressor
from src_cnn.cnn_utils import AirflowSequenceDataset
from src_cnn import config as cfg
from src_cnn.train import train_one_epoch, evaluate

def objective(trial, args):
    """
    The Optuna objective function. A 'trial' represents a single run with one
    set of hyperparameters.
    """
    # 1. Define the Hyperparameter Search Space
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.8)
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [128, 256, 512])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])

    # --- Data Loading and Splitting (using the specified fold) ---
    df_metadata = pd.read_csv(args.metadata_path)
    X = df_metadata.drop("airflow_rate", axis=1)
    y = df_metadata["airflow_rate"]
    groups = df_metadata["video_id"]
    
    gkf = GroupKFold(n_splits=args.total_folds)
    train_idx, val_idx = list(gkf.split(X, y, groups))[args.fold]
    train_df = df_metadata.iloc[train_idx].copy()
    val_df = df_metadata.iloc[val_idx].copy()
    
    available_context_features = [col for col in cfg.CONTEXT_FEATURES if col in train_df.columns]
    available_dynamic_features = [col for col in cfg.DYNAMIC_FEATURES if col in train_df.columns]

    # --- Data Transforms (Now dynamic based on in_channels) ---
    norm_params = cfg.NORM_CONSTANTS.get(args.in_channels)
    if not norm_params: raise ValueError(f"Normalization constants not defined for {args.in_channels} channels.")

    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])
    ])
    val_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    
    train_dataset = AirflowSequenceDataset(train_df, args.dataset_dir, available_context_features, available_dynamic_features, train_transform)
    val_dataset = AirflowSequenceDataset(val_df, args.dataset_dir, available_context_features, available_dynamic_features, val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model Initialization with Trial Hyperparameters ---
    model = UltimateHybridRegressor(
        num_context_features=len(available_context_features),
        num_dynamic_features=len(available_dynamic_features),
        cnn_in_channels=args.in_channels,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers
    ).to(cfg.DEVICE)
    
    model.head[2] = torch.nn.Dropout(dropout_rate)
    criterion = torch.nn.MSELoss()

    param_groups = [{'params': model.cnn[7].parameters(), 'lr': lr / 10}]
    main_model_params = itertools.chain(model.lstm.parameters(), model.attention.parameters())
    param_groups.append({'params': main_model_params, 'lr': lr})
    shared_main_params = itertools.chain(model.context_mlp.parameters(), model.dynamic_mlp.parameters(), model.head.parameters())
    param_groups.append({'params': shared_main_params, 'lr': lr})

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(param_groups, weight_decay=weight_decay)

    # --- Training Loop with Pruning ---
    best_val_rmse = float('inf') 
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)

        val_loss, val_rmse, val_mae, val_r2 = evaluate(model, val_loader, criterion, cfg.DEVICE)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse

        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials to run.")
    parser.add_argument("--num_epochs", type=int, default=75, help="Number of epochs per trial.")
    parser.add_argument("--fold", type=int, default=0, help="Which CV fold to use for hyperparameter tuning.")
    
    parser.add_argument("--in_channels", type=int, required=True, help="Number of input channels for the CNN.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata.csv for the dataset.")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--total_folds", type=int, default=cfg.CV_FOLDS)
    
    args = parser.parse_args()

    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10, n_min_trials=5)
    )
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("\n--- OPTUNA SEARCH COMPLETE ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("\nBest trial:")
    trial = study.best_trial
    # <-- MODIFIED: Report the best RMSE
    print(f"  Value (Best RMSE): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")