# src_cnn/hyperparam_search.py
"""
Hyperparameter search for the CNN-based models using Optuna.
This script runs multiple training trials to find the best set of hyperparameters.
"""
import os
import torch
import torch.optim as optim
import optuna
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Import the necessary components from your existing scripts
# We are essentially re-using the building blocks you've already created.
from src_cnn.cnn_models import UltimateHybridRegressor
from src_cnn.cnn_utils import AirflowSequenceDataset
from src_cnn.train_cnn import train_one_epoch, evaluate, CONTEXT_FEATURES, DYNAMIC_FEATURES
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def objective(trial, args):
    """
    The Optuna objective function. A 'trial' represents a single run with one
    set of hyperparameters.
    """
    
    # 1. Define the Hyperparameter Search Space
    #    Optuna will pick values from these ranges for each trial.
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
    train_df = df_metadata.iloc[train_idx]
    val_df = df_metadata.iloc[val_idx]
    
    # --- Data Transforms (Identical to your training script) ---
    NORM_MEAN = [0.5] # Hardcoded for 1-channel thermal
    NORM_STD = [0.5]
    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), value=0),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    val_transform = transforms.Compose([transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)])
    
    train_dataset = AirflowSequenceDataset(train_df, args.dataset_dir, CONTEXT_FEATURES, DYNAMIC_FEATURES, train_transform)
    val_dataset = AirflowSequenceDataset(val_df, args.dataset_dir, CONTEXT_FEATURES, DYNAMIC_FEATURES, val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Model Initialization with Trial Hyperparameters ---
    model = UltimateHybridRegressor(
        num_context_features=len(CONTEXT_FEATURES),
        num_dynamic_features=len(DYNAMIC_FEATURES),
        cnn_in_channels=1,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers
    ).to(DEVICE)
    
    # Modify the dropout rate in the model's head
    model.head[2] = torch.nn.Dropout(dropout_rate)

    criterion = torch.nn.MSELoss()
    
    # --- Optimizer Selection ---
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Training Loop with Pruning ---
    best_val_r2 = -float('inf')
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_r2, val_rmse = evaluate(model, val_loader, criterion, DEVICE)
        
        # Update the best score
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2

        # 1. Report intermediate results to Optuna.
        trial.report(val_r2, epoch)

        # 2. Handle pruning (stop unpromising trials early).
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the best R² score achieved during this trial
    return best_val_r2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials to run.")
    parser.add_argument("--num_epochs", type=int, default=75, help="Number of epochs per trial.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fold", type=int, default=0, help="Which CV fold to use for hyperparameter tuning.")
    parser.add_argument("--total_folds", type=int, default=5)
    
    # We fix the dataset for this search
    DATASET_DIR = "CNN_dataset/dataset_1ch_thermal"
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--metadata_path", type=str, default=os.path.join(DATASET_DIR, "metadata.csv"))
    
    args = parser.parse_args()

    # Create a study. We want to maximize R², so direction is 'maximize'.
    # The pruner stops bad trials early.
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10) # Don't prune before epoch 10
    )
    
    # Start the optimization
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    # Print the results
    print("\n--- OPTUNA SEARCH COMPLETE ---")
    print(f"Number of finished trials: {len(study.trials)}")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Best R²): {trial.value:.4f}")
    
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")