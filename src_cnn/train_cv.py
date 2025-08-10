# scripts/train.py
"""
Main script for running cross-validation on a given model and dataset.
All configurations are imported from src_cnn.config and command-line arguments.
"""
import os
import sys
import numpy as np
import itertools
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import random
import argparse
import joblib

# --- Import project modules ---

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn.cnn_utils import AirflowSequenceDataset
from src_cnn.cnn_models import UltimateHybridRegressor, SimplifiedCnnAvgRegressor
from src_cnn import config as cfg

# --- For Reproducibility ---
torch.manual_seed(cfg.RANDOM_STATE)
np.random.seed(cfg.RANDOM_STATE)
random.seed(cfg.RANDOM_STATE)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Reusable Training and Evaluation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for seq, context, dynamic, targets in dataloader:
        seq, context, dynamic, targets = seq.to(device), context.to(device), dynamic.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(seq, context, dynamic)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * seq.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for seq, context, dynamic, targets in dataloader:
            seq, context, dynamic, targets = seq.to(device), context.to(device), dynamic.to(device), targets.to(device)
            outputs = model(seq, context, dynamic)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * seq.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    r2 = r2_score(all_targets, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
    return epoch_loss, r2, rmse

# --- Main Execution ---
def main():
    # 1. Setup & Parse Arguments
    parser = argparse.ArgumentParser(description="Run cross-validation for a specific model.")
    parser.add_argument("--fold", type=int, required=True, help="The fold number to run (0-indexed).")
    parser.add_argument("--total_folds", type=int, required=True, help="The total number of folds in the CV.")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'avg'], 
                        help="Type of model to train: 'lstm' for UltimateHybrid or 'avg' for SimplifiedCnnAvg.")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--in_channels", type=int, required=True, 
                        help="Number of input channels for the CNN (1, 2, or 3).")
    args = parser.parse_args()

    current_fold = args.fold
    n_splits = args.total_folds
    is_interactive = sys.stdout.isatty()

    # 2. Configure Paths and Logging
    CNN_DATASET_DIR = args.dataset_dir
    METADATA_PATH = os.path.join(CNN_DATASET_DIR, "metadata.csv") 

    model_name_tag = f"{args.model_type}_{args.in_channels}ch_optuna"
    if cfg.ENABLE_PER_FOLD_SCALING:
        model_name_tag += f"_{cfg.SCALER_KIND}scaled"
        
    MODEL_SAVE_DIR = f"trained_models_{model_name_tag}_CV"
    RESULTS_SAVE_DIR = f"results_{model_name_tag}_CV"
    SCALER_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "scalers") # New directory for scalers

    print(f"--- Starting CNN-based Model Training (Cross-Validation) ---")
    print(f"Model Type: {args.model_type.upper()} | Fold: {current_fold + 1}/{n_splits} | Channels: {args.in_channels}")
    print(f"Dataset: {CNN_DATASET_DIR}")
    print(f"Using device: {cfg.DEVICE.upper()}")

    # 3. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"FATAL ERROR: Metadata file not found at {METADATA_PATH}")
        sys.exit(1)
    df_metadata = pd.read_csv(METADATA_PATH)
    X = df_metadata.drop("airflow_rate", axis=1)
    y = df_metadata["airflow_rate"]
    groups = df_metadata["video_id"]

    # 4. Setup Cross-Validation
    gkf = GroupKFold(n_splits=n_splits)
    train_idx, val_idx = list(gkf.split(X, y, groups))[current_fold]

    print(f"\n----- RUNNING FOLD {current_fold + 1}/{n_splits} -----")
    print(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}")

    # Use .copy() to prevent SettingWithCopyWarning when scaling
    train_df = df_metadata.iloc[train_idx].copy()
    val_df = df_metadata.iloc[val_idx].copy()

    available_context_features = [col for col in cfg.CONTEXT_FEATURES if col in train_df.columns]
    available_dynamic_features = [col for col in cfg.DYNAMIC_FEATURES if col in train_df.columns]
    
    print(f"\nUsing {len(available_context_features)} available context features: {available_context_features}")
    print(f"Using {len(available_dynamic_features)} available dynamic features: {available_dynamic_features}")

    if cfg.ENABLE_PER_FOLD_SCALING:
        print(f"\nApplying per-fold scaling with '{cfg.SCALER_KIND}' scaler...")
        
        # Identify all numeric tabular columns to be scaled
        numeric_cols = available_context_features + available_dynamic_features
        # Ensure we only try to scale columns that actually exist and are numeric
        numeric_cols = [col for col in numeric_cols if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col])]
        
        if cfg.SCALER_KIND == "robust":
            scaler = RobustScaler()
        elif cfg.SCALER_KIND == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler kind: {cfg.SCALER_KIND}")

        # Fit scaler ONLY on the training data for this fold
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        # Transform the validation data using the same fitted scaler
        val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
        
        if cfg.SAVE_SCALERS:
            os.makedirs(SCALER_SAVE_DIR, exist_ok=True)
            scaler_path = os.path.join(SCALER_SAVE_DIR, f"scaler_fold_{current_fold}.pkl")
            joblib.dump(scaler, scaler_path)
            print(f"Saved fitted scaler for fold {current_fold} to {scaler_path}")

    # 5. Data Transforms and Loaders
    norm_params = cfg.NORM_CONSTANTS.get(args.in_channels)
    if not norm_params: raise ValueError(f"Normalization constants not defined for {args.in_channels} channels in config.")
    
    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])
    ])
    val_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    
    train_dataset = AirflowSequenceDataset(train_df, CNN_DATASET_DIR, 
                                           context_feature_cols=available_context_features,
                                           dynamic_feature_cols=available_dynamic_features,
                                           transform=train_transform, is_train=True)
    val_dataset = AirflowSequenceDataset(val_df, CNN_DATASET_DIR, 
                                           context_feature_cols=available_context_features,
                                           dynamic_feature_cols=available_dynamic_features,
                                           transform=val_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 6. Initialize Model, Loss, and Optimizer
    if args.model_type == 'lstm':
        model = UltimateHybridRegressor(
            num_context_features=len(available_context_features),
            num_dynamic_features=len(available_dynamic_features),
            cnn_in_channels=args.in_channels,
            lstm_hidden_size=cfg.OPTUNA_PARAMS['lstm_hidden_size'],
            lstm_layers=cfg.OPTUNA_PARAMS['lstm_layers']
        ).to(cfg.DEVICE)
        model.head[2] = torch.nn.Dropout(cfg.OPTUNA_PARAMS['dropout_rate'])
    elif args.model_type == 'avg':
        model = SimplifiedCnnAvgRegressor(
            num_context_features=len(cfg.CONTEXT_FEATURES),
            num_dynamic_features=len(cfg.DYNAMIC_FEATURES),
            cnn_in_channels=args.in_channels
        ).to(cfg.DEVICE)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    print(f"Initialized {type(model).__name__} with {args.in_channels} input channels and Optuna-tuned parameters.")
    
    criterion = nn.MSELoss()
    
    param_groups = [{'params': model.cnn[7].parameters(), 'lr': cfg.OPTUNA_PARAMS['lr'] / 10}]
    if args.model_type == 'lstm':
        main_model_params = itertools.chain(model.lstm.parameters(), model.attention.parameters())
        param_groups.append({'params': main_model_params, 'lr': cfg.OPTUNA_PARAMS['lr']})
    shared_main_params = itertools.chain(model.context_mlp.parameters(), model.dynamic_mlp.parameters(), model.head.parameters())
    param_groups.append({'params': shared_main_params, 'lr': cfg.OPTUNA_PARAMS['lr']})
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.OPTUNA_PARAMS['weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
    best_val_r2 = -float('inf')
    model_save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_fold_{current_fold}.pth")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 7. Training Loop
    for epoch in tqdm(range(cfg.NUM_EPOCHS_CV), desc=f"Training Fold {current_fold+1}", disable=(not is_interactive)):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_loss, val_r2, val_rmse = evaluate(model, val_loader, criterion, cfg.DEVICE)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
             current_lr = optimizer.param_groups[-1]['lr']
             tqdm.write(f"Epoch {epoch+1:03d}/{cfg.NUM_EPOCHS_CV} | Train Loss: {train_loss:.4f} | Val R²: {val_r2:.4f} | LR: {current_lr:.1e}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), model_save_path)

    # 8. Report Final Results for this Fold
    print(f"\n----- Finished Fold {current_fold + 1} -----")
    print(f"Loading best model from: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    
    final_loss, final_r2, final_rmse = evaluate(model, val_loader, criterion, cfg.DEVICE)

    results_df = pd.DataFrame([{'fold': current_fold, 'r2': final_r2, 'rmse': final_rmse}])
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    results_csv_path = os.path.join(RESULTS_SAVE_DIR, f"fold_{current_fold}_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"Fold {current_fold + 1} Final R²: {final_r2:.4f}, RMSE: {final_rmse:.4f}")
    print(f"Results for this fold saved to: {results_csv_path}")

if __name__ == "__main__":
    main()