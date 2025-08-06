# scripts/train_final.py
"""
Trains a final model on the entire development set (all data except the hold-out set).
All configurations are imported from src_cnn.config and command-line arguments.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import random
import argparse

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

def main():
    # 1. Setup & Parse Arguments
    parser = argparse.ArgumentParser(description="Train a final model on the full development dataset.")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'avg'],
                        help="Type of model to train.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--in_channels", type=int, required=True,
                        help="Number of input channels for the CNN.")
    args = parser.parse_args()

    is_interactive = sys.stdout.isatty()

    # 2. Configure Paths and Logging
    CNN_DATASET_DIR = args.dataset_dir
    TRAIN_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "train_metadata.csv") 
    
    model_name_tag = f"{args.model_type}_{args.in_channels}ch_optuna"
    MODEL_SAVE_DIR = "trained_models_final"
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"final_model_{model_name_tag}.pth")

    print(f"--- Starting FINAL Model Training (Optuna-tuned) ---")
    print(f"Model Type: {args.model_type.upper()} | Channels: {args.in_channels}")
    print(f"Dataset: {CNN_DATASET_DIR}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")
    print(f"Using device: {cfg.DEVICE.upper()}")

    # 3. Load Data
    if not os.path.exists(TRAIN_METADATA_PATH):
        print(f"FATAL ERROR: train_metadata.csv not found at {TRAIN_METADATA_PATH}")
        sys.exit(1)
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    print(f"Loaded {len(train_df)} samples for final training.")
    
    # 4. Data Transforms and Loaders
    if args.in_channels == 1: NORM_MEAN, NORM_STD = [0.5], [0.5]
    elif args.in_channels == 2: NORM_MEAN, NORM_STD = [0.0, 0.0], [1.0, 1.0]
    elif args.in_channels == 3: NORM_MEAN, NORM_STD = [0.5, 0.0, 0.0], [0.5, 1.0, 1.0]
    else: raise ValueError(f"Unsupported number of channels: {args.in_channels}")

    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    train_dataset = AirflowSequenceDataset(
        train_df, CNN_DATASET_DIR, 
        context_feature_cols=cfg.CONTEXT_FEATURES,
        dynamic_feature_cols=cfg.DYNAMIC_FEATURES,
        transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 5. Initialize Model, Loss, and Optimizer
    if args.model_type == 'lstm':
        model = UltimateHybridRegressor(
            num_context_features=len(cfg.CONTEXT_FEATURES),
            num_dynamic_features=len(cfg.DYNAMIC_FEATURES),
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

    print(f"Initialized {type(model).__name__} with Optuna-tuned parameters.")
    
    criterion = nn.MSELoss()
    
    param_groups = [{'params': model.cnn[7].parameters(), 'lr': cfg.OPTUNA_PARAMS['lr'] / 10}]
    if args.model_type == 'lstm':
        main_model_params = itertools.chain(model.lstm.parameters(), model.attention.parameters())
        param_groups.append({'params': main_model_params, 'lr': cfg.OPTUNA_PARAMS['lr']})
    shared_main_params = itertools.chain(model.context_mlp.parameters(), model.dynamic_mlp.parameters(), model.head.parameters())
    param_groups.append({'params': shared_main_params, 'lr': cfg.OPTUNA_PARAMS['lr']})
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.OPTUNA_PARAMS['weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15)

    # 6. Training Loop
    for epoch in tqdm(range(cfg.NUM_EPOCHS_FINAL), desc="Final Training", disable=(not is_interactive)):
        model.train()
        running_loss = 0.0
        
        for seq, context, dynamic, targets in train_loader:
            # Correctly move each distinct tensor to the device
            seq, context, dynamic, targets = seq.to(cfg.DEVICE), context.to(cfg.DEVICE), dynamic.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(seq, context, dynamic)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * seq.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step(epoch_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[-1]['lr']
            tqdm.write(f"Epoch {epoch+1:03d}/{cfg.NUM_EPOCHS_FINAL} | Train Loss: {epoch_loss:.4f} | LR: {current_lr:.1e}")

    # 7. Save the Final Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nFinal trained model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()