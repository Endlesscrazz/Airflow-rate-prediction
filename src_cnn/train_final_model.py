# src_cnn/train_final_model.py
"""
Trains a final model on the entire development set (all data except the hold-out set).
This script is made flexible to train any specified model on any specified dataset.
"""
import os
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
import sys
import argparse

# Import our custom modules
from src_cnn.cnn_utils import AirflowSequenceDataset
from src_cnn.cnn_models import UltimateHybridRegressor, SimplifiedCnnAvgRegressor

# --- Default Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

# Hyperparameters from best CV runs (can be overridden if needed)
BATCH_SIZE = 8
LEARNING_RATE = 0.000242  # From Optuna
NUM_EPOCHS = 150
WEIGHT_DECAY = 0.000289  # From Optuna

# Feature Lists
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

# For reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(RANDOM_STATE)

def main():
    # 1. Setup & Parse Arguments
    parser = argparse.ArgumentParser(description="Train a final model on the full development dataset.")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'avg'],
                        help="Type of model to train: 'lstm' for UltimateHybrid or 'avg' for SimplifiedCnnAvg.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--in_channels", type=int, required=True,
                        help="Number of input channels for the CNN (1, 2, or 3).")
    args = parser.parse_args()

    is_interactive = sys.stdout.isatty()

    # 2. Configure Paths and Logging
    CNN_DATASET_DIR = args.dataset_dir
    TRAIN_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "metadata.csv") 
    
    model_name_tag = f"{args.model_type}_{args.in_channels}ch"
    MODEL_SAVE_DIR = f"trained_models_final"
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"final_model_{model_name_tag}.pth")

    print(f"--- Starting FINAL Model Training on ALL Development Data ---")
    print(f"Model Type: {args.model_type.upper()} | Channels: {args.in_channels}")
    print(f"Dataset: {CNN_DATASET_DIR}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")
    print(f"Using device: {DEVICE.upper()}")

    # 3. Load Data
    if not os.path.exists(TRAIN_METADATA_PATH):
        print(f"FATAL ERROR: Metadata file not found at {TRAIN_METADATA_PATH}")
        sys.exit(1)
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    print(f"Loaded {len(train_df)} samples for final training.")
    
    # 4. Data Transforms and Loaders
    if args.in_channels == 1:
        NORM_MEAN = [0.5]
        NORM_STD = [0.5]
    elif args.in_channels == 2:
        NORM_MEAN = [0.0, 0.0]
        NORM_STD = [1.0, 1.0]
    elif args.in_channels == 3:
        NORM_MEAN = [0.5, 0.0, 0.0]
        NORM_STD = [0.5, 1.0, 1.0]
    else:
        raise ValueError(f"Unsupported number of channels: {args.in_channels}")

    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    train_dataset = AirflowSequenceDataset(
        train_df, 
        CNN_DATASET_DIR, 
        context_feature_cols=CONTEXT_FEATURES,
        dynamic_feature_cols=DYNAMIC_FEATURES,
        transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 5. Initialize Model, Loss, and Optimizer
    if args.model_type == 'lstm':
        model = UltimateHybridRegressor(
            num_context_features=len(CONTEXT_FEATURES),
            num_dynamic_features=len(DYNAMIC_FEATURES),
            cnn_in_channels=args.in_channels,
            lstm_hidden_size=512, # From Optuna
            lstm_layers=3         # From Optuna
        ).to(DEVICE)
        
        model.head[2] = torch.nn.Dropout(0.309) # From Optuna
    elif args.model_type == 'avg':
        model = SimplifiedCnnAvgRegressor(
            num_context_features=len(CONTEXT_FEATURES),
            num_dynamic_features=len(DYNAMIC_FEATURES),
            cnn_in_channels=args.in_channels
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    print(f"Initialized {type(model).__name__} with {args.in_channels} input channels.")
    
    criterion = nn.MSELoss()
    
    param_groups = [{'params': model.cnn[7].parameters(), 'lr': LEARNING_RATE / 10}]
    if args.model_type == 'lstm':
        main_model_params = itertools.chain(model.lstm.parameters(), model.attention.parameters())
        param_groups.append({'params': main_model_params, 'lr': LEARNING_RATE})
    shared_main_params = itertools.chain(model.context_mlp.parameters(), model.dynamic_mlp.parameters(), model.head.parameters())
    param_groups.append({'params': shared_main_params, 'lr': LEARNING_RATE})
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15)

    # 6. Training Loop
    for epoch in tqdm(range(NUM_EPOCHS), desc="Final Training", disable=(not is_interactive)):
        model.train()
        running_loss = 0.0
        
        for seq, context, dynamic, targets in train_loader:
            seq, context, dynamic, targets = seq.to(DEVICE), context.to(DEVICE), dynamic.to(DEVICE), targets.to(DEVICE)
            
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
            tqdm.write(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | LR: {current_lr:.1e}")

    # 7. Save the Final Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nFinal trained model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()