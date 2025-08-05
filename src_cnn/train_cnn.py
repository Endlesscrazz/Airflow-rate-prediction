# src_cnn/train_cnn.py
"""
Main script to train and evaluate the CNN+LSTM model for airflow prediction.
(Version for single-phase training with a frozen CNN backbone)
"""
import os
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
from tqdm import tqdm
import random
import argparse
import sys

# Import our custom modules
from src_cnn.cnn_utils import AirflowSequenceDataset
#from src_cnn.cnn_models import CnnLstmRegressor
from src_cnn.cnn_models import UltimateHybridRegressor

# --- Configuration ---
CNN_DATASET_DIR = "cnn_dataset/dataset_cnn-lstm-all-split-holes"
METADATA_PATH = os.path.join(CNN_DATASET_DIR, "metadata.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

# Training Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 3e-4  # A higher LR is suitable as we only train the LSTM/head
NUM_EPOCHS = 150
WEIGHT_DECAY = 1e-5

# Output Directories
MODEL_SAVE_DIR = "trained_models_hybrid_CV"
RESULTS_SAVE_DIR = "results_cnn_lstm_hybrid"

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

# --- Training and Evaluation Functions ---

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    # Unpack three inputs instead of two
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
        # Unpack three inputs instead of two
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
    # Argument Parser
    parser = argparse.ArgumentParser(description="Train CNN+LSTM model for a specific CV fold.")
    parser.add_argument("--fold", type=int, required=True, help="The fold number to run (0-indexed).")
    parser.add_argument("--total_folds", type=int, required=True, help="The total number of folds in the CV.")
    args = parser.parse_args()

    current_fold = args.fold
    n_splits = args.total_folds
    is_interactive = sys.stdout.isatty()

    print(f"--- Starting CNN+LSTM Model Training for FOLD {current_fold + 1}/{n_splits} ---")
    print(f"Using device: {DEVICE.upper()}")

    # 1. Load Data
    df_metadata = pd.read_csv(METADATA_PATH)
    X = df_metadata.drop("airflow_rate", axis=1)
    y = df_metadata["airflow_rate"]
    groups = df_metadata["video_id"]

    # 2. Setup Cross-Validation
    gkf = GroupKFold(n_splits=n_splits)
    train_idx, val_idx = list(gkf.split(X, y, groups))[current_fold]

    print(f"\n----- RUNNING FOLD {current_fold + 1}/{n_splits} -----")
    print(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}")

    train_df = df_metadata.iloc[train_idx]
    val_df = df_metadata.iloc[val_idx]

    # --- Data Transforms ---
    # Add augmentation for the training set to combat overfitting
    train_transform = transforms.Compose([
    # The dataset outputs tensors in [0, 1] range.
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    # Add jitter to simulate thermal variations
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # Add a bit of noise
    transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
    # Erase a random patch to force learning from context
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    # Final normalization for the model
    transforms.Normalize(mean=[0.5], std=[0.5])
])
    
    # Validation set only gets normalization
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create Datasets with appropriate transforms
    train_dataset = AirflowSequenceDataset(train_df, CNN_DATASET_DIR, 
                                           context_feature_cols=CONTEXT_FEATURES,
                                           dynamic_feature_cols=DYNAMIC_FEATURES,
                                           transform=train_transform)
    val_dataset = AirflowSequenceDataset(val_df, CNN_DATASET_DIR,
                                         context_feature_cols=CONTEXT_FEATURES,
                                         dynamic_feature_cols=DYNAMIC_FEATURES,
                                         transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = UltimateHybridRegressor(
        num_context_features=len(CONTEXT_FEATURES),
        num_dynamic_features=len(DYNAMIC_FEATURES)
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    # Group 1: The newly unfrozen CNN layers (fine-tuning with a small LR)
    cnn_finetune_params = model.cnn[7].parameters()

    # Group 2: All other trainable parts of the model (LSTM and MLPs)
    main_model_params = itertools.chain(
        model.lstm.parameters(),
        model.context_mlp.parameters(),
        model.dynamic_mlp.parameters(),
        model.head.parameters(),
        model.attention.parameters()
    )

    optimizer = optim.AdamW([
        {'params': cnn_finetune_params, 'lr': LEARNING_RATE / 10}, # e.g., 3e-5
        {'params': main_model_params, 'lr': LEARNING_RATE}         # e.g., 3e-4
    ], weight_decay=WEIGHT_DECAY)

    # --- END OF CHANGE ---

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
    best_val_r2 = -float('inf')
    model_save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_fold_{current_fold}.pth")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 4. Single, Continuous Training Loop
    for epoch in tqdm(range(NUM_EPOCHS), desc=f"Training Fold {current_fold+1}", disable=(not is_interactive)):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_r2, val_rmse = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
             current_lr = optimizer.param_groups[0]['lr']
             tqdm.write(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val R²: {val_r2:.4f} | LR: {current_lr:.1e}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), model_save_path)

    # 5. Report Final Results for this Fold
    print(f"\n----- Finished Fold {current_fold + 1} -----")
    print(f"Loading best model from: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    
    final_loss, final_r2, final_rmse = evaluate(model, val_loader, criterion, DEVICE)

    results_df = pd.DataFrame([{'fold': current_fold, 'r2': final_r2, 'rmse': final_rmse}])
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    results_csv_path = os.path.join(RESULTS_SAVE_DIR, f"fold_{current_fold}_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"Fold {current_fold + 1} Final R²: {final_r2:.4f}, RMSE: {final_rmse:.4f}")
    print(f"Results for this fold saved to: {results_csv_path}")

if __name__ == "__main__":
    main()