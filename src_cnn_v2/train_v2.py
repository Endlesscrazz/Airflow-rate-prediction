# src_cnn_v2/train_v2.py
"""
Main training script for the V2 (bottom-up) pipeline.
- Trains a model using the training set.
- Uses the validation set for early stopping and model selection.
- Saves the best model, scaler, and training logs.
- Implements Automatic Mixed Precision (AMP) for faster training.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import random
import joblib

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2.logging_utils_v2 import log_experiment_details
from src_cnn_v2.models_v2 import SimpleCropRegressor
from src_cnn_v2.dataset_utils_v2 import CroppedSequenceDataset
from src_cnn_v2 import config_v2 as cfg

# --- Seed everything for reproducibility ---
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(cfg.RANDOM_STATE)

# --- Corrected Asymmetric Loss Function ---
def asymmetric_loss(preds, targets, over_prediction_penalty=2.0):
    """
    A robust asymmetric SmoothL1 loss that penalizes over-predictions more heavily.
    """
    base_loss_fn = nn.SmoothL1Loss()
    
    over_mask = preds > targets
    under_mask = ~over_mask

    over_loss = base_loss_fn(preds[over_mask], targets[over_mask]) if over_mask.sum() > 0 else 0.0
    under_loss = base_loss_fn(preds[under_mask], targets[under_mask]) if under_mask.sum() > 0 else 0.0

    num_over = over_mask.sum()
    num_under = under_mask.sum()
    total = num_over + num_under
    
    if total == 0:
        return 0.0

    final_loss = (over_prediction_penalty * over_loss * num_over + under_loss * num_under) / total
    return final_loss

# --- Training and Evaluation Functions (with AMP) ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_targets_orig_scale, all_outputs_orig_scale = [], []

    for seq, delta_t, targets_scaled in dataloader:
        seq, delta_t, targets_scaled = seq.to(device), delta_t.to(device), targets_scaled.to(device)
        
        optimizer.zero_grad()

        # Use autocast for the forward pass (mixed precision)
        with torch.cuda.amp.autocast():
            outputs_scaled = model(seq, delta_t)
            loss = criterion(outputs_scaled, targets_scaled)

        # Scale the loss and call backward()
        scaler.scale(loss).backward()
        
        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        scaler.step(optimizer)

        # Update the scaler for the next iteration
        scaler.update()

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
            
            # Use autocast in evaluation for consistency and speed
            with torch.cuda.amp.autocast():
                outputs_scaled = model(seq, delta_t)

            all_targets.extend((targets_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())
            all_outputs.extend((outputs_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())

    all_outputs = np.array(all_outputs).clip(min=0)
    all_targets = np.array(all_targets)
    
    mae = np.mean(np.abs(all_targets - all_outputs))
    rmse = np.sqrt(np.mean((all_targets - all_outputs)**2))
    ss_res = np.sum((all_targets - all_outputs)**2)
    ss_tot = np.sum((all_targets - np.mean(all_targets))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return mae, rmse, r2

def main():
    # --- Setup Paths ---
    DATASET_DIR = cfg.DATASET_DIR
    TRAIN_METADATA_PATH = cfg.TRAIN_METADATA_PATH
    VAL_METADATA_PATH = cfg.VAL_METADATA_PATH
    RESULTS_DIR = cfg.EXPERIMENT_RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_save_path = os.path.join(RESULTS_DIR, "best_model_v2.pth")
    scaler_save_path = os.path.join(RESULTS_DIR, "scaler_v2.pkl")
    log_save_path = os.path.join(RESULTS_DIR, "training_log.csv")

    print(f"--- Starting V2 Model Training for Experiment: {cfg.EXPERIMENT_NAME} ---")
    print(f"  - Using data split from random seed: {cfg.RANDOM_STATE}")
    print(f"  - Normalizing with max flow rate: {cfg.MAX_FLOW_RATE}")
    print(f"  - All results will be saved to: {RESULTS_DIR}")

    try:
        train_df_orig = pd.read_csv(TRAIN_METADATA_PATH)
        val_df_orig = pd.read_csv(VAL_METADATA_PATH)
    except FileNotFoundError as e:
        sys.exit(f"FATAL: Metadata file not found. Error: {e}")

    print(f"Loaded {len(train_df_orig)} training and {len(val_df_orig)} validation samples.")

    train_df, val_df = train_df_orig.copy(), val_df_orig.copy()

    if cfg.ENABLE_PER_FOLD_SCALING:
        scaler_tabular = RobustScaler() if cfg.SCALER_KIND == "robust" else StandardScaler()
        train_df['delta_T'] = scaler_tabular.fit_transform(train_df[['delta_T']])
        val_df['delta_T'] = scaler_tabular.transform(val_df[['delta_T']])
        if cfg.SAVE_SCALERS:
            joblib.dump(scaler_tabular, scaler_save_path)
            print(f"  - Saved tabular scaler to: {scaler_save_path}")

    # --- Dataset and DataLoader ---
    # Normalization for image data is usually handled inside the dataset/model, but can be added here if needed
    train_dataset = CroppedSequenceDataset(train_df, DATASET_DIR, transform=None)
    val_dataset = CroppedSequenceDataset(val_df, DATASET_DIR, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model, Loss, Optimizer, Scheduler ---
    model = SimpleCropRegressor(
        lstm_hidden_size=cfg.INITIAL_PARAMS['lstm_hidden_size'],
        lstm_layers=cfg.INITIAL_PARAMS['lstm_layers'],
        dropout=cfg.INITIAL_PARAMS['dropout_rate']
    ).to(cfg.DEVICE)

    # Configure overpred weight
    over_pred_penalty = 1.7   #hardyboard
    criterion = lambda preds, targets: asymmetric_loss(preds, targets, over_prediction_penalty=over_pred_penalty)

    optimizer_name = cfg.INITIAL_PARAMS.get('optimizer', 'AdamW')
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.INITIAL_PARAMS['lr'], weight_decay=cfg.INITIAL_PARAMS['weight_decay'])
        print("  - Using Adam optimizer.")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.INITIAL_PARAMS['lr'], weight_decay=cfg.INITIAL_PARAMS['weight_decay'])
        print("  - Using AdamW optimizer.")
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
    
    # Initialize the GradScaler for mixed-precision training
    scaler_amp = torch.amp.GradScaler('cuda')

    # --- Training Loop ---
    history = []
    best_val_mae = float('inf')
    epochs_no_improve = 0
    patience = 25 # Early stopping patience

    for epoch in range(cfg.NUM_EPOCHS):
        train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE, scaler_amp)
        val_mae, val_rmse, val_r2 = evaluate(model, val_loader, cfg.DEVICE)
        
        scheduler.step(val_mae) # Step scheduler based on validation MAE
        
        print(f"Epoch {epoch+1:03d}/{cfg.NUM_EPOCHS} | Train Loss: {train_loss:.5f} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | Val R²: {val_r2:.4f}")
        
        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_mae': train_mae,
            'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2
        })
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
            print(f"  - Validation MAE improved to {best_val_mae:.4f}. Saving model.")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break

    print("\n--- Training Complete ---")

    history_df = pd.DataFrame(history)
    history_df.to_csv(log_save_path, index=False)
    print(f"Training log saved to: {log_save_path}")
    print(f"Best model (Val MAE: {best_val_mae:.4f}) saved to: {model_save_path}")

    # --- Log Final Details ---
    log_filepath = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "experiment_summary.txt")
    final_training_params = {
        "Experiment Name": cfg.EXPERIMENT_NAME,
        "Batch Size": cfg.BATCH_SIZE,
        "Over-pred-penalt": over_pred_penalty,
        "Number of Epochs Run": len(history_df),
        "Best Model Found at Epoch": history_df['val_mae'].idxmin() + 1,
        "Best Validation MAE": best_val_mae,
        "Final Hyperparameters Used": cfg.INITIAL_PARAMS
    }
    log_experiment_details(log_filepath, "Final Model Training Parameters", final_training_params)

if __name__ == "__main__":
    main()

# python src_cnn_v2/train_v2.py