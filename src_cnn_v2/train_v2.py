# src_cnn_v2/train_v2.py
"""
Main training script for the V2 (bottom-up) pipeline.
Supports Ensemble Training via CLI arguments.
Settings: Raw Delta T, No Train Clamp, Eval Clamp, Penalty 1.5.
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
import argparse 

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

def asymmetric_loss(preds, targets, over_prediction_penalty=1.3):
    base_loss_fn = nn.SmoothL1Loss()
    
    over_mask = preds > targets
    under_mask = ~over_mask

    if over_mask.sum() > 0:
        over_loss = base_loss_fn(preds[over_mask], targets[over_mask])
    else:
        over_loss = 0.0

    if under_mask.sum() > 0:
        under_loss = base_loss_fn(preds[under_mask], targets[under_mask])
    else:
        under_loss = 0.0

    return over_prediction_penalty * over_loss + under_loss

# --- Training and Evaluation Functions (with AMP) ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_targets_orig_scale, all_outputs_orig_scale = [], []

    for seq, delta_t, targets_scaled in dataloader:
        seq, delta_t, targets_scaled = seq.to(device), delta_t.to(device), targets_scaled.to(device)
        
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs_scaled = model(seq, delta_t)
            
            loss = criterion(outputs_scaled, targets_scaled)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
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
            
            with torch.amp.autocast('cuda'):
                outputs_scaled = model(seq, delta_t)
                
                # We clamp for validation metrics to match the test-time logic.
                outputs_scaled = outputs_scaled.clamp(0, 1)

            all_targets.extend((targets_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())
            all_outputs.extend((outputs_scaled * cfg.MAX_FLOW_RATE).cpu().numpy())

    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    
    mae = np.mean(np.abs(all_targets - all_outputs))
    rmse = np.sqrt(np.mean((all_targets - all_outputs)**2))
    ss_res = np.sum((all_targets - all_outputs)**2)
    ss_tot = np.sum((all_targets - np.mean(all_targets))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return mae, rmse, r2

def main():
    # --- ARGUMENT PARSING FOR ENSEMBLE ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE, 
                        help="Random seed for model initialization (default: config seed)")
    args = parser.parse_args()

    # 1. Set Training Seed (Controls Weights/Batching)
    print(f"--- Starting V2 Model Training ---")
    print(f"  - Model Init Seed: {args.seed}")
    seed_everything(args.seed)

    # 2. Set Data Paths (Always use Config Seed 42 for Splits)
    # This ensures we train on the exact same data regardless of the model seed
    DATASET_DIR = cfg.DATASET_DIR
    TRAIN_METADATA_PATH = cfg.TRAIN_METADATA_PATH 
    VAL_METADATA_PATH = cfg.VAL_METADATA_PATH
    
    # 3. Create Unique Results Directory per Seed
    # Appends _SEED_XX to the folder name
    RESULTS_DIR = f"{cfg.EXPERIMENT_RESULTS_DIR}_SEED_{args.seed}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    model_save_path = os.path.join(RESULTS_DIR, "best_model_v2.pth")
    log_save_path = os.path.join(RESULTS_DIR, "training_log.csv")

    print(f"  - Data Split: Seed {cfg.RANDOM_STATE} (Fixed)")
    print(f"  - Saving Results to: {RESULTS_DIR}")

    # --- Load Data ---
    try:
        train_df = pd.read_csv(TRAIN_METADATA_PATH)
        val_df = pd.read_csv(VAL_METADATA_PATH)
    except FileNotFoundError as e:
        sys.exit(f"FATAL: Metadata file not found. Error: {e}")

    print(f"Loaded {len(train_df)} training and {len(val_df)} validation samples.")

    # --- Dataset and DataLoader ---
    train_dataset = CroppedSequenceDataset(train_df, DATASET_DIR, transform=None)
    val_dataset = CroppedSequenceDataset(val_df, DATASET_DIR, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = SimpleCropRegressor(
        lstm_hidden_size=cfg.INITIAL_PARAMS['lstm_hidden_size'],
        lstm_layers=cfg.INITIAL_PARAMS['lstm_layers'],
        dropout=cfg.INITIAL_PARAMS['dropout_rate']
    ).to(cfg.DEVICE)

    #over_pred_penalty = 2.5 #hardybaord maksym
    #over_pred_penalty = 2.1 #brickcladding maksym
    over_pred_penalty = 1.7 #gypsum
    criterion = lambda preds, targets: asymmetric_loss(preds, targets, over_prediction_penalty=over_pred_penalty)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.INITIAL_PARAMS['lr'], weight_decay=cfg.INITIAL_PARAMS['weight_decay'])
    
    # Scheduler: Patience=5 (Matches Colleague)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)
    
    scaler_amp = torch.amp.GradScaler('cuda')

    # --- Training Loop ---
    history = []
    best_val_mae = float('inf')
    epochs_no_improve = 0
    patience = 50 # Early stopping patience

    for epoch in range(cfg.NUM_EPOCHS):
        train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE, scaler_amp)
        val_mae, val_rmse, val_r2 = evaluate(model, val_loader, cfg.DEVICE)
        
        scheduler.step(val_mae)
        
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
    log_filepath = os.path.join(RESULTS_DIR, "experiment_summary.txt")
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