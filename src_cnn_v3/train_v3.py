# src_cnn_v3/train_v3.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v3 import config_v3 as cfg
from src_cnn_v3.dataset_utils_v3 import HybridDataset
from src_cnn_v3.models_v3 import HybridCropRegressor
from src_cnn_v3.logging_utils_v3 import log_section

def asymmetric_loss(preds, targets, over_prediction_penalty=1.5):
    # Both preds and targets are in [0, 1] range here
    diff = preds - targets
    loss = torch.where(torch.abs(diff) < 1, 0.5 * diff ** 2, torch.abs(diff) - 0.5)
    weights = torch.where(diff > 0, over_prediction_penalty, 1.0)
    return (loss * weights).mean()

def save_checkpoint(model, scaler, epoch, val_mae, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'val_mae': val_mae,
        'config': cfg.INITIAL_PARAMS
    }, path)

def train_one_epoch(model, loader, optimizer, scaler_amp, device):
    model.train()
    total_loss = 0
    all_preds_real, all_targets_real = [], [] # For MAE calculation
    
    for video, features, target in loader:
        video, features, target = video.to(device), features.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            preds = model(video, features)
            loss = asymmetric_loss(preds, target) # Loss in [0,1] space
            
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        
        total_loss += loss.item() * video.size(0)
        
        # Unscale for metrics (Real L/min)
        real_preds = (preds * cfg.MAX_FLOW_RATE).detach().cpu().numpy()
        real_targets = (target * cfg.MAX_FLOW_RATE).cpu().numpy()
        all_preds_real.extend(real_preds)
        all_targets_real.extend(real_targets)
        
    avg_loss = total_loss / len(loader.dataset)
    # Calculate MAE in real units
    mae = np.mean(np.abs(np.array(all_targets_real) - np.array(all_preds_real)))
    return avg_loss, mae

def validate(model, loader, device):
    model.eval()
    all_preds_real, all_targets_real = [], []
    
    with torch.no_grad():
        for video, features, target in loader:
            video, features = video.to(device), features.to(device)
            
            with torch.cuda.amp.autocast():
                preds = model(video, features)
                preds = torch.clamp(preds, 0, 1) # Valid range check
            
            real_preds = (preds * cfg.MAX_FLOW_RATE).cpu().numpy()
            real_targets = (target * cfg.MAX_FLOW_RATE).numpy()
            
            all_preds_real.extend(real_preds)
            all_targets_real.extend(real_targets)
            
    all_preds = np.array(all_preds_real)
    all_targets = np.array(all_targets_real)
    
    mae = np.mean(np.abs(all_targets - all_preds))
    return mae

def main():
    print(f"--- V3 Training: {cfg.EXPERIMENT_NAME} ---")
    print(f"Max Flow Rate for Scaling: {cfg.MAX_FLOW_RATE}")
    print(f"Hyperparams: {cfg.INITIAL_PARAMS}")
    
    os.makedirs(cfg.EXPERIMENT_RESULTS_DIR, exist_ok=True)
    
    # Load Data
    df_train = pd.read_csv(os.path.join(cfg.DATASET_DIR, "train_metadata.csv"))
    df_val = pd.read_csv(os.path.join(cfg.DATASET_DIR, "val_metadata.csv"))
    
    # Feature Scaler
    feat_cols = ['delta_T']
    if cfg.USE_HANDCRAFTED_FEATURES:
        feat_cols += ['feat_area', 'feat_aspect', 'feat_extent']
    
    print(f"Fitting Scaler on features: {feat_cols}")
    scaler = StandardScaler()
    scaler.fit(df_train[feat_cols].values)
    joblib.dump(scaler, os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "feature_scaler.pkl"))
    
    # Datasets
    train_ds = HybridDataset(df_train, cfg.DATASET_DIR, scaler)
    val_ds = HybridDataset(df_val, cfg.DATASET_DIR, scaler)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model Setup
    model = HybridCropRegressor(
        num_tabular_features=len(feat_cols),
        lstm_hidden=cfg.INITIAL_PARAMS['lstm_hidden_size'],
        lstm_layers=cfg.INITIAL_PARAMS['lstm_layers'],
        dropout=cfg.INITIAL_PARAMS['dropout_rate']
    ).to(cfg.DEVICE)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.INITIAL_PARAMS['lr'], 
        weight_decay=cfg.INITIAL_PARAMS['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    scaler_amp = torch.cuda.amp.GradScaler()
    
    # Training Loop
    best_val_mae = float('inf')
    patience = 20
    epochs_no_improve = 10
    history = []
    
    print("Starting training...")
    for epoch in range(100):
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, scaler_amp, cfg.DEVICE)
        val_mae = validate(model, val_loader, cfg.DEVICE)
        
        scheduler.step(val_mae)
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.5f} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | LR: {lr:.1e}")
        
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'train_mae': train_mae, 'val_mae': val_mae})
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            save_path = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "best_model.pth")
            save_checkpoint(model, scaler, epoch, val_mae, save_path)
            print(f"  --> Best Model Saved! ({val_mae:.4f})")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print("Early stopping.")
            break
            
    pd.DataFrame(history).to_csv(os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "training_log.csv"), index=False)
    
    log_data = {
        "Best MAE": best_val_mae, 
        "Hyperparams": cfg.INITIAL_PARAMS,
        "Max Flow": cfg.MAX_FLOW_RATE
    }
    log_section("Model Training (V3.1)", log_data)

if __name__ == "__main__":
    main()