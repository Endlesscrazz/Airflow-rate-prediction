# src_cnn_v2/train_v2.py
"""
Main training script for the V2 (bottom-up) pipeline.
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

from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.dataset_utils_v2 import CroppedSequenceDataset
from src_cnn_v2.models_v2 import SimpleCropRegressor

# --- Seed everything for reproducibility ---
torch.manual_seed(cfg.RANDOM_STATE)
np.random.seed(cfg.RANDOM_STATE)
random.seed(cfg.RANDOM_STATE)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for seq, delta_t, targets in dataloader:
        seq, delta_t, targets = seq.to(device), delta_t.to(device), targets.to(device)
        log_targets = torch.log1p(targets)
        optimizer.zero_grad()
        outputs = model(seq, delta_t)
        loss = criterion(outputs, log_targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * seq.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for seq, delta_t, targets in dataloader:
            seq, delta_t = seq.to(device), delta_t.to(device)
            outputs = model(seq, delta_t)
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(torch.expm1(outputs).cpu().numpy())
            
    all_outputs = np.array(all_outputs).clip(min=0) # Ensure no negative predictions
    all_targets = np.array(all_targets)
    
    mae = np.mean(np.abs(all_targets - all_outputs))
    rmse = np.sqrt(np.mean((all_targets - all_outputs)**2))
    
    ss_res = np.sum((all_targets - all_outputs)**2)
    ss_tot = np.sum((all_targets - np.mean(all_targets))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return mae, rmse, r2

def test_final_model(model, dataloader, device):
    """Evaluates the model and returns predictions and targets."""
    model.eval()
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for seq, delta_t, targets in dataloader:
            seq, delta_t = seq.to(device), delta_t.to(device)
            outputs = model(seq, delta_t)
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(torch.expm1(outputs).cpu().numpy())

    all_outputs = np.array(all_outputs).clip(min=0)
    all_targets = np.array(all_targets)
    return all_targets, all_outputs

def main():
    # --- Setup Paths ---
    CNN_DATASET_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.V2_DATASET_PARAMS["OUTPUT_SUBDIR"])
    TRAIN_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "train_metadata_v2.csv")
    VAL_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "val_metadata_v2.csv")
    TEST_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "test_metadata_v2.csv") # Uses the split file
    
    MODEL_SAVE_DIR = os.path.join(cfg.OUTPUT_DIR, "trained_models")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_SAVE_DIR, "best_model_v2.pth")
    log_save_path = os.path.join(MODEL_SAVE_DIR, "training_log.csv")
    test_preds_save_path = os.path.join(MODEL_SAVE_DIR, "test_predictions.csv")

    print("--- Starting V2 Model Training ---")
    
    try:
        train_df_orig = pd.read_csv(TRAIN_METADATA_PATH)
        val_df_orig = pd.read_csv(VAL_METADATA_PATH)
        test_df_orig = pd.read_csv(TEST_METADATA_PATH)
    except FileNotFoundError as e:
        sys.exit(f"FATAL: Metadata file not found. Have you run split_data_v2.py and create_dataset_v2.py? Error: {e}")
    
    print(f"Loaded {len(train_df_orig)} training, {len(val_df_orig)} validation, and {len(test_df_orig)} test samples.")

    train_df, val_df, test_df = train_df_orig.copy(), val_df_orig.copy(), test_df_orig.copy()

    if cfg.ENABLE_PER_FOLD_SCALING:
        scaler = RobustScaler() if cfg.SCALER_KIND == "robust" else StandardScaler()
        train_df['delta_T'] = scaler.fit_transform(train_df[['delta_T']])
        val_df['delta_T'] = scaler.transform(val_df[['delta_T']])
        test_df['delta_T'] = scaler.transform(test_df[['delta_T']])
        if cfg.SAVE_SCALERS:
            joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler_v2.pkl"))

    norm_params = cfg.NORM_CONSTANTS[1]
    train_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    val_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    
    train_dataset = CroppedSequenceDataset(train_df, CNN_DATASET_DIR, transform=train_transform)
    val_dataset = CroppedSequenceDataset(val_df, CNN_DATASET_DIR, transform=val_transform)
    # The test dataset uses the same class but doesn't apply augmentation.
    test_dataset = CroppedSequenceDataset(test_df, CNN_DATASET_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleCropRegressor().to(cfg.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.INITIAL_PARAMS['lr'], weight_decay=cfg.INITIAL_PARAMS['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
    
    history = []
    best_val_mae = float('inf')
    epochs_no_improve = 0
    patience = 25
    
    for epoch in range(cfg.NUM_EPOCHS_CV):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_mae, val_rmse, val_r2 = evaluate(model, val_loader, cfg.DEVICE)
        scheduler.step(train_loss)

        print(f"Epoch {epoch+1:03d}/{cfg.NUM_EPOCHS_CV} | Train Loss: {train_loss:.5f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")
        
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break
            
    # --- Final Actions ---
    print("\n--- Training Complete ---")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(log_save_path, index=False)
    print(f"Training log saved to: {log_save_path}")

    # Load best model and evaluate on the test set
    print(f"Loading best model from epoch {history_df['val_mae'].idxmin() + 1} to evaluate on test set...")
    model.load_state_dict(torch.load(model_save_path))
    
    test_targets, test_predictions = test_final_model(model, test_loader, cfg.DEVICE)
    
    test_mae = np.mean(np.abs(test_targets - test_predictions))
    test_rmse = np.sqrt(np.mean((test_targets - test_predictions)**2))
    ss_res = np.sum((test_targets - test_predictions)**2)
    ss_tot = np.sum((test_targets - np.mean(test_targets))**2)
    test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    print("\n--- FINAL TEST SET PERFORMANCE ---")
    print(f"Test MAE:   {test_mae:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"Test R²:    {test_r2:.4f}")

    # Save test predictions for plotting
    test_results_df = pd.DataFrame({'true_airflow': test_targets, 'predicted_airflow': test_predictions})
    test_results_df.to_csv(test_preds_save_path, index=False)
    print(f"Test predictions saved to: {test_preds_save_path}")

if __name__ == "__main__":
    main()

# python src_cnn_v2/train_v2.py