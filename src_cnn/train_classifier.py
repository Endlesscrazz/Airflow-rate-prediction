# src_cnn/train_classifier.py
"""
Trains and evaluates a CNN+LSTM CLASSIFIER model for airflow prediction.
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import random
import argparse
import sys

# Import our custom modules
from src_cnn.cnn_utils import AirflowSequenceDataset
from src_cnn.cnn_models import UltimateHybridClassifier

# --- Configuration ---
CNN_DATASET_DIR = "cnn_dataset/dataset_cnn_lstm_flow"
METADATA_PATH = os.path.join(CNN_DATASET_DIR, "metadata.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42
NUM_CLASSES = 3

# Training Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 150
WEIGHT_DECAY = 1e-5

# Output Directories
MODEL_SAVE_DIR = "trained_models_classifier_CV"
RESULTS_SAVE_DIR = "results_classifier"

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
    for seq, context, dynamic, targets in dataloader:
        # For classification, target needs to be Long
        seq, context, dynamic, targets = seq.to(device), context.to(device), dynamic.to(device), targets.to(device).long()
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
    all_targets, all_preds = [], []
    with torch.no_grad():
        for seq, context, dynamic, targets in dataloader:
            seq, context, dynamic, targets = seq.to(device), context.to(device), dynamic.to(device), targets.to(device).long()
            outputs = model(seq, context, dynamic)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * seq.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    return epoch_loss, accuracy, f1

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Train CNN+LSTM classifier for a specific CV fold.")
    parser.add_argument("--fold", type=int, required=True, help="The fold number to run (0-indexed).")
    parser.add_argument("--total_folds", type=int, required=True, help="The total number of folds in the CV.")
    args = parser.parse_args()
    current_fold = args.fold
    n_splits = args.total_folds
    is_interactive = sys.stdout.isatty()

    print(f"--- Starting CNN+LSTM Classifier Training for FOLD {current_fold + 1}/{n_splits} ---")
    print(f"Using device: {DEVICE.upper()}")

    df_metadata = pd.read_csv(METADATA_PATH)
    bins = [0, 1.7, 2.3, float('inf')]
    labels = [0, 1, 2]
    y_stratify = pd.cut(df_metadata['airflow_rate'], bins=bins, labels=labels, right=False).fillna(0).astype(int)
    X = df_metadata
    groups = df_metadata["video_id"]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = list(sgkf.split(X, y_stratify, groups))[current_fold]

    print(f"\n----- RUNNING FOLD {current_fold + 1}/{n_splits} -----")
    print(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}")
    train_df = df_metadata.iloc[train_idx]
    val_df = df_metadata.iloc[val_idx]

    # --- START OF FIX ---
    FLOW_MEAN = [0.0, 0.0]
    FLOW_STD = [1.0, 1.0]
    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=FLOW_MEAN, std=FLOW_STD)
    ])
    val_transform = transforms.Compose([transforms.Normalize(mean=FLOW_MEAN, std=FLOW_STD)])
    # --- END OF FIX ---

    train_dataset = AirflowSequenceDataset(train_df, CNN_DATASET_DIR, 
                                           context_feature_cols=CONTEXT_FEATURES,
                                           dynamic_feature_cols=DYNAMIC_FEATURES,
                                           transform=train_transform,
                                           task='classification') # <-- Pass the new flag
    val_dataset = AirflowSequenceDataset(val_df, CNN_DATASET_DIR,
                                         context_feature_cols=CONTEXT_FEATURES,
                                         dynamic_feature_cols=DYNAMIC_FEATURES,
                                         transform=val_transform,
                                         task='classification')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = UltimateHybridClassifier(num_context_features=len(CONTEXT_FEATURES), num_dynamic_features=len(DYNAMIC_FEATURES), num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    cnn_finetune_params = model.cnn[7].parameters()
    main_model_params = itertools.chain(model.lstm.parameters(), model.context_mlp.parameters(), model.dynamic_mlp.parameters(), model.head.parameters(), model.attention.parameters())
    optimizer = optim.AdamW([{'params': cnn_finetune_params, 'lr': LEARNING_RATE / 10}, {'params': main_model_params, 'lr': LEARNING_RATE}], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
    
    model_save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_fold_{current_fold}.pth")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    best_val_f1 = -float('inf')
    for epoch in tqdm(range(NUM_EPOCHS), desc=f"Training Fold {current_fold+1}", disable=(not is_interactive)):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
             current_lr = optimizer.param_groups[1]['lr']
             tqdm.write(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.1e}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)

    print(f"\n----- Finished Fold {current_fold + 1} -----")
    print(f"Loading best model from: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    
    final_loss, final_accuracy, final_f1 = evaluate(model, val_loader, criterion, DEVICE)
    results_df = pd.DataFrame([{'fold': current_fold, 'accuracy': final_accuracy, 'f1_score': final_f1}])
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    results_csv_path = os.path.join(RESULTS_SAVE_DIR, f"fold_{current_fold}_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"Fold {current_fold + 1} Final Accuracy: {final_accuracy:.4f}, F1-score: {final_f1:.4f}")
    print(f"Results for this fold saved to: {results_csv_path}")

if __name__ == "__main__":
    main()