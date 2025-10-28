# src_cnn/train_final_classifier.py
"""
Trains the final classifier model on the entire development set.
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

from src_cnn.cnn_utils import AirflowSequenceDataset
from src_cnn.cnn_models import UltimateHybridClassifier

# --- Configuration (Unchanged) ---
CNN_DATASET_DIR = "cnn_dataset/dataset_cnn_lstm_flow"
TRAIN_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "train_metadata.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42
NUM_CLASSES = 3
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 150
WEIGHT_DECAY = 1e-5
MODEL_SAVE_PATH = "trained_models_classifier_final/final_classifier_model.pth"

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

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def main():
    print("--- Starting FINAL Classifier Model Training on ALL Development Data ---")
    print(f"Using device: {DEVICE.upper()}")
    
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    print(f"Loaded {len(train_df)} samples for final training.")
    
    FLOW_MEAN = [0.0, 0.0]
    FLOW_STD = [1.0, 1.0]
    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=FLOW_MEAN, std=FLOW_STD)
    ])
    
    # --- START OF FIX ---

    # Let the Dataset class handle the label creation.
    train_dataset = AirflowSequenceDataset(
        train_df, 
        CNN_DATASET_DIR, 
        context_feature_cols=CONTEXT_FEATURES,
        dynamic_feature_cols=DYNAMIC_FEATURES,
        transform=train_transform,
        task='classification'  # <-- This tells the Dataset to yield class labels
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = UltimateHybridClassifier(
        num_context_features=len(CONTEXT_FEATURES),
        num_dynamic_features=len(DYNAMIC_FEATURES),
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    cnn_finetune_params = model.cnn[7].parameters()
    main_model_params = itertools.chain(model.lstm.parameters(), model.attention.parameters(), model.context_mlp.parameters(), model.dynamic_mlp.parameters(), model.head.parameters())
    optimizer = optim.AdamW([{'params': cnn_finetune_params, 'lr': LEARNING_RATE / 10}, {'params': main_model_params, 'lr': LEARNING_RATE}], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15)

    is_interactive = sys.stdout.isatty()

    for epoch in tqdm(range(NUM_EPOCHS), desc="Final Training", disable=(not is_interactive)):
        model.train()
        running_loss = 0.0
        
        # The loop is now simple and correct. 'targets' will be the class labels.
        for seq, context, dynamic, targets in train_loader:
            seq, context, dynamic, targets = seq.to(DEVICE), context.to(DEVICE), dynamic.to(DEVICE), targets.to(DEVICE).long()
            
            optimizer.zero_grad()
            outputs = model(seq, context, dynamic)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * seq.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step(epoch_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[1]['lr']
            tqdm.write(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | LR: {current_lr:.1e}")

    # --- END OF FIX ---

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nFinal trained classifier saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()