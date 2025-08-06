# src_cnn/evaluate_holdout_classifier.py
"""
Evaluates the final trained classifier model on the hold-out set.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from src_cnn.cnn_utils import AirflowSequenceDataset
from src_cnn.cnn_models import UltimateHybridClassifier

# --- Configuration ---
CNN_DATASET_DIR = "cnn_dataset/dataset_cnn_lstm_flow"
HOLDOUT_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "holdout_metadata.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3
MODEL_PATH = "trained_models_classifier_final/final_classifier_model.pth"
RESULTS_DIR = "results_classifier"

# --- START OF FIX ---
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
# --- END OF FIX ---

def main():
    print("--- Evaluating Final Classifier on Hold-Out Set ---")
    
    holdout_df = pd.read_csv(HOLDOUT_METADATA_PATH)
    
    bins = [0, 1.7, 2.3, float('inf')]
    class_names = ["Low (<1.7)", "Medium (1.7-2.3)", "High (>2.3)"]
    holdout_df['airflow_category'] = pd.cut(holdout_df['airflow_rate'], bins=bins, labels=range(len(class_names)), right=False).fillna(0).astype(int)

    print(f"Loaded {len(holdout_df)} samples from the hold-out set.")
    
    FLOW_MEAN = [0.0, 0.0]
    FLOW_STD = [1.0, 1.0]
    val_transform = transforms.Compose([transforms.Normalize(mean=FLOW_MEAN, std=FLOW_STD)])
    
    # --- START OF FIX 2 ---
    # Pass the task='classification' flag to the dataset
    holdout_dataset = AirflowSequenceDataset(
        holdout_df, 
        CNN_DATASET_DIR, 
        context_feature_cols=CONTEXT_FEATURES, 
        dynamic_feature_cols=DYNAMIC_FEATURES, 
        transform=val_transform,
        task='classification'
    )
    # --- END OF FIX 2 ---
    
    holdout_loader = DataLoader(holdout_dataset, batch_size=8, shuffle=False)

    model = UltimateHybridClassifier(
        num_context_features=len(CONTEXT_FEATURES), 
        num_dynamic_features=len(DYNAMIC_FEATURES), 
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_targets, all_preds = [], []
    with torch.no_grad():
        # --- START OF FIX 3 ---
        # The dataloader now provides the correct targets
        for seq, context, dynamic, targets in holdout_loader:
            seq, context, dynamic = seq.to(DEVICE), context.to(DEVICE), dynamic.to(DEVICE)
            outputs = model(seq, context, dynamic)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy()) # Get targets from loader
    # --- END OF FIX 3 ---

    # We get all_targets from the loader now, not the dataframe
    # all_targets = holdout_df['airflow_category'].values 

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')

    print("\n--- Hold-Out Set Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Hold-Out Set Confusion Matrix')
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm_path = os.path.join(RESULTS_DIR, "holdout_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    main()

# python -m src_cnn.evaluate_holdout_classifier