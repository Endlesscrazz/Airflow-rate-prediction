# scripts/train_final_model.py
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import random
import argparse
import joblib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn.cnn_utils import AirflowSequenceDataset
from src_cnn.cnn_models import UltimateHybridRegressor
from src_cnn import config as cfg

torch.manual_seed(cfg.RANDOM_STATE)
np.random.seed(cfg.RANDOM_STATE)
random.seed(cfg.RANDOM_STATE)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Train a final model on the full development dataset.")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'avg'])
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--in_channels", type=int, required=True)
    args = parser.parse_args()

    is_interactive = sys.stdout.isatty()

    CNN_DATASET_DIR = args.dataset_dir
    TRAIN_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "train_metadata.csv") 
    
    model_name_tag = f"{args.model_type}_{args.in_channels}ch_optuna"
    if cfg.ENABLE_PER_FOLD_SCALING: model_name_tag += f"_{cfg.SCALER_KIND}scaled"
        
    MODEL_SAVE_DIR = os.path.join(project_root, "trained_models_final")
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"final_model_{model_name_tag}.pth")
    SCALER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"final_scaler_{model_name_tag}.pkl")

    print(f"--- Starting FINAL Model Training (Optuna-tuned) ---")
    print(f"Model Type: {args.model_type.upper()} | Channels: {args.in_channels}")
    print(f"Dataset: {CNN_DATASET_DIR}")

    if not os.path.exists(TRAIN_METADATA_PATH):
        sys.exit(f"FATAL ERROR: train_metadata.csv not found at {TRAIN_METADATA_PATH}")
        
    train_df_raw = pd.read_csv(TRAIN_METADATA_PATH)
    print(f"Loaded {len(train_df_raw)} samples for final training.")

    # --- UNIFIED FEATURE PREPARATION (Identical to train.py) ---
    def prepare_features(df_in):
        df = df_in.copy()
        if 'delta_T' in df.columns: df['delta_T_log'] = np.log1p(df['delta_T'])
        if cfg.LOG_TRANSFORM_AREA and 'hotspot_area' in df.columns: df['hotspot_area_log'] = np.log1p(df['hotspot_area'])
        if cfg.NORMALIZE_AVG_RATE_INITIAL and 'hotspot_avg_temp_change_rate_initial' in df.columns:
            df['hotspot_avg_temp_change_rate_initial_norm'] = df.apply(lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r.get('delta_T', 0) != 0 else 0, axis=1)
        if cfg.NORMALIZE_CUMULATIVE_FEATURES:
            features_to_normalize = ['cumulative_raw_delta_sum', 'cumulative_abs_delta_sum', 'auc_mean_temp_delta', 'mean_pixel_volatility']
            for feature in features_to_normalize:
                if feature in df.columns:
                    df[f"{feature}_norm"] = df.apply(lambda r: r[feature] / r['delta_T'] if r.get('delta_T', 0) != 0 and pd.notna(r.get(feature)) else np.nan, axis=1)
        if 'material' in df.columns:
            # Use the definitive list from config to ensure all dummies are created
            df['material'] = pd.Categorical(df['material'], categories=cfg.ALL_POSSIBLE_MATERIALS)
            df = pd.concat([df, pd.get_dummies(df['material'], prefix='material', dtype=int)], axis=1)
        return df

    train_df = prepare_features(train_df_raw)

    # --- DYNAMIC FEATURE DEFINITION (THE FIX) ---
    base_context_features = [f for f in cfg.CONTEXT_FEATURES if f in train_df.columns]
    discovered_material_features = [col for col in train_df.columns if col.startswith('material_')]
    available_context_features = base_context_features + discovered_material_features
    available_dynamic_features = [col for col in cfg.DYNAMIC_FEATURES if col in train_df.columns]
    # --- END FIX ---
    
    print(f"\nUsing {len(available_context_features)} context features: {available_context_features}")
    print(f"Using {len(available_dynamic_features)} dynamic features: {available_dynamic_features}")

    if cfg.ENABLE_PER_FOLD_SCALING:
        print(f"\nApplying scaling with '{cfg.SCALER_KIND}' scaler to the full development set...")
        numeric_cols_to_scale = [f for f in available_context_features + available_dynamic_features if not f.startswith('material_')]
        
        if cfg.SCALER_KIND == "robust": scaler = RobustScaler()
        else: scaler = StandardScaler()

        train_df.loc[:, numeric_cols_to_scale] = scaler.fit_transform(train_df[numeric_cols_to_scale])
        
        if cfg.SAVE_SCALERS:
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            joblib.dump(scaler, SCALER_SAVE_PATH)
            print(f"Saved final fitted scaler to {SCALER_SAVE_PATH}")
    
    norm_params = cfg.NORM_CONSTANTS.get(args.in_channels)
    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])
    ])
    
    train_dataset = AirflowSequenceDataset(train_df, CNN_DATASET_DIR, context_feature_cols=available_context_features, dynamic_feature_cols=available_dynamic_features, transform=train_transform, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    if args.model_type != 'lstm':
        raise ValueError(f"Model type '{args.model_type}' is not supported.")
        
    model = UltimateHybridRegressor(
        num_context_features=len(available_context_features),
        num_dynamic_features=len(available_dynamic_features),
        cnn_in_channels=args.in_channels,
        lstm_hidden_size=cfg.OPTUNA_PARAMS['lstm_hidden_size'],
        lstm_layers=cfg.OPTUNA_PARAMS['lstm_layers']
    ).to(cfg.DEVICE)
    model.head[2] = torch.nn.Dropout(cfg.OPTUNA_PARAMS['dropout_rate'])

    print(f"Initialized {type(model).__name__} with Optuna-tuned parameters.")
    
    criterion = nn.MSELoss()
    param_groups = [{'params': model.cnn[7].parameters(), 'lr': cfg.OPTUNA_PARAMS['lr'] / 10}]
    main_model_params = itertools.chain(model.lstm.parameters(), model.attention.parameters())
    param_groups.append({'params': main_model_params, 'lr': cfg.OPTUNA_PARAMS['lr']})
    shared_main_params = itertools.chain(model.context_mlp.parameters(), model.dynamic_mlp.parameters(), model.head.parameters())
    param_groups.append({'params': shared_main_params, 'lr': cfg.OPTUNA_PARAMS['lr']})
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.OPTUNA_PARAMS['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15)

    for epoch in tqdm(range(cfg.NUM_EPOCHS_FINAL), desc="Final Training", disable=(not is_interactive)):
        model.train()
        running_loss = 0.0
        for seq, context, dynamic, targets in train_loader:
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

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nFinal trained model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()