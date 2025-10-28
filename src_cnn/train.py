# scripts/train.py
import os
import sys
import numpy as np
import itertools
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
    for seq, context, dynamic, targets in dataloader:
        seq, context, dynamic, targets = seq.to(device), context.to(device), dynamic.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(seq, context, dynamic)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * seq.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for seq, context, dynamic, targets in dataloader:
            seq, context, dynamic, targets = seq.to(device), context.to(device), dynamic.to(device), targets.to(device)
            outputs = model(seq, context, dynamic)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * seq.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    all_outputs, all_targets = np.expm1(all_outputs), np.expm1(all_targets)
    r2 = r2_score(all_targets, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
    mae = mean_absolute_error(all_targets, all_outputs)
    return epoch_loss, rmse, mae, r2 

def main():
    parser = argparse.ArgumentParser(description="Run cross-validation for a specific model.")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--total_folds", type=int, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'avg'])
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--in_channels", type=int, required=True)
    args = parser.parse_args()

    is_interactive = sys.stdout.isatty()

    CNN_DATASET_DIR = args.dataset_dir
    METADATA_PATH = os.path.join(CNN_DATASET_DIR, "train_metadata.csv") 
    model_name_tag = f"{args.model_type}_{args.in_channels}ch_optuna"

    if cfg.ENABLE_PER_FOLD_SCALING: 
        model_name_tag += f"_{cfg.SCALER_KIND}scaled"

    MODEL_SAVE_DIR = os.path.join(project_root, f"trained_models_{model_name_tag}_CV")
    RESULTS_SAVE_DIR = os.path.join(project_root, f"results_{model_name_tag}_CV")
    SCALER_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "scalers")

    print(f"--- Starting CNN-based Model Training (Cross-Validation) ---")
    print(f"Model Type: {args.model_type.upper()} | Fold: {args.fold + 1}/{args.total_folds}")

    df_metadata = pd.read_csv(METADATA_PATH)
    
    df_metadata['base_video_id'] = df_metadata['video_id'].apply(lambda x: x.split('_hole_')[0])
    groups = df_metadata["base_video_id"]

    gkf = GroupKFold(n_splits=args.total_folds)
    
    train_idx, val_idx = list(gkf.split(df_metadata, df_metadata["airflow_rate"], groups))[args.fold]
    train_df_raw, val_df_raw = df_metadata.iloc[train_idx].copy(), df_metadata.iloc[val_idx].copy()

    print(f"\n----- RUNNING FOLD {args.fold + 1}/{args.total_folds} -----")
    print(f"Train set size: {len(train_df_raw)}, Validation set size: {len(val_df_raw)}")

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
    val_df = prepare_features(val_df_raw)

    # --- DYNAMIC FEATURE DEFINITION (THE FIX) ---
    base_context_features = [f for f in cfg.CONTEXT_FEATURES if f in train_df.columns]
    discovered_material_features = [col for col in train_df.columns if col.startswith('material_')]
    available_context_features = base_context_features + discovered_material_features
    available_dynamic_features = [col for col in cfg.DYNAMIC_FEATURES if col in train_df.columns]
    # --- END FIX ---
    
    print(f"\nUsing {len(available_context_features)} context features: {available_context_features}")
    print(f"Using {len(available_dynamic_features)} dynamic features: {available_dynamic_features}")

    if cfg.ENABLE_PER_FOLD_SCALING:
        print(f"\nApplying per-fold scaling with '{cfg.SCALER_KIND}' scaler...")
        numeric_cols_to_scale = [f for f in available_context_features + available_dynamic_features if not f.startswith('material_')]
        
        if cfg.SCALER_KIND == "robust": scaler = RobustScaler()
        else: scaler = StandardScaler()

        # Use .loc to avoid SettingWithCopyWarning
        train_df.loc[:, numeric_cols_to_scale] = scaler.fit_transform(train_df[numeric_cols_to_scale])
        val_df.loc[:, numeric_cols_to_scale] = scaler.transform(val_df[numeric_cols_to_scale])
        
        if cfg.SAVE_SCALERS:
            os.makedirs(SCALER_SAVE_DIR, exist_ok=True)
            joblib.dump(scaler, os.path.join(SCALER_SAVE_DIR, f"scaler_fold_{args.fold}.pkl"))
            print(f"Saved fitted scaler for fold {args.fold} to {os.path.join(SCALER_SAVE_DIR, f'scaler_fold_{args.fold}.pkl')}")

    norm_params = cfg.NORM_CONSTANTS.get(args.in_channels)
    train_transform = transforms.Compose([transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)), transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)), transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    val_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])
    
    train_dataset = AirflowSequenceDataset(train_df, CNN_DATASET_DIR, context_feature_cols=available_context_features, dynamic_feature_cols=available_dynamic_features, transform=train_transform, is_train=True)
    val_dataset = AirflowSequenceDataset(val_df, CNN_DATASET_DIR, context_feature_cols=available_context_features, dynamic_feature_cols=available_dynamic_features, transform=val_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    if args.model_type != 'lstm':
        raise ValueError(f"Model type '{args.model_type}' is not fully supported. Please use 'lstm'.")

    model = UltimateHybridRegressor(
        num_context_features=len(available_context_features),
        num_dynamic_features=len(available_dynamic_features),
        cnn_in_channels=args.in_channels,
        lstm_hidden_size=cfg.OPTUNA_PARAMS['lstm_hidden_size'],
        lstm_layers=cfg.OPTUNA_PARAMS['lstm_layers']
    ).to(cfg.DEVICE)
    model.head[2] = torch.nn.Dropout(cfg.OPTUNA_PARAMS['dropout_rate'])
    print(f"Initialized {type(model).__name__} with {args.in_channels} input channels and Optuna-tuned parameters.")
    
    criterion = nn.MSELoss()
    param_groups = [{'params': model.cnn[7].parameters(), 'lr': cfg.OPTUNA_PARAMS['lr'] / 10}]
    main_model_params = itertools.chain(model.lstm.parameters(), model.attention.parameters())
    param_groups.append({'params': main_model_params, 'lr': cfg.OPTUNA_PARAMS['lr']})
    shared_main_params = itertools.chain(model.context_mlp.parameters(), model.dynamic_mlp.parameters(), model.head.parameters())
    param_groups.append({'params': shared_main_params, 'lr': cfg.OPTUNA_PARAMS['lr']})
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.OPTUNA_PARAMS['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
    
    model_save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_fold_{args.fold}.pth")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    patience = 25
    epochs_no_improve = 0
    best_val_rmse = float('inf') 
    
    for epoch in tqdm(range(cfg.NUM_EPOCHS_CV), desc=f"Training Fold {args.fold+1}", disable=(not is_interactive)):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_loss, val_rmse, val_mae, val_r2 = evaluate(model, val_loader, criterion, cfg.DEVICE)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
             current_lr = optimizer.param_groups[-1]['lr']
             tqdm.write(f"Epoch {epoch+1:03d}/{cfg.NUM_EPOCHS_CV} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} | Val R²: {val_r2:.4f} | LR: {current_lr:.1e}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n--- Early stopping triggered at epoch {epoch + 1} after {patience} epochs with no improvement. ---")
            break

    print(f"\n----- Finished Fold {args.fold + 1} -----")
    print(f"Loading best model from: {model_save_path} (Best Val RMSE: {best_val_rmse:.4f})")
    model.load_state_dict(torch.load(model_save_path))
    
    final_loss, final_rmse, final_mae, final_r2 = evaluate(model, val_loader, criterion, cfg.DEVICE)

    results_df = pd.DataFrame([{'fold': args.fold, 'rmse': final_rmse, 'mae': final_mae, 'r2': final_r2}])
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    results_csv_path = os.path.join(RESULTS_SAVE_DIR, f"fold_{args.fold}_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"Fold {args.fold + 1} Final RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}, R²: {final_r2:.4f}")
    print(f"Results for this fold saved to: {results_csv_path}")

if __name__ == "__main__":
    main()