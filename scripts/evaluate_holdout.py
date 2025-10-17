# scripts/evaluate_holdout.py
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse
import joblib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn import config as cfg
from src_cnn.cnn_models import UltimateHybridRegressor
from src_cnn.cnn_utils import AirflowSequenceDataset

def get_predictions(model, dataloader, device):
    model.to(device)
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for seq, context, dynamic, _ in tqdm(dataloader, desc="Predicting", leave=False):
            seq, context, dynamic = seq.to(device), context.to(device), dynamic.to(device)
            outputs = model(seq, context, dynamic)
            all_outputs.extend(outputs.cpu().numpy())
    return np.array(all_outputs)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a final model or ensemble on the hold-out dataset.")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'avg'])
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--in_channels", type=int, required=True)
    parser.add_argument("--optuna_tuned", action='store_true')
    parser.add_argument("--ensemble", action='store_true')
    args = parser.parse_args()

    CNN_DATASET_DIR = args.dataset_dir
    HOLDOUT_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "holdout_metadata.csv")
    model_name_tag = f"{args.model_type}_{args.in_channels}ch"
    if args.optuna_tuned: model_name_tag += "_optuna"
    if cfg.ENABLE_PER_FOLD_SCALING: model_name_tag += f"_{cfg.SCALER_KIND}scaled"

    print(f"--- Evaluating on Hold-Out Set ---")
    print(f"Mode: {'Ensemble' if args.ensemble else 'Single Model'}")
    print(f"Model Type: {args.model_type.upper()} | Channels: {args.in_channels} | Tuned: {args.optuna_tuned}")
    print(f"Dataset: {CNN_DATASET_DIR}")

    if not os.path.isfile(HOLDOUT_METADATA_PATH):
        sys.exit(f"FATAL ERROR: Hold-out metadata not found at '{HOLDOUT_METADATA_PATH}'")
        
    holdout_df_raw = pd.read_csv(HOLDOUT_METADATA_PATH)
    all_targets = holdout_df_raw['airflow_rate'].values

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
            df = pd.concat([df, pd.get_dummies(df['material'], prefix='material', dtype=int)], axis=1)
        return df

    holdout_df = prepare_features(holdout_df_raw)
    
    all_context_features = cfg.CONTEXT_FEATURES + [f"material_{m}" for m in cfg.ALL_POSSIBLE_MATERIALS]
    available_context_features = [col for col in all_context_features if col in holdout_df.columns]
    available_dynamic_features = [col for col in cfg.DYNAMIC_FEATURES if col in holdout_df.columns]

    norm_params = cfg.NORM_CONSTANTS.get(args.in_channels)
    val_transform = transforms.Compose([transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])

    all_outputs = None
    if not args.ensemble:
        MODEL_SAVE_DIR = os.path.join(project_root, "trained_models_final")
        MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"final_model_{model_name_tag}.pth")
        SCALER_PATH = os.path.join(MODEL_SAVE_DIR, f"final_scaler_{model_name_tag}.pkl")
        print(f"Loading single model from: {MODEL_PATH}")
        
        df_to_predict = holdout_df.copy()
        if cfg.ENABLE_PER_FOLD_SCALING:
            scaler = joblib.load(SCALER_PATH)
            numeric_cols_to_scale = [f for f in available_context_features + available_dynamic_features if not f.startswith('material_')]
            df_to_predict[numeric_cols_to_scale] = scaler.transform(df_to_predict[numeric_cols_to_scale])

        holdout_dataset = AirflowSequenceDataset(df_to_predict, CNN_DATASET_DIR, available_context_features, available_dynamic_features, val_transform)
        holdout_loader = DataLoader(holdout_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
        
        model = UltimateHybridRegressor(
            num_context_features=len(available_context_features),
            num_dynamic_features=len(available_dynamic_features),
            cnn_in_channels=args.in_channels,
            lstm_hidden_size=cfg.OPTUNA_PARAMS['lstm_hidden_size'],
            lstm_layers=cfg.OPTUNA_PARAMS['lstm_layers']
        )
        model.head[2] = torch.nn.Dropout(cfg.OPTUNA_PARAMS['dropout_rate'])
        model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.DEVICE))
        all_outputs = get_predictions(model, holdout_loader, cfg.DEVICE)
    else:
        CV_MODEL_DIR = os.path.join(project_root, f"trained_models_{model_name_tag}_CV")
        print(f"Loading fold models from: {CV_MODEL_DIR}")

        fold_predictions = []
        for fold in range(cfg.CV_FOLDS):
            print(f"-- Processing Fold {fold} --")
            MODEL_PATH = os.path.join(CV_MODEL_DIR, f"best_model_fold_{fold}.pth")
            if not os.path.exists(MODEL_PATH):
                print(f"Warning: Model for fold {fold} not found. Skipping.")
                continue

            fold_holdout_df = holdout_df.copy()
            if cfg.ENABLE_PER_FOLD_SCALING:
                SCALER_PATH = os.path.join(CV_MODEL_DIR, "scalers", f"scaler_fold_{fold}.pkl")
                scaler = joblib.load(SCALER_PATH)
                numeric_cols_to_scale = [f for f in available_context_features + available_dynamic_features if not f.startswith('material_')]
                
                scaler_feature_names = scaler.get_feature_names_out()
                cols_to_transform_in_holdout = [col for col in numeric_cols_to_scale if col in scaler_feature_names]
                
                fold_holdout_df.loc[:, cols_to_transform_in_holdout] = scaler.transform(fold_holdout_df[cols_to_transform_in_holdout])

            holdout_dataset = AirflowSequenceDataset(fold_holdout_df, CNN_DATASET_DIR, available_context_features, available_dynamic_features, val_transform)
            holdout_loader = DataLoader(holdout_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

            model = UltimateHybridRegressor(
                num_context_features=len(available_context_features),
                num_dynamic_features=len(available_dynamic_features),
                cnn_in_channels=args.in_channels,
                lstm_hidden_size=cfg.OPTUNA_PARAMS['lstm_hidden_size'],
                lstm_layers=cfg.OPTUNA_PARAMS['lstm_layers']
            )
            model.head[2] = torch.nn.Dropout(cfg.OPTUNA_PARAMS['dropout_rate'])
            model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.DEVICE))

            preds = get_predictions(model, holdout_loader, cfg.DEVICE)
            fold_predictions.append(preds)

        if not fold_predictions:
            sys.exit("FATAL ERROR: No fold models found for ensembling.")

        all_outputs = np.mean(fold_predictions, axis=0)

    all_outputs = np.expm1(all_outputs)
    r2 = r2_score(all_targets, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
    mae = mean_absolute_error(all_targets, all_outputs)

    print("\n--- Hold-Out Set Performance ---")
    print(f"R-squared (R²):               {r2:.4f}")
    print(f"Mean Absolute Error (MAE):    {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_outputs, alpha=0.6, edgecolors='k')
    min_val, max_val = min(all_targets.min(), all_outputs.min()) * 0.9, max(all_targets.max(), all_outputs.max()) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("True Airflow Rate (L/min)")
    plt.ylabel("Predicted Airflow Rate (L/min)")
    plot_save_dir = os.path.join(project_root, "holdout_results")
    os.makedirs(plot_save_dir, exist_ok=True)

    if args.ensemble:
        title = (f"Hold-Out Set (Ensemble): True vs. Predicted\n"
                 f"R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")
        plot_save_path = os.path.join(plot_save_dir, f"holdout_plot_{model_name_tag}_ensemble.png")
    else:
        title = (f"Hold-Out Set (Single Model): True vs. Predicted\n"
                 f"R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")
        plot_save_path = os.path.join(plot_save_dir, f"holdout_plot_{model_name_tag}.png")
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.savefig(plot_save_path)
    print(f"\nSaved results plot to: {plot_save_path}")

if __name__ == "__main__":
    main()

"""

ensmeble:
python -m scripts.evaluate_holdout --model_type lstm \
    --dataset_dir CNN_dataset/gypusm_8_hole_dataset/dataset_2ch_thermal_masked \
    --in_channels 2 \
    --optuna_tuned \
    --ensemble

single-final-model:
python -m scripts.evaluate_holdout --model_type lstm \
    --dataset_dir CNN_dataset/gypusm_8_hole_dataset/dataset_2ch_thermal_masked \
    --in_channels 2 \
    --optuna_tuned 



"""
