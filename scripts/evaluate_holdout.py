# scripts/evaluate_holdout.py
"""
Evaluates a final trained model OR an ensemble of CV models on the unseen hold-out set.
"""
from src_cnn import config as cfg
from src_cnn.cnn_models import UltimateHybridRegressor, SimplifiedCnnAvgRegressor
from src_cnn.cnn_utils import AirflowSequenceDataset
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


def get_predictions(model, dataloader, device):
    """Helper function to get model predictions for a given dataloader."""
    model.to(device)
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for seq, context, dynamic, _ in tqdm(dataloader, desc="Predicting", leave=False):
            seq, context, dynamic = seq.to(
                device), context.to(device), dynamic.to(device)
            outputs = model(seq, context, dynamic)
            all_outputs.extend(outputs.cpu().numpy())
    return np.array(all_outputs)


def main():
    # 1. Setup & Parse Arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a final model or ensemble on the hold-out dataset.")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=['lstm', 'avg'], help="The type of model architecture.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--in_channels", type=int, required=True,
                        help="Number of input channels for the CNN.")
    parser.add_argument("--optuna_tuned", action='store_true',
                        help="Specify if the models were tuned with Optuna.")
    parser.add_argument("--ensemble", action='store_true',
                        help="Enable ensemble evaluation mode.")
    args = parser.parse_args()

    # 2. Configure Paths and Logging
    CNN_DATASET_DIR = args.dataset_dir
    HOLDOUT_METADATA_PATH = os.path.join(
        CNN_DATASET_DIR, "holdout_metadata.csv")

    model_name_tag = f"{args.model_type}_{args.in_channels}ch"
    if args.optuna_tuned:
        model_name_tag += "_optuna"
    if cfg.ENABLE_PER_FOLD_SCALING:
        model_name_tag += f"_{cfg.SCALER_KIND}scaled"

    print(f"--- Evaluating on Hold-Out Set ---")
    print(f"Mode: {'Ensemble' if args.ensemble else 'Single Model'}")
    print(
        f"Model Type: {args.model_type.upper()} | Channels: {args.in_channels} | Tuned: {args.optuna_tuned}")
    print(f"Dataset: {CNN_DATASET_DIR}")

    # 3. Load Hold-Out Data
    if not os.path.isfile(HOLDOUT_METADATA_PATH):
        print(
            f"FATAL ERROR: Hold-out metadata not found at '{HOLDOUT_METADATA_PATH}'")
        sys.exit(1)
    holdout_df = pd.read_csv(HOLDOUT_METADATA_PATH)
    all_targets = holdout_df['airflow_rate'].values

    available_context_features = [
        col for col in cfg.CONTEXT_FEATURES if col in holdout_df.columns]
    available_dynamic_features = [
        col for col in cfg.DYNAMIC_FEATURES if col in holdout_df.columns]

    norm_params = cfg.NORM_CONSTANTS.get(args.in_channels)
    if not norm_params:
        raise ValueError(
            f"Normalization constants not defined for {args.in_channels} channels.")
    val_transform = transforms.Compose(
        [transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])])

    # --- 4. Prediction Logic: Single Model vs. Ensemble ---
    if not args.ensemble:
        # SINGLE MODEL EVALUATION
        MODEL_PATH = f"trained_models_final/final_model_{model_name_tag}.pth"
        SCALER_PATH = f"trained_models_final/final_scaler_{model_name_tag}.pkl"
        print(f"Loading single model from: {MODEL_PATH}")

        if cfg.ENABLE_PER_FOLD_SCALING:
            print(
                f"\nApplying saved '{cfg.SCALER_KIND}' scaler to hold-out data...")
            scaler = joblib.load(SCALER_PATH)
            numeric_cols = [col for col in (available_context_features + available_dynamic_features)
                            if col in holdout_df.columns and pd.api.types.is_numeric_dtype(holdout_df[col])]
            holdout_df[numeric_cols] = scaler.transform(
                holdout_df[numeric_cols])

        holdout_dataset = AirflowSequenceDataset(
            holdout_df, CNN_DATASET_DIR, available_context_features, available_dynamic_features, val_transform)
        holdout_loader = DataLoader(
            holdout_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

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
        # ENSEMBLE EVALUATION
        CV_MODEL_DIR = f"trained_models_{model_name_tag}_CV"
        print(f"Loading fold models from: {CV_MODEL_DIR}")

        fold_predictions = []
        for fold in range(cfg.CV_FOLDS):
            print(f"-- Processing Fold {fold} --")
            MODEL_PATH = os.path.join(
                CV_MODEL_DIR, f"best_model_fold_{fold}.pth")
            if not os.path.exists(MODEL_PATH):
                print(
                    f"Warning: Model for fold {fold} not found at {MODEL_PATH}. Skipping.")
                continue

            fold_holdout_df = holdout_df.copy()
            if cfg.ENABLE_PER_FOLD_SCALING:
                SCALER_PATH = os.path.join(
                    CV_MODEL_DIR, "scalers", f"scaler_fold_{fold}.pkl")
                scaler = joblib.load(SCALER_PATH)
                numeric_cols = [col for col in (available_context_features + available_dynamic_features)
                                if col in fold_holdout_df.columns and pd.api.types.is_numeric_dtype(fold_holdout_df[col])]
                fold_holdout_df[numeric_cols] = scaler.transform(
                    fold_holdout_df[numeric_cols])

            holdout_dataset = AirflowSequenceDataset(
                fold_holdout_df, CNN_DATASET_DIR, available_context_features, available_dynamic_features, val_transform)
            holdout_loader = DataLoader(
                holdout_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

            model = UltimateHybridRegressor(
                num_context_features=len(available_context_features),
                num_dynamic_features=len(available_dynamic_features),
                cnn_in_channels=args.in_channels,
                lstm_hidden_size=cfg.OPTUNA_PARAMS['lstm_hidden_size'],
                lstm_layers=cfg.OPTUNA_PARAMS['lstm_layers']
            )
            model.head[2] = torch.nn.Dropout(cfg.OPTUNA_PARAMS['dropout_rate'])
            model.load_state_dict(torch.load(
                MODEL_PATH, map_location=cfg.DEVICE))

            preds = get_predictions(model, holdout_loader, cfg.DEVICE)
            fold_predictions.append(preds)

        if not fold_predictions:
            print("FATAL ERROR: No fold models found for ensembling. Aborting.")
            sys.exit(1)

        print(
            f"\nEnsembling predictions from {len(fold_predictions)} folds...")
        all_outputs = np.mean(fold_predictions, axis=0)

    # Inverse transform the model's predictions from log-space to the original airflow scale
    all_outputs = np.expm1(all_outputs)

    # 5. Calculate and Print Metrics
    r2 = r2_score(all_targets, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
    mae = mean_absolute_error(all_targets, all_outputs)

    print("\n--- Hold-Out Set Performance ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):    {mae:.4f}")
    print(f"R-squared (R²):               {r2:.4f}")

    # 6. Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_outputs, alpha=0.6, edgecolors='k')
    min_val = min(min(all_targets), min(all_outputs)) * 0.95
    max_val = max(max(all_targets), max(all_outputs)) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("True Airflow Rate")
    plt.ylabel("Predicted Airflow Rate")

    plot_save_dir = "holdout_results"
    os.makedirs(plot_save_dir, exist_ok=True)

    if args.ensemble:
        plt.title(
            f"Hold-Out Set (Ensemble): True vs. Predicted \n"
            f"R² = {r2:.4f}  |  MAE = {mae:.4f} L/min  |  RMSE = {rmse:.4f} L/min")
        plot_save_path = os.path.join(
            plot_save_dir, f"holdout_plot_{model_name_tag}_ensemble.png")
    else:
        plt.title(
            f"Hold-Out Set (Single Model): True vs. Predicted \n"
            f"R² = {r2:.4f}  |  MAE = {mae:.4f} L/min  |  RMSE = {rmse:.4f} L/min")
        plot_save_path = os.path.join(
            plot_save_dir, f"holdout_plot_{model_name_tag}.png")

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
single-final-model:
python -m scripts.evaluate_holdout --model_type lstm \
    --dataset_dir CNN_dataset/dataset_2ch_thermal_masked \
    --in_channels 2 \
    --optuna_tuned 

python -m scripts.evaluate_holdout --model_type lstm \
    --dataset_dir CNN_dataset/dataset_2ch_thermal_masked \
    --in_channels 2 \
    --optuna_tuned \
    --ensemble

"""
