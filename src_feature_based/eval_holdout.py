# scripts/evaluate_feature_model.py
"""
Evaluates the final feature-based model on the unseen hold-out set.
Calculates R-squared, MAE, and RMSE.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # <-- Import MAE
import matplotlib.pyplot as plt
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_feature_based import config as cfg
from src_feature_based import plotting


def main():
    print("--- Evaluating Final Feature-Based Model on Hold-Out Set ---")
    
    parser = argparse.ArgumentParser(description="Evaluate the final feature-based model.")
    parser.add_argument("best_model_name", type=str, 
                        help="The name of the best model to evaluate (e.g., 'MLPRegressor').")
    args = parser.parse_args()
    
    # 1. Load Hold-Out Data and Final Model
    holdout_path = os.path.join(cfg.OUTPUT_DIR, "holdout_features.csv")
    model_dir = os.path.join(cfg.OUTPUT_DIR, "trained_cv_model")
    model_path = os.path.join(model_dir, f"final_model_{args.best_model_name}.joblib")

    
    if not (os.path.exists(holdout_path) and os.path.exists(model_path)):
        print("FATAL ERROR: Holdout data or final model not found. Please run training script first.")
        sys.exit(1)
        
    holdout_df = pd.read_csv(holdout_path)
    final_model = joblib.load(model_path)
    
    y_true = holdout_df['airflow_rate']
    X_raw_holdout = holdout_df.drop(columns=['airflow_rate', 'video_id'])
    
    print("\nTransforming raw holdout features...")
    X_transformed_holdout = pd.DataFrame(index=X_raw_holdout.index)

    if 'delta_T' in X_raw_holdout.columns:
        X_transformed_holdout['delta_T_log'] = np.log1p(X_raw_holdout['delta_T'])
    if 'hotspot_area' in X_raw_holdout.columns and cfg.LOG_TRANSFORM_AREA:
        X_transformed_holdout['hotspot_area_log'] = np.log1p(X_raw_holdout['hotspot_area'])
    if 'hotspot_avg_temp_change_rate_initial' in X_raw_holdout.columns and cfg.NORMALIZE_AVG_RATE_INITIAL:
        X_transformed_holdout['hotspot_avg_temp_change_rate_initial_norm'] = X_raw_holdout.apply(
            lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and pd.notna(r['hotspot_avg_temp_change_rate_initial']) else np.nan, axis=1
        )
    
    special_raw = ['delta_T', 'hotspot_area', 'hotspot_avg_temp_change_rate_initial', 'material']
    other_features = [col for col in X_raw_holdout.columns if col not in special_raw]
    X_transformed_holdout = pd.concat([X_transformed_holdout, X_raw_holdout[other_features]], axis=1)
    
    X_transformed_holdout = pd.concat([X_transformed_holdout, pd.get_dummies(X_raw_holdout['material'], prefix='material', dtype=int)], axis=1)

    # Reorder columns to match the training order to prevent warnings
    X_holdout = X_transformed_holdout.reindex(columns=cfg.SELECTED_FEATURES)

    # 2. Make Predictions
    y_pred_scaled = final_model.predict(X_holdout)
    
    # 3. Inverse Transform Predictions if scaling was used
    y_pred = y_pred_scaled 
    if cfg.ENABLE_TARGET_SCALING:
        print("\nTarget scaling was enabled. Inverse transforming predictions...")
        scaler_path = os.path.join(model_dir, f"final_target_scaler_{args.best_model_name}.joblib")
        if not os.path.exists(scaler_path):
            print(f"FATAL ERROR: Target scaler not found at {scaler_path}")
            sys.exit(1)
        y_scaler = joblib.load(scaler_path)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 4. --- MODIFIED: Calculate and Print All Metrics ---
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print("\n--- Hold-Out Set Performance ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} L/min")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} L/min")

    # 5. --- MODIFIED: Plotting with All Metrics in Title ---
    plots_dir = os.path.join(cfg.OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_path = os.path.join(plots_dir, f"holdout_performance_plot_{args.best_model_name}.png")
    plot_title = (
        f"Hold-Out Set Performance for {args.best_model_name}\n"
        f"R² = {r2:.4f}  |  MAE = {mae:.4f} L/min  |  RMSE = {rmse:.4f} L/min"
    )

    plotting.plot_actual_vs_predicted(
        y_true.to_numpy(), y_pred, plot_title, plot_path, material_labels=holdout_df['material']
    )
    print(f"\nSaved performance plot to: {plot_path}")

if __name__ == "__main__":
    main()
    
# python -m src_feature_based.eval_holdout MLPRegressor