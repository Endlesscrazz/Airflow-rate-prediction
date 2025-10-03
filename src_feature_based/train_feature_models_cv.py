# scripts/train_feature_models_cv.py
"""
Performs a full cross-validation workflow for feature-based models.
1. Loads pre-computed features and transforms them.
2. Applies optional data augmentation and target transformation.
3. Runs GridSearchCV with GroupKFold for multiple model types.
4. Saves the best overall model for evaluation.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.base import clone
import warnings
from sklearn.exceptions import ConvergenceWarning

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_feature_based import config as cfg
from src_feature_based import modeling
from src_feature_based import plotting
from src_feature_based.utils import setup_logging, log_experiment_configs

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def augment_features(X_df, y_series, groups_series, num_copies=2, noise_level=0.02):
    """
    Augments features by adding Gaussian noise.
    """
    if num_copies == 0:
        return X_df, y_series, groups_series

    augmented_X_list, augmented_y_list, augmented_groups_list = [], [], []
    numeric_cols = [col for col in X_df.columns if not col.startswith('material_')]

    for _ in range(num_copies):
        X_aug = X_df.copy()
        for col in numeric_cols:
            noise = np.random.normal(0, noise_level, size=len(X_df)) * X_df[col]
            X_aug[col] += noise
        augmented_X_list.append(X_aug)
        augmented_y_list.append(y_series)
        augmented_groups_list.append(groups_series)

    X_final = pd.concat([X_df] + augmented_X_list, ignore_index=True)
    y_final = pd.concat([y_series] + augmented_y_list, ignore_index=True)
    groups_final = pd.concat([groups_series] + augmented_groups_list, ignore_index=True)
    
    return X_final, y_final, groups_final

def main():
    setup_logging(output_dir=cfg.OUTPUT_DIR, script_name="train_cv")
    
    print("--- Training and Tuning Handcrafted Feature-Based Models ---")
    
    # 1. Load Data
    train_path = os.path.join(cfg.OUTPUT_DIR, "train_features.csv")
    if not os.path.exists(train_path):
        print(f"FATAL ERROR: {train_path} not found. Run generate_features.py and split_data.py first.")
        sys.exit(1)
        
    train_df = pd.read_csv(train_path)
    
    y_dev_original = train_df['airflow_rate']
    groups_dev = train_df['video_id'].apply(lambda x: x.split('_hole_')[0])
    X_raw = train_df.drop(columns=['airflow_rate', 'video_id'])

    # 2. Transform Features
    print("\nTransforming raw features...")
    X_transformed = pd.DataFrame(index=X_raw.index)
    if 'delta_T' in X_raw.columns:
        X_transformed['delta_T_log'] = np.log1p(X_raw['delta_T'])
    if 'hotspot_area' in X_raw.columns and cfg.LOG_TRANSFORM_AREA:
        X_transformed['hotspot_area_log'] = np.log1p(X_raw['hotspot_area'])
    if 'hotspot_avg_temp_change_rate_initial' in X_raw.columns and cfg.NORMALIZE_AVG_RATE_INITIAL:
        X_transformed['hotspot_avg_temp_change_rate_initial_norm'] = X_raw.apply(
            lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and pd.notna(r['hotspot_avg_temp_change_rate_initial']) else np.nan, axis=1
        )

    special_raw = ['delta_T', 'hotspot_area', 'hotspot_avg_temp_change_rate_initial', 'material', 'session']
    other_features = [col for col in X_raw.columns if col not in special_raw]
    X_transformed = pd.concat([X_transformed, X_raw[other_features]], axis=1)

    X_transformed = pd.concat([X_transformed, pd.get_dummies(X_raw['material'], prefix='material', dtype=int)], axis=1)
    X_transformed = pd.concat([X_transformed, pd.get_dummies(X_raw['session'], prefix='session', dtype=int)], axis=1)
    
    # Reindex to ensure consistent column order, adding missing columns if necessary
    print("\nCreating interaction features...")
    features_to_interact = [
        'mean_area_significant_change',
        'hotspot_area_log',
        'hotspot_avg_temp_change_rate_initial_norm',
        'mean_gradient_at_edge', 'temperature_kurtosis', 'temp_std_avg_initial',
        'peak_to_average_ratio', 'circularity'
    ]
    context_cols = [col for col in X_transformed.columns if col.startswith('material_') or col.startswith('session_')]
    
    for feature in features_to_interact:
        if feature in X_transformed.columns:
            for context_col in context_cols:
                interaction_col_name = f"{feature}_x_{context_col}"
                X_transformed[interaction_col_name] = X_transformed[feature] * X_transformed[context_col]

    
    # The SELECTED_FEATURES list in config.py should now include the interaction features you want to use
    X_dev = X_transformed.reindex(columns=cfg.SELECTED_FEATURES, fill_value=0)
    print(f"\nUsing {len(X_dev.columns)} selected features for training.")
    log_experiment_configs(selected_features=X_dev.columns.tolist())

    # 3. Augment Data (Optional)
    X_dev_final, y_dev_final, groups_dev_final = augment_features(
        X_dev, y_dev_original, groups_dev,
        num_copies=1,       # to augment data
        noise_level=0.03
    )
    print(f"\nAugmented training set from {len(X_dev)} to {len(X_dev_final)} samples.")
    
    # 4. Transform Target (Optional)
    y_to_train = y_dev_final.copy() # Default to original values
    if cfg.ENABLE_TARGET_SCALING:
        print(f"\nTarget transformation enabled. Applying log1p transformation.")
        y_to_train = np.log1p(y_to_train)
    
    # 5. Run Hyperparameter Tuning via Cross-Validation
    scoring_metrics = {
        'r2': 'r2',
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_root_mean_squared_error': 'neg_root_mean_squared_error'
    }
    
    cv_splitter = GroupKFold(n_splits=cfg.CV_FOLDS)
    models, param_grids = modeling.get_models_and_grids()
    best_estimators = {}
    all_cv_results = pd.DataFrame(columns=['R2', 'MAE (log)', 'RMSE (log)'])

    for model_name, model in models.items():
        print(f"\n--- Tuning {model_name} ---")
        
        pipeline = modeling.build_pipeline(model)
        param_grid = param_grids.get(model_name, {})
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring=scoring_metrics,
            refit='r2',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        grid_search.fit(X_dev_final, y_to_train, groups=groups_dev_final)
        
        best_index = grid_search.best_index_
        results = grid_search.cv_results_
        
        best_r2 = results['mean_test_r2'][best_index]
        best_mae_log = -results['mean_test_neg_mean_absolute_error'][best_index]
        best_rmse_log = -results['mean_test_neg_root_mean_squared_error'][best_index]

        print(f"Best CV Scores for {model_name} (on transformed target):")
        print(f"  - R²:   {best_r2:.4f}")
        print(f"  - MAE:  {best_mae_log:.4f} (log-space)")
        print(f"  - RMSE: {best_rmse_log:.4f} (log-space)")
        print(f"Best parameters: {grid_search.best_params_}")
        
        best_estimators[model_name] = grid_search.best_estimator_
        all_cv_results.loc[model_name] = [best_r2, best_mae_log, best_rmse_log]

    print("\n--- Cross-Validation Summary (Scores on Transformed Target) ---")
    print(all_cv_results.sort_values(by='R2', ascending=False))
    
    best_model_name = all_cv_results['R2'].idxmax()
    final_model_pipeline = best_estimators[best_model_name]
    
    print(f"\n--- Best model from CV is: {best_model_name} (Avg R² = {all_cv_results.loc[best_model_name, 'R2']:.4f}) ---")

    # 6. Save Best Model
    model_save_dir = os.path.join(cfg.OUTPUT_DIR, "trained_cv_model")
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_filename = f"final_model_{best_model_name}.joblib"
    model_save_path = os.path.join(model_save_dir, model_filename)
    joblib.dump(final_model_pipeline, model_save_path)
    print(f"Saved final trained model to: {model_save_path}")
    
    # NOTE: We no longer save a target scaler, as np.expm1 is the inverse.

    # 7. Generate Diagnostic Plots
    print("\n--- Generating Diagnostic Plots for the Best Model ---")
    plots_dir = os.path.join(cfg.OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    lc_title = f"Learning Curve for {best_model_name}"
    lc_path = os.path.join(plots_dir, f"learning_curve_{best_model_name}.png")
    # Learning curve is plotted on the original, un-augmented, un-transformed data for consistency
    plotting.plot_learning_curves(
        clone(final_model_pipeline), lc_title, X_dev, y_dev_original, groups=groups_dev, save_path=lc_path
    )
    final_model_instance = final_model_pipeline.named_steps['model']
    fi_title = f"Feature Importance for {best_model_name}"
    fi_path = os.path.join(plots_dir, f"feature_importance_{best_model_name}.png")
    plotting.plot_feature_importance(
        final_model_instance, X_dev.columns.tolist(), fi_title, fi_path
    )

if __name__ == "__main__":
    main()

# python -m src_feature_based.train_feature_models_cv