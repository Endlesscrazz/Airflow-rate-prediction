# main.py
"""
Main script to run the airflow prediction regression experiment using
pre-computed hotspot masks and updated feature engineering.
Allows easy switching of feature sets. Includes difference, area, and stabilized features.
Saves plots for ALL evaluated models.
"""

import os
import pandas as pd
import joblib
import numpy as np
import time
import warnings
import fnmatch
import traceback
from tqdm import tqdm

# Import project modules
import config
import data_utils
import feature_engineering
import modeling
import plotting
import tuning

from modeling import get_regressors
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*SettingWithCopyWarning*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Constants ---
PRECOMPUTED_MASK_DIR = getattr(
    config, 'MASK_DIR', 'hotspot_masks/slope_p01_focus5s_q995_roi20')
THRESHOLD_ABS_CHANGE_FOR_AREA = getattr(config, 'FIXED_AREA_THRESHOLD', 0.5)

# --- Feature Selection Control ---
ALL_POSSIBLE_FEATURE_NAMES = [
    'hotspot_area',
    'hotspot_avg_temp_change_rate_initial', 'hotspot_avg_temp_change_magnitude_initial',
    'peak_pixel_temp_change_rate_initial', 'peak_pixel_temp_change_magnitude_initial',
    'temp_mean_avg_initial', 'temp_std_avg_initial',
    'temp_min_overall_initial', 'temp_max_overall_initial',
    'stabilized_mean_deltaT', 'overall_mean_deltaT', 'max_abs_mean_deltaT',
    'stabilized_std_deltaT', 'overall_std_deltaT',
    'mean_area_significant_change', 'stabilized_area_significant_change', 'max_area_significant_change',
]


SELECTED_FEATURE_NAMES_FROM_MASK = [
    'hotspot_area',
    'hotspot_avg_temp_change_rate_initial',
    # 'hotspot_avg_temp_change_magnitude_initial',
    'temp_max_overall_initial',
    'overall_std_deltaT',
    #'max_area_significant_change'
]

# Parameters for feature calculation
FPS_FOR_FEATURES = getattr(config, 'FPS', 5.0)
FOCUS_DURATION_SEC_FOR_FEATURES = getattr(config, 'FOCUS_DURATION_SEC', 5.0)
ENVIR_PARA = getattr(config, 'ENVIR_PARA', -1)

# --- FLAGS FOR NORMALIZATION/TRANSFORMATION (Control these for experiments) ---
NORMALIZE_AVG_RATE_INITIAL = True
NORMALIZE_AVG_MAG_INITIAL = True
NORMALIZE_PEAK_RATE_INITIAL = True  # If using peak features
NORMALIZE_PEAK_MAG_INITIAL = True  # If using peak features
NORMALIZE_STABILIZED_MEAN_DELTAT = True  # If using stabilized features
LOG_TRANSFORM_DELTA_T = True
LOG_TRANSFORM_AREA = True


def run_experiment():
    print("--- Starting Airflow Prediction Experiment (Regression) ---")
    print(
        f"--- Feature Params: FPS={FPS_FOR_FEATURES}, InitialFocus={FOCUS_DURATION_SEC_FOR_FEATURES}s, AreaThresh={THRESHOLD_ABS_CHANGE_FOR_AREA}°C ---")
    print(
        f"--- RAW Features Selected from Mask: {SELECTED_FEATURE_NAMES_FROM_MASK} ---")
    print(f"--- Models: {list(modeling.get_regressors().keys())} ---")

    # 1. Find Data Files and Corresponding Masks
    all_samples_info = []
    input_dataset_dir = config.DATASET_FOLDER
    if not input_dataset_dir or not os.path.isdir(input_dataset_dir):
        print(f"Error: Dataset folder not found: {input_dataset_dir}")
        return
    print(f"\nScanning for .mat files in: {input_dataset_dir}")
    file_count, skipped_meta_count, skipped_mask_count = 0, 0, 0
    for root, _, files in os.walk(input_dataset_dir):
        folder_name = os.path.basename(root)
        for mat_filename in sorted(fnmatch.filter(files, '*.mat')):
            file_count += 1
            mat_filepath = os.path.join(root, mat_filename)
            relative_dir = os.path.relpath(root, input_dataset_dir)
            mat_filename_no_ext = os.path.splitext(mat_filename)[0]
            mask_filename = mat_filename_no_ext + '_mask.npy'
            mask_path = os.path.join(
                PRECOMPUTED_MASK_DIR, relative_dir, mask_filename)
            if not os.path.exists(mask_path):
                mask_path_alt = os.path.join(
                    PRECOMPUTED_MASK_DIR, relative_dir, mat_filename_no_ext, mask_filename)
                mask_path = mask_path_alt if os.path.exists(
                    mask_path_alt) else mask_path
            if not os.path.exists(mask_path):
                skipped_mask_count += 1
                continue
            try:
                airflow_rate = data_utils.parse_airflow_rate(folder_name)
                delta_T = data_utils.parse_delta_T(mat_filename)
                assert delta_T is not None
            except Exception as e:
                skipped_meta_count += 1
                continue
            all_samples_info.append({"mat_filepath": mat_filepath, "mask_path": mask_path, "delta_T": float(
                delta_T), "airflow_rate": float(airflow_rate), "source_folder_name": folder_name, "mat_filename_no_ext": mat_filename_no_ext})
    print(f"\nFound {file_count} .mat files. Skipped Meta: {skipped_meta_count}, Skipped Mask: {skipped_mask_count}. Proceeding with {len(all_samples_info)} samples.")
    if not all_samples_info:
        print("Error: No valid samples found.")
        return

    # 2. Extract Features using Masks
    feature_list = []
    print("\n--- Extracting Features ---")
    start_time = time.time()
    processed_samples, feature_error_count = 0, 0
    for i, sample_info in enumerate(tqdm(all_samples_info, desc="Extracting Features", ncols=100)):
        extracted_features_dict = feature_engineering.extract_features_with_mask(
            frames_or_path=sample_info["mat_filepath"],
            mask_path=sample_info["mask_path"],
            fps=FPS_FOR_FEATURES,
            focus_duration_sec=FOCUS_DURATION_SEC_FOR_FEATURES,
            envir_para=ENVIR_PARA,
            threshold_abs_change=THRESHOLD_ABS_CHANGE_FOR_AREA,
            source_folder_name=sample_info["source_folder_name"],
            mat_filename_no_ext=sample_info["mat_filename_no_ext"]
        )

        combined_features = {
            name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES}
        combined_features["delta_T"] = sample_info["delta_T"]
        combined_features["airflow_rate"] = sample_info["airflow_rate"]

        if extracted_features_dict and isinstance(extracted_features_dict, dict):
            has_valid_selected_feature = False
            for key, value in extracted_features_dict.items():
                if key in combined_features:
                    combined_features[key] = value
                    if key in SELECTED_FEATURE_NAMES_FROM_MASK and np.isfinite(value):
                        has_valid_selected_feature = True
            if has_valid_selected_feature:
                feature_list.append(combined_features)
                processed_samples += 1
            else:
                feature_error_count += 1
        else:
            feature_error_count += 1

    end_time = time.time()
    print(
        f"\nFeature extraction finished. Processed: {processed_samples}, Errors/Skipped: {feature_error_count}. Took: {end_time - start_time:.2f}s.")
    if not feature_list:
        print("Error: No feature rows generated.")
        return

    # 3. Create DataFrame and Prepare X, y
    df = pd.DataFrame(feature_list)
    df = df.infer_objects()
    print(f"\nDataFrame with ALL extracted features shape: {df.shape}")

    # --- Prepare y (Target Variable) ---
    if "airflow_rate" not in df.columns:
        print("Error: 'airflow_rate' column missing.")
        return
    if df["airflow_rate"].isnull().any():
        print(
            f"Warning: Found {df['airflow_rate'].isnull().sum()} NaN values in target. Dropping rows.")
        df = df.dropna(subset=["airflow_rate"])
    if df.empty:
        print("Error: All rows dropped after handling target NaNs.")
        return
    y = df["airflow_rate"].astype(float)

    # --- Construct Final X based on SELECTED_FEATURE_NAMES_FROM_MASK and transformations ---
    print(f"\nConstructing final feature set for model...")
    # Use index from potentially y-NaN-dropped df
    X_final = pd.DataFrame(index=df.index)
    final_feature_names_for_model = []

    # a) Process delta_T
    if LOG_TRANSFORM_DELTA_T:
        if 'delta_T' in df.columns:
            if (df['delta_T'] <= 0).any():
                print("Warning: Non-positive delta_T values.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
            X_final['delta_T_log'] = np.log1p(df['delta_T'].astype(float))
            final_feature_names_for_model.append('delta_T_log')
            print("  Added: delta_T_log")
    elif 'delta_T' in df.columns:  # Keep raw if not log-transforming
        X_final['delta_T'] = df['delta_T']
        final_feature_names_for_model.append('delta_T')
        print("  Added: delta_T (raw)")

    # b) Process hotspot_area
    if 'hotspot_area' in SELECTED_FEATURE_NAMES_FROM_MASK and 'hotspot_area' in df.columns:
        if LOG_TRANSFORM_AREA:
            if (df['hotspot_area'] < 0).any():
                print("Warning: Negative hotspot_area values.")
            X_final['hotspot_area_log'] = np.log1p(
                df['hotspot_area'].astype(float).clip(lower=0))
            final_feature_names_for_model.append('hotspot_area_log')
            print("  Added: hotspot_area_log")
        else:
            X_final['hotspot_area'] = df['hotspot_area']
            final_feature_names_for_model.append('hotspot_area')
            print("  Added: hotspot_area (raw)")

    # c) Process hotspot_avg_temp_change_rate_initial
    if 'hotspot_avg_temp_change_rate_initial' in SELECTED_FEATURE_NAMES_FROM_MASK and \
       'hotspot_avg_temp_change_rate_initial' in df.columns:
        if NORMALIZE_AVG_RATE_INITIAL:
            X_final['hotspot_avg_temp_change_rate_initial_norm'] = df.apply(
                lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_rate_initial']) else np.nan, axis=1)
            final_feature_names_for_model.append(
                'hotspot_avg_temp_change_rate_initial_norm')
            print("  Added: hotspot_avg_temp_change_rate_initial_norm")
        else:
            X_final['hotspot_avg_temp_change_rate_initial'] = df['hotspot_avg_temp_change_rate_initial']
            final_feature_names_for_model.append(
                'hotspot_avg_temp_change_rate_initial')
            print("  Added: hotspot_avg_temp_change_rate_initial (raw)")

    # d) Process hotspot_avg_temp_change_magnitude_initial
    if 'hotspot_avg_temp_change_magnitude_initial' in SELECTED_FEATURE_NAMES_FROM_MASK and \
       'hotspot_avg_temp_change_magnitude_initial' in df.columns:
        if NORMALIZE_AVG_MAG_INITIAL:
            X_final['hotspot_avg_temp_change_magnitude_initial_norm'] = df.apply(
                lambda r: r['hotspot_avg_temp_change_magnitude_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_magnitude_initial']) else np.nan, axis=1)
            final_feature_names_for_model.append(
                'hotspot_avg_temp_change_magnitude_initial_norm')
            print("  Added: hotspot_avg_temp_change_magnitude_initial_norm")
        else:
            X_final['hotspot_avg_temp_change_magnitude_initial'] = df['hotspot_avg_temp_change_magnitude_initial']
            final_feature_names_for_model.append(
                'hotspot_avg_temp_change_magnitude_initial')
            print("  Added: hotspot_avg_temp_change_magnitude_initial (raw)")

    # e) Process peak_pixel_temp_change_rate_initial
    if 'peak_pixel_temp_change_rate_initial' in SELECTED_FEATURE_NAMES_FROM_MASK and \
       'peak_pixel_temp_change_rate_initial' in df.columns:
        if NORMALIZE_PEAK_RATE_INITIAL:
            X_final['peak_pixel_rate_norm'] = df.apply(
                lambda r: r['peak_pixel_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['peak_pixel_temp_change_rate_initial']) else np.nan, axis=1)
            final_feature_names_for_model.append('peak_pixel_rate_norm')
            print("  Added: peak_pixel_rate_norm")
        else:
            X_final['peak_pixel_temp_change_rate_initial'] = df['peak_pixel_temp_change_rate_initial']
            final_feature_names_for_model.append(
                'peak_pixel_temp_change_rate_initial')
            print("  Added: peak_pixel_temp_change_rate_initial (raw)")

    # f) Process peak_pixel_temp_change_magnitude_initial
    if 'peak_pixel_temp_change_magnitude_initial' in SELECTED_FEATURE_NAMES_FROM_MASK and \
       'peak_pixel_temp_change_magnitude_initial' in df.columns:
        if NORMALIZE_PEAK_MAG_INITIAL:
            X_final['peak_pixel_mag_norm'] = df.apply(
                lambda r: r['peak_pixel_temp_change_magnitude_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['peak_pixel_temp_change_magnitude_initial']) else np.nan, axis=1)
            final_feature_names_for_model.append('peak_pixel_mag_norm')
            print("  Added: peak_pixel_mag_norm")
        else:
            X_final['peak_pixel_temp_change_magnitude_initial'] = df['peak_pixel_temp_change_magnitude_initial']
            final_feature_names_for_model.append(
                'peak_pixel_temp_change_magnitude_initial')
            print("  Added: peak_pixel_temp_change_magnitude_initial (raw)")

    # g) Process other selected features (used as raw from SELECTED_FEATURE_NAMES_FROM_MASK)
    # This includes temp distribution features, stabilized features etc.
    other_raw_features = [
        'temp_mean_avg_initial', 'temp_std_avg_initial', 'temp_min_overall_initial', 'temp_max_overall_initial',
        'stabilized_mean_deltaT', 'overall_mean_deltaT', 'max_abs_mean_deltaT',
        'stabilized_std_deltaT', 'overall_std_deltaT',
        'mean_area_significant_change', 'stabilized_area_significant_change', 'max_area_significant_change',
        'activity_mean', 'activity_median', 'activity_std', 'activity_max', 'activity_sum'
    ]
    for col in other_raw_features:
        if col in SELECTED_FEATURE_NAMES_FROM_MASK and col in df.columns:
            # Check if a normalized version was already explicitly created (e.g., stabilized_mean_deltaT_norm)
            # Only add raw if not already processed and selected for normalization.
            if col == 'stabilized_mean_deltaT' and NORMALIZE_STABILIZED_MEAN_DELTAT:
                if 'stabilized_mean_deltaT_norm' in X_final.columns:  # If norm version exists and was selected
                    print(f"  Using normalized version for {col}")
                else:  # Normalized version not selected, so add raw if it was selected raw
                    X_final[col] = df[col]
                    final_feature_names_for_model.append(col)
                    print(f"  Added: {col} (raw)")
            else:  # For other features, just add if selected
                X_final[col] = df[col]
                final_feature_names_for_model.append(col)
                print(f"  Added: {col} (raw)")

    # Ensure no duplicates and select final columns
    X = X_final[sorted(list(set(final_feature_names_for_model)))]
    print(f"\nFinal Features for Model: {list(X.columns)}")

    # --- Handle NaNs/Infs and Final Checks ---
    if X.isnull().values.any() or np.isinf(X.values).any():
        print(f"\nWarning: NaN/Inf detected BEFORE imputation.")
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        print(f"ERROR: All-NaN columns found: {all_nan_cols}.")
        return
    if X.empty or y.empty:
        print("Error: X or y empty.")
        return
    if X.shape[1] == 0:
        print("Error: X has no columns left.")
        return
    if X.shape[0] != len(y):
        print(
            f"CRITICAL Error: Mismatch X rows ({X.shape[0]}) vs y ({len(y)}).")
        return
    print(f"Final Feature matrix X shape: {X.shape}")
    print(f"Final Target vector y shape: {y.shape}")

    # --- Steps 4-8 (Tuning, Evaluation, Plotting, Saving) ---

    print("\n--- Running Hyperparameter Tuning ---")  # 4
    all_best_params, all_best_scores = tuning.run_grid_search_all_models(X, y)
    print("\n--- Tuning Results Summary ---")
    for name in all_best_params:
        score = all_best_scores.get(name, -np.inf)
        mse = -score if np.isfinite(score) else np.inf
        rmse = np.sqrt(mse) if np.isfinite(mse) else np.inf
        print(f"{name}: Best Score={score:.4f} (MSE={mse:.4f}, RMSE={rmse:.4f}), Best Params={all_best_params.get(name, {})}" if np.isfinite(
            score) else f"{name}: Tuning failed/skipped.")

    # CV Setup # 5
    num_samples_final = len(y)
    cv_strategy = None
    cv_name = "Unknown"
    n_splits = 0
    if config.CV_METHOD == 'LeaveOneOut':
        cv_strategy = LeaveOneOut()
        n_splits = num_samples_final
        cv_name = "LeaveOneOut"
    elif config.CV_METHOD == 'KFold':
        k_folds_config = getattr(config, 'K_FOLDS', 3)
        n_splits = min(k_folds_config, num_samples_final)
        cv_strategy = KFold(n_splits=n_splits, shuffle=True,
                            random_state=config.RANDOM_STATE) if n_splits >= 2 else LeaveOneOut()
        cv_name = f"{n_splits}-Fold KFold" if n_splits >= 2 else "LeaveOneOut(Fallback)"
    else:
        cv_strategy = LeaveOneOut()
        n_splits = num_samples_final
        cv_name = "LeaveOneOut(Default)"
    print(
        f"\nUsing {cv_name} Cross-Validation ({n_splits} splits) for evaluation.")

    # Evaluation # 6
    print(f"\n--- Evaluating ALL Models using Best Parameters ---")
    evaluation_results = {}
    all_regressors_to_eval = modeling.get_regressors()

    for name, model_instance_loop in all_regressors_to_eval.items():  # Use different var name
        print(f"\n===== Processing Model: {name} =====")
        current_model_for_eval = modeling.get_regressors()[
            name]  # Get a fresh instance
        tuned_params_raw = all_best_params.get(
            name, current_model_for_eval.get_params())
        tuned_params = {k.replace('model__', ''): v for k, v in tuned_params_raw.items()
                        if k in current_model_for_eval.get_params() or k.startswith('model__')}

        if tuned_params and name in all_best_params and np.isfinite(all_best_scores.get(name, -np.inf)):
            print(f"  Applying Tuned Parameters for {name}: {tuned_params}")
            try:
                current_model_for_eval.set_params(**tuned_params)
            except ValueError as e:
                print(
                    f"  Warning: Error setting tuned params for {name}: {e}. Using defaults.")
        else:
            print(f"  Using default parameters for {name}.")

        pipeline_model = modeling.build_pipeline(
            current_model_for_eval, pca_components=None)

        try:
            y_pred_cv = cross_val_predict(
                pipeline_model, X, y, cv=cv_strategy, n_jobs=-1)
            if np.isnan(y_pred_cv).any():
                y_pred_cv = np.nan_to_num(y_pred_cv, nan=np.nanmean(y))
            mse = mean_squared_error(y, y_pred_cv)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred_cv)
            r2 = r2_score(y, y_pred_cv)
            results_dict = {'mse': mse, 'rmse': rmse,
                            'mae': mae, 'r2': r2, 'y_pred_cv': y_pred_cv}
            evaluation_results[name] = results_dict
            print(f"--- CV Results for {name} ---")
            print(
                f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        except Exception as e:
            print(f"Error CV eval {name}: {e}")
            evaluation_results[name] = None
            continue

        plot_title_cv = f'Actual vs. Predicted - {name} (CV, Tuned*)'
        actual_pred_plot_base = getattr(
            config, 'ACTUAL_VS_PREDICTED_PLOT_SAVE_PATH', 'output/actual_vs_predicted.png')
        base_plot, ext_plot = os.path.splitext(actual_pred_plot_base)
        plot_save_path_cv = f"{base_plot}_{name}{ext_plot}"
        print(
            f"Saving Actual vs Predicted plot for {name} to: {plot_save_path_cv}")
        os.makedirs(os.path.dirname(plot_save_path_cv), exist_ok=True)

        try:
            if len(y) == results_dict['y_pred_cv'].shape[0]:
                plotting.plot_actual_vs_predicted(
                    y, results_dict['y_pred_cv'], plot_title_cv, plot_save_path_cv)
            else:
                print(f"Warning: Length mismatch plot {name}.")
        except Exception as e:
            print(f"Error plotting Actual vs Predicted for {name}: {e}")

        # --- Final Model Training & Permutation Importance FOR THIS MODEL ---
        print(
            f"Training final {name} regressor on all data for Permutation Importance...")
        try:
            # Use the same tuned model instance for final fit for permutation
            final_pipeline_for_perm = modeling.build_pipeline(
                current_model_for_eval, pca_components=None)  # current_model_for_eval has tuned params
            final_pipeline_for_perm.fit(X, y)
            print(f"Permutation Importance for {name}:")
            perm_result = permutation_importance(
                final_pipeline_for_perm, X, y, n_repeats=15, random_state=config.RANDOM_STATE, n_jobs=-1, scoring='r2')
            if hasattr(perm_result, 'importances_mean'):
                sorted_idx = perm_result.importances_mean.argsort()
                importances_df = pd.DataFrame(
                    perm_result.importances[sorted_idx].T, columns=X.columns[sorted_idx])
                plt.figure(figsize=(10, max(6, len(X.columns)*0.4)))
                importances_df.plot.box(vert=False, whis=10, ax=plt.gca())
                plt.title(f"Permutation Importances for {name} (Train Set)")
                plt.axvline(x=0, color="k", linestyle="--")
                plt.xlabel("Decrease in R² score")
                plt.tight_layout()
                perm_plot_path = f"output/permutation_importance_{name}.png"
                os.makedirs(os.path.dirname(perm_plot_path), exist_ok=True)
                plt.savefig(perm_plot_path, dpi=150)
                plt.close()
                print(
                    f"Permutation importance plot for {name} saved to: {perm_plot_path}")
                print(
                    f"Feature Importances for {name} (Mean Decrease in R² Score):")
                [print(f"  {X.columns[i]:<35}: {perm_result.importances_mean[i]:.4f} +/- {perm_result.importances_std[i]:.4f}")
                 for i in sorted_idx[::-1]]
            else:
                print(f"Warning: Permutation importance failed for {name}.")
        except Exception as e:
            print(f"Error training/permutation for {name}: {e}")

    # 7. Analyze Overall Best Model
    valid_results_overall = {name: res for name, res in evaluation_results.items(
    ) if res is not None and 'r2' in res and np.isfinite(res['r2'])}
    if not valid_results_overall:
        print("\nCRITICAL Error: All model CV evaluations failed.")
        return
    best_model_name_overall = max(
        valid_results_overall, key=lambda name: valid_results_overall[name]['r2'])
    best_result_metrics_overall = valid_results_overall[best_model_name_overall]
    print(
        f"\n--- OVERALL Best Model based on CV R²: {best_model_name_overall} ---")
    print(
        f"MSE: {best_result_metrics_overall['mse']:.4f} | RMSE: {best_result_metrics_overall['rmse']:.4f} | MAE: {best_result_metrics_overall['mae']:.4f} | R²: {best_result_metrics_overall['r2']:.4f}")

    # 8. Final Training and Saving of the OVERALL BEST Model
    print(
        f"\n--- Training final OVERALL BEST model ({best_model_name_overall}) on all data for deployment ---")
    try:
        final_model_instance_overall = modeling.get_regressors()[
            best_model_name_overall]
        best_params_raw_overall = all_best_params.get(
            best_model_name_overall, {})
        best_params_final_overall = {k.replace('model__', ''): v for k, v in best_params_raw_overall.items(
        ) if k in final_model_instance_overall.get_params() or k.startswith('model__')}

        if best_params_final_overall and best_model_name_overall in all_best_params and np.isfinite(all_best_scores.get(best_model_name_overall, -np.inf)):
            print(f"  Applying Tuned Parameters: {best_params_final_overall}")
            final_model_instance_overall.set_params(
                **best_params_final_overall)
        else:
            print(
                f"  Using default parameters for final {best_model_name_overall} model.")

        final_pipeline_overall = modeling.build_pipeline(
            final_model_instance_overall, pca_components=None)
        final_pipeline_overall.fit(X, y)
        print("Overall best model final training complete.")
        # Plot loss curve only if overall best is MLP
        save_loss_plot = getattr(config, 'SAVE_LOSS_CURVE_PLOT', False)
        loss_plot_dir = getattr(config, 'LOSS_CURVE_PLOT_SAVE_DIR', 'output')

        if save_loss_plot and isinstance(final_pipeline_overall.named_steps['model'], MLPRegressor) and \
                hasattr(final_pipeline_overall.named_steps['model'], 'loss_curve_'):
            print(
                f"\nPlotting Training Loss Curve for OVERALL BEST MLP ({best_model_name_overall})...")
            loss_filename = f"final_loss_curve_{best_model_name_overall}.png"

            plotting.plot_loss_curve(
                final_pipeline_overall.named_steps['model'].loss_curve_,
                f"MLP Training Loss Curve ({best_model_name_overall})",
                loss_plot_dir,  # Pass the directory
                filename=loss_filename  # Pass the filename
            )

        # Save the OVERALL BEST model
        model_base, model_ext = os.path.splitext(
            getattr(config, 'MODEL_SAVE_PATH', 'output/final_model.joblib'))
        final_model_save_path_overall = f"{model_base}_{best_model_name_overall}{model_ext}"
        os.makedirs(os.path.dirname(
            final_model_save_path_overall), exist_ok=True)
        joblib.dump(final_pipeline_overall, final_model_save_path_overall)
        print(
            f"\nFinal OVERALL BEST model ({best_model_name_overall}) pipeline saved to: {final_model_save_path_overall}")
    except Exception as e:
        print(
            f"Error during final {best_model_name_overall} training/saving: {e}")
        traceback.print_exc()

    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    run_experiment()

