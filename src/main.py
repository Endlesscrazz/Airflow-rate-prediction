# --- START OF FILE main.py ---

# main.py
"""
Main script to run the airflow prediction regression experiment using
pre-computed hotspot masks. Allows easy switching of feature sets.
RUNNING EXPERIMENT 11: Combining Peak Pixel rate/mag with temp distribution.
"""

import os
import pandas as pd
import joblib
import numpy as np
import time
import warnings
import fnmatch
import traceback
from tqdm import tqdm # Added for feature extraction loop progress

# Import project modules
import config
import data_utils
import feature_engineering # Imports the modified version
import modeling
import plotting
import tuning

# ... (rest of imports: models, metrics, exceptions) ...
from modeling import get_regressors
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.inspection import permutation_importance # Keep for analysis
import matplotlib.pyplot as plt # Keep for analysis plotting

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*SettingWithCopyWarning*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Constants ---
PRECOMPUTED_MASK_DIR = getattr(config, 'MASK_DIR', 'output_hotspot_mask_SM3') # Use 10s masks

# --- Feature Selection Control ---
# Define ALL possible features that feature_engineering now returns
ALL_POSSIBLE_FEATURE_NAMES = [
    'hotspot_area',
    'hotspot_temp_change_rate', 'hotspot_temp_change_magnitude',
    'peak_pixel_temp_change_rate', 'peak_pixel_temp_change_magnitude', # Added peak
    'temp_mean_avg', 'temp_std_avg', 'temp_min_overall', 'temp_max_overall',
    # Add activity features here if you recalculate/load the activity map later
]

# === FEATURE SET FOR EXPERIMENT 11 ===
SELECTED_FEATURE_NAMES_FROM_MASK = [
    'hotspot_temp_change_rate',      # Use AVERAGE rate again
    'hotspot_temp_change_magnitude', # Use AVERAGE magnitude again
    'temp_mean_avg',
    'temp_std_avg',
    'temp_min_overall',
    'temp_max_overall',
]
# ==================================

# Parameters for feature calculation (should match mask generation -> 10s)
FPS_FOR_FEATURES = getattr(config, 'FPS', 5.0)
FOCUS_DURATION_SEC_FOR_FEATURES = getattr(config, 'FOCUS_DURATION_SEC', 10.0)
ENVIR_PARA = getattr(config, 'ENVIR_PARA', -1) # Needed by feature_eng


def run_experiment():
    """Loads data, extracts features (peak pixel + dist), selects, trains/evaluates."""

    print("--- Starting Airflow Prediction Experiment (Regression) ---")
    print("--- Using Pre-computed Hotspot Masks ---")
    print(f"--- Mask Directory: {PRECOMPUTED_MASK_DIR} ---")
    print(f"--- Feature Calculation (Peak Pixel + Dist): FPS={FPS_FOR_FEATURES}, Focus Duration={FOCUS_DURATION_SEC_FOR_FEATURES}s ---")
    print(f"--- Selected Raw Mask Features: {SELECTED_FEATURE_NAMES_FROM_MASK} ---")
    print(f"--- Models: {list(modeling.get_regressors().keys())} ---")

    # 1. Find Data Files and Corresponding Masks
    # ... (Keep file scanning logic exactly as before) ...
    all_samples_info = []
    input_dataset_dir = config.DATASET_FOLDER
    # ... (rest of file scanning logic) ...
    if not input_dataset_dir or not os.path.isdir(input_dataset_dir): print(f"Error: Dataset folder not found: {input_dataset_dir}"); return
    print(f"\nScanning for .mat files in: {input_dataset_dir}")
    file_count, skipped_meta_count, skipped_mask_count = 0, 0, 0
    for root, _, files in os.walk(input_dataset_dir):
        folder_name = os.path.basename(root)
        for mat_filename in sorted(fnmatch.filter(files, '*.mat')):
            file_count += 1; mat_filepath = os.path.join(root, mat_filename)
            relative_dir = os.path.relpath(root, input_dataset_dir)
            mat_filename_no_ext = os.path.splitext(mat_filename)[0]
            mask_filename = mat_filename_no_ext + '_mask.npy'
            # Adjust mask path finding if structure changed
            mask_path_try1 = os.path.join(PRECOMPUTED_MASK_DIR, relative_dir, mat_filename_no_ext, mask_filename)
            mask_path_try2 = os.path.join(PRECOMPUTED_MASK_DIR, relative_dir, mask_filename)
            if os.path.exists(mask_path_try1): mask_path = mask_path_try1
            elif os.path.exists(mask_path_try2): mask_path = mask_path_try2
            else: skipped_mask_count += 1; continue
            try: airflow_rate = data_utils.parse_airflow_rate(folder_name); delta_T = data_utils.parse_delta_T(mat_filename); assert delta_T is not None
            except Exception as e: skipped_meta_count += 1; continue
            all_samples_info.append({"mat_filepath": mat_filepath, "mask_path": mask_path, "delta_T": float(delta_T), "airflow_rate": float(airflow_rate),"source_folder_name": folder_name, "mat_filename_no_ext": mat_filename_no_ext})
    print(f"\nFound {file_count} .mat files. Skipped Meta: {skipped_meta_count}, Skipped Mask: {skipped_mask_count}. Proceeding with {len(all_samples_info)} samples.")
    if not all_samples_info: print("Error: No valid samples found."); return


    # 2. Extract Features using Masks
    feature_list = []
    print("\n--- Extracting Features (Peak Pixel + Dist) ---")
    start_time = time.time()
    processed_samples, feature_error_count = 0, 0
    for i, sample_info in enumerate(tqdm(all_samples_info, desc="Extracting Features", ncols=100)):
        # Call the updated feature extraction function
        extracted_features_dict = feature_engineering.extract_features_with_mask(
            frames_or_path=sample_info["mat_filepath"],
            mask_path=sample_info["mask_path"],
            fps=FPS_FOR_FEATURES,
            focus_duration_sec=FOCUS_DURATION_SEC_FOR_FEATURES,
            envir_para=ENVIR_PARA, # Still needed for peak calc inside feature_eng
            source_folder_name=sample_info["source_folder_name"],
            mat_filename_no_ext=sample_info["mat_filename_no_ext"]
)

        # --- Combine features ---
        # Initialize with ALL POSSIBLE feature names + metadata
        combined_features = {name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES} # Use updated list
        combined_features["delta_T"] = sample_info["delta_T"]
        combined_features["airflow_rate"] = sample_info["airflow_rate"]

        if extracted_features_dict and isinstance(extracted_features_dict, dict):
             has_valid_selected_feature = False
             for key, value in extracted_features_dict.items():
                 if key in combined_features: combined_features[key] = value
                 # Check if any *selected* feature is valid
                 if key in SELECTED_FEATURE_NAMES_FROM_MASK and np.isfinite(value): has_valid_selected_feature = True

             if has_valid_selected_feature: feature_list.append(combined_features); processed_samples += 1
             else: feature_error_count += 1;
        else: feature_error_count += 1;

    end_time = time.time()
    print(f"\nFeature extraction finished. Processed: {processed_samples}, Errors/Skipped: {feature_error_count}. Took: {end_time - start_time:.2f}s.")
    if not feature_list: print("Error: No feature rows generated."); return

    # 3. Create DataFrame and Prepare X, y
    df = pd.DataFrame(feature_list)
    print(f"\nDataFrame shape before processing: {df.shape}")

    # --- Define BASE feature columns ---
    current_feature_columns = ["delta_T"] + SELECTED_FEATURE_NAMES_FROM_MASK
    X = df[current_feature_columns].copy() # Start with selected raw features + delta_T

    # --- Prepare y ---
    if df["airflow_rate"].isnull().any(): df = df.dropna(subset=["airflow_rate"]); X = X.loc[df.index] # Ensure X matches y index
    if df.empty: print("Error: All rows dropped."); return
    y = df["airflow_rate"].astype(float)

    # --- Calculate Derived Features (Normalization, Log Transform) ---
    cols_to_add = []
    cols_to_remove = []

    # Normalize peak pixel features
    if 'peak_pixel_temp_change_rate' in X.columns:
         X['peak_pixel_rate_norm'] = df.apply(lambda r: r['peak_pixel_temp_change_rate'] / r['delta_T'] if r['delta_T']!=0 else np.nan, axis=1)
         cols_to_add.append('peak_pixel_rate_norm')
         cols_to_remove.append('peak_pixel_temp_change_rate')
    if 'peak_pixel_temp_change_magnitude' in X.columns:
         X['peak_pixel_mag_norm'] = df.apply(lambda r: r['peak_pixel_temp_change_magnitude'] / r['delta_T'] if r['delta_T']!=0 else np.nan, axis=1)
         cols_to_add.append('peak_pixel_mag_norm')
         cols_to_remove.append('peak_pixel_temp_change_magnitude')

    # Log Transform Delta T
    apply_log_transform_deltaT = True
    if apply_log_transform_deltaT and 'delta_T' in X.columns:
        if (df['delta_T'] <= 0).any(): print("Warning: Non-positive delta_T values.")
        X['delta_T_log'] = np.log1p(df['delta_T'].astype(float))
        cols_to_add.append('delta_T_log')
        cols_to_remove.append('delta_T')

    # Keep selected raw temp distribution features (no transform/norm applied here)
    for col in ['temp_mean_avg', 'temp_std_avg', 'temp_min_overall', 'temp_max_overall']:
         if col in current_feature_columns:
             # No changes needed, keep original name
             pass

    # --- Final Feature Selection for X ---
    final_feature_columns = [col for col in current_feature_columns if col in X.columns] # Start with selected base columns
    final_feature_columns.extend([col for col in cols_to_add if col in X.columns]) # Add new derived columns
    final_feature_columns = [col for col in final_feature_columns if col not in cols_to_remove] # Remove originals that were replaced
    final_feature_columns = sorted(list(set(final_feature_columns))) # Ensure unique and sorted
    X = X[final_feature_columns] # Select final set
    print(f"\nFinal Features for Model: {list(X.columns)}")


    # --- Handle NaNs/Infs and Final Checks ---
    # ... (keep checks as before) ...
    if X.isnull().any().any() or np.isinf(X.values).any(): print("\nWarning: NaN/Inf detected BEFORE imputation.")
    cols_before_drop = X.shape[1]; all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols: X = X.drop(columns=all_nan_cols); print(f"Warning: Dropped all-NaN columns: {all_nan_cols}")
    if X.empty or y.empty: print("Error: X or y empty."); return
    if X.shape[1] == 0: print("Error: X has no columns left."); return
    if X.shape[0] != len(y): print(f"Error: Mismatch X rows ({X.shape[0]}) vs y length ({len(y)})."); return
    print(f"Final Feature matrix X shape: {X.shape}"); print(f"Final Target vector y shape: {y.shape}")


    # --- Steps 4-8 (Tuning, Evaluation, Plotting, Saving) ---
    # ... (Keep these sections exactly as in the previous version) ...
    # ... (No changes needed here, they operate on the final X) ...

    print("\n--- Running Hyperparameter Tuning ---") # 4
    all_best_params, all_best_scores = tuning.run_grid_search_all_models(X, y)
    # ... (Print tuning summary) ...
    print("\n--- Tuning Results Summary ---")
    for name in all_best_params: score = all_best_scores.get(name, -np.inf); mse = -score if np.isfinite(score) else np.inf; rmse = np.sqrt(mse) if np.isfinite(mse) else np.inf; print(f"{name}: Best Score={score:.4f} (MSE={mse:.4f}, RMSE={rmse:.4f}), Best Params={all_best_params.get(name, {})}" if np.isfinite(score) else f"{name}: Tuning failed/skipped.")

    # CV Setup # 5
    # ... (CV setup logic) ...
    num_samples_final=len(y); cv_strategy=None; cv_name="Unknown"; # ... (rest of CV setup) ...
    if config.CV_METHOD == 'LeaveOneOut': cv_strategy=LeaveOneOut(); n_splits=num_samples_final; cv_name="LeaveOneOut"
    elif config.CV_METHOD == 'KFold': k_folds_config=getattr(config,'K_FOLDS',5); n_splits=min(k_folds_config,num_samples_final); cv_strategy=KFold(n_splits=n_splits,shuffle=True,random_state=config.RANDOM_STATE) if n_splits>=2 else LeaveOneOut(); cv_name=f"{n_splits}-Fold KFold" if n_splits>=2 else "LeaveOneOut(Fallback)"
    else: cv_strategy=LeaveOneOut(); n_splits=num_samples_final; cv_name="LeaveOneOut(Default)"
    print(f"\nUsing {cv_name} Cross-Validation ({n_splits} splits) for evaluation.")

    # Evaluation # 6
    # ... (Evaluation loop) ...
    print(f"\n--- Evaluating ALL Models using Best Parameters ---")
    evaluation_results = {}; all_regressors_to_eval = modeling.get_regressors()
    for name, model_instance in all_regressors_to_eval.items():
        print(f"\nEvaluating model: {name}...") # ... (rest of evaluation loop) ...
        tuned_params_raw = all_best_params.get(name, model_instance.get_params()); tuned_params = {k.replace('model__', ''): v for k, v in tuned_params_raw.items() if k.startswith('model__') or k in model_instance.get_params() }
        if tuned_params and name in all_best_params and np.isfinite(all_best_scores.get(name, -np.inf)): print(f"  Applying Tuned Parameters: {tuned_params}"); model_instance.set_params(**tuned_params)
        else: print(f"  Using default parameters for {name}."); model_instance = modeling.get_regressors()[name]
        pipeline_model = modeling.build_pipeline(model_instance, pca_components=None)
        try:
            y_pred_cv = cross_val_predict(pipeline_model, X, y, cv=cv_strategy, n_jobs=-1)
            if np.isnan(y_pred_cv).any(): print(f"  Warning: NaN values in CV predictions for {name}.")
            mse = mean_squared_error(y, y_pred_cv); rmse = np.sqrt(mse); mae = mean_absolute_error(y, y_pred_cv); r2 = r2_score(y, y_pred_cv)
            results_dict = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred_cv': y_pred_cv }; evaluation_results[name] = results_dict
            print(f"--- CV Results for {name} ---"); print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        except Exception as e: print(f"Error during CV evaluation for {name}: {e}"); traceback.print_exc(); evaluation_results[name] = None

    # Analyze & Plot # 7
    # ... (Analyze results, select best model, plot) ...
    valid_results = {name: res for name, res in evaluation_results.items() if res is not None and np.isfinite(res['r2'])}
    if not valid_results: print("\nError: All model evaluations failed or yielded non-finite R2."); return
    best_model_name = max(valid_results, key=lambda name: valid_results[name]['r2']); best_result_metrics = valid_results[best_model_name]
    print(f"\n--- Best Model based on CV R²: {best_model_name} ---"); print(f"MSE: {best_result_metrics['mse']:.4f} | RMSE: {best_result_metrics['rmse']:.4f} | MAE: {best_result_metrics['mae']:.4f} | R²: {best_result_metrics['r2']:.4f}")
    # Plotting
    save_actual_pred_plot = getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', False); actual_pred_plot_path = getattr(config, 'ACTUAL_VS_PREDICTED_PLOT_SAVE_PATH', None); plot_title = f'Actual vs. Predicted - {best_model_name} (Tuned*, {cv_name}-CV)'; save_path_arg = None
    if save_actual_pred_plot and actual_pred_plot_path: base, ext = os.path.splitext(actual_pred_plot_path); plot_save_path = f"{base}_{best_model_name}{ext}"; print(f"\nSaving plot for {best_model_name} to: {plot_save_path}"); os.makedirs(os.path.dirname(plot_save_path), exist_ok=True); save_path_arg = plot_save_path
    else: print("\nDisplaying plot (not saving).")
    try:
        if len(y) == len(best_result_metrics['y_pred_cv']): plotting.plot_actual_vs_predicted(y, best_result_metrics['y_pred_cv'], plot_title, save_path_arg)
        else: print("Warning: Length mismatch for plot.")
    except Exception as e: print(f"Error plotting: {e}")


    # Final Train, Save, Analyze # 8
    # ... (Keep final training, permutation importance, saving logic as before) ...
    print(f"\n--- Training final {best_model_name} regressor on all data ---")
    try:
        # ... (final model training code) ...
        final_model_instance = modeling.get_regressors()[best_model_name]; best_params_raw = all_best_params.get(best_model_name, {}); best_params_final = {k.replace('model__', ''): v for k, v in best_params_raw.items() if k in final_model_instance.get_params() or k.replace('model__','') in final_model_instance.get_params() }
        if best_params_final and best_model_name in all_best_params and np.isfinite(all_best_scores.get(best_model_name,-np.inf)): print(f"  Applying Tuned Parameters: {best_params_final}"); final_model_instance.set_params(**best_params_final)
        else: print(f"  Using default parameters for final {best_model_name} model.")
        final_pipeline = modeling.build_pipeline(final_model_instance, pca_components=None); final_pipeline.fit(X, y); print("Final model training complete.")

        # Permutation Importance
        print(f"\n--- Analyzing Features for Final {best_model_name} Model ---")
        print("Running Permutation Importance (on training data)...")
        start_perm_time = time.time(); perm_result = permutation_importance(final_pipeline, X, y, n_repeats=15, random_state=config.RANDOM_STATE, n_jobs=-1, scoring='r2'); end_perm_time = time.time() # Use r2 scoring for importance
        print(f"Permutation Importance calculation took {end_perm_time - start_perm_time:.2f} seconds.")
        sorted_importances_idx = perm_result.importances_mean.argsort(); importances = pd.DataFrame(perm_result.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx],)
        plt.figure(figsize=(10, max(6, len(X.columns)*0.5))); importances.plot.box(vert=False, whis=10, ax=plt.gca()); plt.title(f"Permutation Importances for {best_model_name} (Train Set)"); plt.axvline(x=0, color="k", linestyle="--"); plt.xlabel("Decrease in R² score"); plt.tight_layout(); perm_plot_path = f"output/permutation_importance_{best_model_name}.png"; os.makedirs(os.path.dirname(perm_plot_path), exist_ok=True); plt.savefig(perm_plot_path, dpi=150); print(f"Permutation importance plot saved to: {perm_plot_path}"); plt.close()
        print("\nFeature Importances (Mean Decrease in R² Score):"); [print(f"{X.columns[i]:<35}: {perm_result.importances_mean[i]:.4f} +/- {perm_result.importances_std[i]:.4f}") for i in sorted_importances_idx[::-1]]

        # Save Model
        model_base, model_ext = os.path.splitext(getattr(config, 'MODEL_SAVE_PATH', 'final_model.joblib')); final_model_save_path = f"{model_base}_{best_model_name}{model_ext}"; os.makedirs(os.path.dirname(final_model_save_path), exist_ok=True); joblib.dump(final_pipeline, final_model_save_path)
        print(f"\nFinal {best_model_name} pipeline saved to: {final_model_save_path}")
    except Exception as e: print(f"Error during final {best_model_name} training/saving/analysis: {e}"); traceback.print_exc()


    print("\n--- Experiment Finished ---")

if __name__ == "__main__":
    run_experiment()
# --- END OF FILE main.py ---