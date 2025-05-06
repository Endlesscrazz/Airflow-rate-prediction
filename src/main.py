# --- START OF MODIFIED main.py (for Baseline B1) ---

# main.py
"""
Main script to run the airflow prediction regression experiment using
pre-computed hotspot masks and updated feature engineering.
Allows easy switching of feature sets. Includes difference, area, and stabilized features.
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
import feature_engineering # Imports the version that calculates many features
import modeling
import plotting
import tuning

# --- Import models for typing and direct use if needed ---
from modeling import get_regressors # Ensure this returns all models you want to test
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression # Make sure LinearRegression is in get_regressors
from sklearn.svm import SVR

from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*SettingWithCopyWarning*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Constants ---
PRECOMPUTED_MASK_DIR = getattr(config, 'MASK_DIR', 'hotspot_masks/slope_p01_focus5s_q995_roi20') 
#RELATIVE_THRESHOLD_FRACTION = getattr(config, 'RELATIVE_AREA_THRESHOLD', 0.5) # Used in feature_engineering
THRESHOLD_ABS_CHANGE_FOR_AREA = getattr(config, 'FIXED_AREA_THRESHOLD', 0.5) 
# RELATIVE_THRESHOLD_FRACTION = 0.25 # 

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
    'hotspot_avg_temp_change_rate_initial',
    'hotspot_avg_temp_change_magnitude_initial',
    #'peak_pixel_temp_change_rate_initial',
    #'peak_pixel_temp_change_magnitude_initial',
    # 'temp_mean_avg_initial',
    # 'temp_std_avg_initial',
    # 'temp_min_overall_initial',
    'temp_max_overall_initial',
    'hotspot_area',
    # 'stabilized_mean_deltaT',
    # 'overall_mean_deltaT',
    # 'max_abs_mean_deltaT',
    # 'mean_area_significant_change',
    # 'stabilized_area_significant_change',
]


# Parameters for feature calculation (used in feature_engineering.py)
FPS_FOR_FEATURES = getattr(config, 'FPS', 5.0) # Should match mask generation fps
FOCUS_DURATION_SEC_FOR_FEATURES = getattr(config, 'FOCUS_DURATION_SEC', 5.0) # For initial features, should match mask gen
ENVIR_PARA = getattr(config, 'ENVIR_PARA', -1)


## FLAGS FOR NORMALIZATION
hotspot_avg_temp_change_rate_initial_flag = True
hotspot_avg_temp_change_magnitude_initial_flag = True

def run_experiment():
    """Loads data, extracts features, selects final set, trains/evaluates models."""

    print("--- Starting Airflow Prediction Experiment (Regression) ---")
    print("--- Using Pre-computed Hotspot Masks & Iterative Feature Selection ---")
    print(f"--- Mask Directory: {PRECOMPUTED_MASK_DIR} ---")
    #print(f"--- Feature Params: FPS={FPS_FOR_FEATURES}, InitialFocus={FOCUS_DURATION_SEC_FOR_FEATURES}s, RelAreaThresh={RELATIVE_THRESHOLD_FRACTION} ---")
    print(f"--- Feature Params: FPS={FPS_FOR_FEATURES}, InitialFocus={FOCUS_DURATION_SEC_FOR_FEATURES}s, AreaThresh (Fixed Abs)={THRESHOLD_ABS_CHANGE_FOR_AREA}°C ---")
    print(f"--- RAW Features Selected from Mask: {SELECTED_FEATURE_NAMES_FROM_MASK} ---")
    print(f"--- Models: {list(modeling.get_regressors().keys())} ---")

    # 1. Find Data Files and Corresponding Masks
    all_samples_info = []
    input_dataset_dir = config.DATASET_FOLDER
    # ... (Keep file scanning logic as before) ...
    if not input_dataset_dir or not os.path.isdir(input_dataset_dir): print(f"Error: Dataset folder not found: {input_dataset_dir}"); return
    print(f"\nScanning for .mat files in: {input_dataset_dir}")
    file_count, skipped_meta_count, skipped_mask_count = 0, 0, 0
    for root, _, files in os.walk(input_dataset_dir):
        folder_name = os.path.basename(root)
        for mat_filename in sorted(fnmatch.filter(files, '*.mat')):
            file_count += 1; mat_filepath = os.path.join(root, mat_filename); relative_dir = os.path.relpath(root, input_dataset_dir)
            mat_filename_no_ext = os.path.splitext(mat_filename)[0]; mask_filename = mat_filename_no_ext + '_mask.npy'
            mask_path = os.path.join(PRECOMPUTED_MASK_DIR, relative_dir, mask_filename)
            if not os.path.exists(mask_path): mask_path_alt = os.path.join(PRECOMPUTED_MASK_DIR, relative_dir, mat_filename_no_ext, mask_filename); mask_path = mask_path_alt if os.path.exists(mask_path_alt) else mask_path
            if not os.path.exists(mask_path): print(f"Mask skip: {mask_filename}"); skipped_mask_count += 1; continue
            try: airflow_rate = data_utils.parse_airflow_rate(folder_name); delta_T = data_utils.parse_delta_T(mat_filename); assert delta_T is not None
            except Exception as e: print(f"Meta skip {mat_filename}: {e}"); skipped_meta_count += 1; continue
            all_samples_info.append({"mat_filepath": mat_filepath, "mask_path": mask_path, "delta_T": float(delta_T), "airflow_rate": float(airflow_rate),"source_folder_name": folder_name, "mat_filename_no_ext": mat_filename_no_ext})
    print(f"\nFound {file_count} .mat files. Skipped Meta: {skipped_meta_count}, Skipped Mask: {skipped_mask_count}. Proceeding with {len(all_samples_info)} samples.")
    if not all_samples_info: print("Error: No valid samples found."); return


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
            #relative_threshold_frac=RELATIVE_THRESHOLD_FRACTION,   # used for relative threshold 
            source_folder_name=sample_info["source_folder_name"],
            mat_filename_no_ext=sample_info["mat_filename_no_ext"]
        )
        # Combine features
        combined_features = {name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES}
        combined_features["delta_T"] = sample_info["delta_T"]; combined_features["airflow_rate"] = sample_info["airflow_rate"]
        if extracted_features_dict:
            has_valid_selected_feature = False
            for key, value in extracted_features_dict.items():
                if key in combined_features: combined_features[key] = value
                if key in SELECTED_FEATURE_NAMES_FROM_MASK and np.isfinite(value): has_valid_selected_feature = True
            if has_valid_selected_feature: feature_list.append(combined_features); processed_samples += 1
            else: feature_error_count += 1;
        else: feature_error_count += 1;

    end_time = time.time()
    print(f"\nFeature extraction finished. Processed: {processed_samples}, Errors/Skipped: {feature_error_count}. Took: {end_time - start_time:.2f}s.")
    if not feature_list: print("Error: No feature rows generated."); return

    # 3. Create DataFrame and Prepare X, y
    df = pd.DataFrame(feature_list)
    df = df.infer_objects()
    print(f"\nDataFrame shape with ALL extracted features: {df.shape}")

    # --- Define BASE feature columns (start with delta_T plus the user-selected raw list) ---
    initial_feature_selection = ["delta_T"] + [col for col in SELECTED_FEATURE_NAMES_FROM_MASK if col in df.columns]
    X_temp = df[initial_feature_selection].copy() # Temporary X with raw selected features

    # --- Prepare y ---
    if "airflow_rate" not in df.columns: print("Error: 'airflow_rate' missing."); return
    if df["airflow_rate"].isnull().any(): df = df.dropna(subset=["airflow_rate"]); X_temp = X_temp.loc[df.index]
    if df.empty: print("Error: All rows dropped."); return
    y = df["airflow_rate"].astype(float)

    # --- Calculate Derived Features & Construct Final X ---
    print(f"\nConstructing final feature set for model...")
    X_final = pd.DataFrame(index=X_temp.index) # Start with an empty DataFrame with correct index
    final_feature_names_for_model = [] # List to build the final feature names

    # --- Process delta_T (Always include its log-transformed version) ---
    if 'delta_T' in X_temp.columns:
        if (X_temp['delta_T'] <= 0).any(): print("Warning: Non-positive delta_T values.")
        with warnings.catch_warnings(): warnings.simplefilter("ignore", category=RuntimeWarning)
        X_final['delta_T_log'] = np.log1p(X_temp['delta_T'].astype(float))
        final_feature_names_for_model.append('delta_T_log')
        print("  Added: delta_T_log")
    else: print("Error: delta_T is missing from initial selection!"); return

    # --- Process SELECTED_FEATURE_NAMES_FROM_MASK for normalization/transformation ---
    if hotspot_avg_temp_change_rate_initial_flag:
        if 'hotspot_avg_temp_change_rate_initial' in SELECTED_FEATURE_NAMES_FROM_MASK and \
        'hotspot_avg_temp_change_rate_initial' in X_temp.columns:
            X_final['hotspot_avg_temp_change_rate_initial_norm'] = X_temp.apply(
                lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_rate_initial']) else np.nan, axis=1)
            final_feature_names_for_model.append('hotspot_avg_temp_change_rate_initial_norm')
            print("  Added: hotspot_avg_temp_change_rate_initial_norm")
    else:
        if 'hotspot_avg_temp_change_rate_initial' in SELECTED_FEATURE_NAMES_FROM_MASK and \
        'hotspot_avg_temp_change_rate_initial' in X_temp.columns:
            X_final['hotspot_avg_temp_change_rate_initial'] = X_temp['hotspot_avg_temp_change_rate_initial']
            final_feature_names_for_model.append('hotspot_avg_temp_change_rate_initial')
            print("  Added: hotspot_avg_temp_change_rate_initial (raw)")

    if hotspot_avg_temp_change_magnitude_initial_flag:
        if 'hotspot_avg_temp_change_magnitude_initial' in SELECTED_FEATURE_NAMES_FROM_MASK and \
        'hotspot_avg_temp_change_magnitude_initial' in X_temp.columns:
            X_final['hotspot_avg_temp_change_magnitude_initial_norm'] = X_temp.apply(
                lambda r: r['hotspot_avg_temp_change_magnitude_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_magnitude_initial']) else np.nan, axis=1)
            final_feature_names_for_model.append('hotspot_avg_temp_change_magnitude_initial_norm')
            print("  Added: hotspot_avg_temp_change_magnitude_initial_norm")

    # --- For BASELINE B2: (If 'peak_pixel_temp_change_rate_initial' was selected)
    if 'peak_pixel_temp_change_rate_initial' in SELECTED_FEATURE_NAMES_FROM_MASK and \
       'peak_pixel_temp_change_rate_initial' in X_temp.columns:
        X_final['peak_pixel_rate_norm'] = X_temp.apply( # Using the final name
            lambda r: r['peak_pixel_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['peak_pixel_temp_change_rate_initial']) else np.nan, axis=1)
        final_feature_names_for_model.append('peak_pixel_rate_norm')
        print("  Added: peak_pixel_rate_norm")

    if 'peak_pixel_temp_change_magnitude_initial' in SELECTED_FEATURE_NAMES_FROM_MASK and \
       'peak_pixel_temp_change_magnitude_initial' in X_temp.columns:
        X_final['peak_pixel_mag_norm'] = X_temp.apply( # Using the final name
            lambda r: r['peak_pixel_temp_change_magnitude_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['peak_pixel_temp_change_magnitude_initial']) else np.nan, axis=1)
        final_feature_names_for_model.append('peak_pixel_mag_norm')
        print("  Added: peak_pixel_mag_norm")

    # Example for hotspot_area (if selected)
    if 'hotspot_area' in SELECTED_FEATURE_NAMES_FROM_MASK and 'hotspot_area' in X_temp.columns:
        if (X_temp['hotspot_area'] < 0).any(): print("Warning: Negative hotspot_area values.")
        X_final['hotspot_area_log'] = np.log1p(X_temp['hotspot_area'].astype(float).clip(lower=0))
        final_feature_names_for_model.append('hotspot_area_log')
        print("  Added: hotspot_area_log")

    # -- Temprature features initial and full duration of video
    #Add RAW temperature distribution features directly
    raw_dist_features = ['temp_mean_avg_initial', 'temp_std_avg_initial', 'temp_min_overall_initial', 'temp_max_overall_initial']
    for col in raw_dist_features:
        if col in X_temp.columns:
            X_final[col] = X_temp[col]
            final_feature_names_for_model.append(col)
            print(f"  Added: {col} (raw)")

    # Make sure X_final only contains the columns intended for this experiment
    X = X_final[sorted(list(set(final_feature_names_for_model)))] # Ensure unique and sorted
    print(f"\nFinal Features for Model: {list(X.columns)}")


    # --- Handle NaNs/Infs and Final Checks ---
    if X.isnull().values.any() or np.isinf(X.values).any(): print(f"\nWarning: NaN/Inf detected BEFORE imputation ({X.isnull().sum().sum()} NaNs, {np.isinf(X.values).sum()} Infs).")
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols: print(f"ERROR: All-NaN columns found: {all_nan_cols}."); return
    if X.empty or y.empty: print("Error: X or y empty."); return
    if X.shape[1] == 0: print("Error: X has no columns left."); return
    if X.shape[0] != len(y): print(f"CRITICAL Error: Mismatch X rows ({X.shape[0]}) vs y ({len(y)})."); return
    print(f"Final Feature matrix X shape: {X.shape}"); print(f"Final Target vector y shape: {y.shape}")
    if len(y) < 5: print(f"Warning: Very small dataset size ({len(y)} samples).")


    # --- Steps 4-8 (Tuning, Evaluation, Plotting, Saving) ---
    # These sections remain unchanged and operate on the final X, y

    print("\n--- Running Hyperparameter Tuning ---") # 4
    all_best_params, all_best_scores = tuning.run_grid_search_all_models(X, y)
    print("\n--- Tuning Results Summary ---")
    for name in all_best_params: score = all_best_scores.get(name, -np.inf); mse = -score if np.isfinite(score) else np.inf; rmse = np.sqrt(mse) if np.isfinite(mse) else np.inf; print(f"{name}: Best Score={score:.4f} (MSE={mse:.4f}, RMSE={rmse:.4f}), Best Params={all_best_params.get(name, {})}" if np.isfinite(score) else f"{name}: Tuning failed/skipped.")

    # CV Setup # 5
    num_samples_final=len(y); cv_strategy=None; cv_name="Unknown"; n_splits=0;
    if config.CV_METHOD == 'LeaveOneOut': cv_strategy=LeaveOneOut(); n_splits=num_samples_final; cv_name="LeaveOneOut"
    elif config.CV_METHOD == 'KFold': k_folds_config=getattr(config,'K_FOLDS',5); n_splits=min(k_folds_config,num_samples_final); cv_strategy=KFold(n_splits=n_splits,shuffle=True,random_state=config.RANDOM_STATE) if n_splits>=2 else LeaveOneOut(); cv_name=f"{n_splits}-Fold KFold" if n_splits>=2 else "LeaveOneOut(Fallback)"
    else: cv_strategy=LeaveOneOut(); n_splits=num_samples_final; cv_name="LeaveOneOut(Default)"
    print(f"\nUsing {cv_name} Cross-Validation ({n_splits} splits) for evaluation.")

    # Evaluation # 6
    print(f"\n--- Evaluating ALL Models using Best Parameters ---")
    evaluation_results = {}; all_regressors_to_eval = modeling.get_regressors()
    for name, model_instance in all_regressors_to_eval.items():
        print(f"\nEvaluating model: {name}...")
        tuned_params_raw = all_best_params.get(name, model_instance.get_params()); tuned_params = {k.replace('model__', ''): v for k, v in tuned_params_raw.items() if k.startswith('model__') or k in model_instance.get_params() }
        model_to_eval = modeling.get_regressors()[name]; # Fresh instance
        if tuned_params and name in all_best_params and np.isfinite(all_best_scores.get(name, -np.inf)): print(f"  Applying Tuned Parameters: {tuned_params}"); model_to_eval.set_params(**tuned_params)
        else: print(f"  Using default parameters for {name}.")
        pipeline_model = modeling.build_pipeline(model_to_eval, pca_components=None)
        try:
            y_pred_cv = cross_val_predict(pipeline_model, X, y, cv=cv_strategy, n_jobs=-1)
            if np.isnan(y_pred_cv).any(): print(f"  Warning: NaN values ({np.isnan(y_pred_cv).sum()}) in CV predictions for {name}.") ; y_pred_cv = np.nan_to_num(y_pred_cv, nan=np.nanmean(y))
            mse = mean_squared_error(y, y_pred_cv); rmse = np.sqrt(mse); mae = mean_absolute_error(y, y_pred_cv); r2 = r2_score(y, y_pred_cv)
            results_dict = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred_cv': y_pred_cv }; evaluation_results[name] = results_dict
            print(f"--- CV Results for {name} ---"); print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        except Exception as e: print(f"Error during CV evaluation for {name}: {e}"); traceback.print_exc(); evaluation_results[name] = None

    # Analyze & Plot # 7
    valid_results = {name: res for name, res in evaluation_results.items() if res is not None and 'r2' in res and np.isfinite(res['r2'])}
    if not valid_results: print("\nError: All model evaluations failed or yielded non-finite R2 scores."); return
    best_model_name = max(valid_results, key=lambda name: valid_results[name]['r2']); best_result_metrics = valid_results[best_model_name]
    print(f"\n--- Best Model based on CV R²: {best_model_name} ---"); print(f"MSE: {best_result_metrics['mse']:.4f} | RMSE: {best_result_metrics['rmse']:.4f} | MAE: {best_result_metrics['mae']:.4f} | R²: {best_result_metrics['r2']:.4f}")
    # Plotting
    save_actual_pred_plot = getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True); actual_pred_plot_path = getattr(config, 'ACTUAL_VS_PREDICTED_PLOT_SAVE_PATH', 'output/actual_vs_predicted.png'); plot_title = f'Actual vs. Predicted - {best_model_name} (Tuned*, {cv_name}-CV)'; save_path_arg = None
    if save_actual_pred_plot and actual_pred_plot_path: base, ext = os.path.splitext(actual_pred_plot_path); plot_save_path = f"{base}_{best_model_name}{ext}"; print(f"\nSaving plot for {best_model_name} to: {plot_save_path}"); os.makedirs(os.path.dirname(plot_save_path), exist_ok=True); save_path_arg = plot_save_path
    else: print("\nDisplaying plot (not saving).")
    try:
        y_true_plot = y; y_pred_plot = best_result_metrics.get('y_pred_cv', None)
        if y_pred_plot is not None and len(y_true_plot) == len(y_pred_plot): plotting.plot_actual_vs_predicted(y_true_plot, y_pred_plot, plot_title, save_path_arg)
        else: print("Warning: Length mismatch or missing predictions for plot.")
    except Exception as e: print(f"Error plotting: {e}")

    # Final Train, Save, Analyze # 8
    print(f"\n--- Training final {best_model_name} regressor on all data ---")
    try:
        final_model_instance = modeling.get_regressors()[best_model_name]; best_params_raw = all_best_params.get(best_model_name, {}); best_params_final = {k.replace('model__', ''): v for k, v in best_params_raw.items() if k in final_model_instance.get_params() or k.startswith('model__') }
        if best_params_final and best_model_name in all_best_params and np.isfinite(all_best_scores.get(best_model_name,-np.inf)): print(f"  Applying Tuned Parameters: {best_params_final}"); final_model_instance.set_params(**best_params_final)
        else: print(f"  Using default parameters for final {best_model_name} model.")
        final_pipeline = modeling.build_pipeline(final_model_instance, pca_components=None); final_pipeline.fit(X, y); print("Final model training complete.")
        print(f"\n--- Analyzing Features for Final {best_model_name} Model ---")
        print("Running Permutation Importance (on training data)...")
        start_perm_time = time.time(); perm_result = permutation_importance(final_pipeline, X, y, n_repeats=15, random_state=config.RANDOM_STATE, n_jobs=-1, scoring='r2'); end_perm_time = time.time()
        print(f"Permutation Importance calculation took {end_perm_time - start_perm_time:.2f} seconds.")
        if hasattr(perm_result, 'importances_mean'):
            sorted_importances_idx = perm_result.importances_mean.argsort(); importances_df = pd.DataFrame( perm_result.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx],)
            plt.figure(figsize=(10, max(6, len(X.columns)*0.4))); importances_df.plot.box(vert=False, whis=10, ax=plt.gca()); plt.title(f"Permutation Importances for {best_model_name} (Train Set)"); plt.axvline(x=0, color="k", linestyle="--"); plt.xlabel("Decrease in R² score"); plt.tight_layout(); perm_plot_path = f"output/permutation_importance_{best_model_name}.png"; os.makedirs(os.path.dirname(perm_plot_path), exist_ok=True); plt.savefig(perm_plot_path, dpi=150); print(f"Permutation importance plot saved to: {perm_plot_path}"); plt.close()
            print("\nFeature Importances (Mean Decrease in R² Score):"); [print(f"  {X.columns[i]:<35}: {perm_result.importances_mean[i]:.4f} +/- {perm_result.importances_std[i]:.4f}") for i in sorted_importances_idx[::-1]]
        else: print("Warning: Permutation importance calculation did not return expected results.")
        model_base, model_ext = os.path.splitext(getattr(config, 'MODEL_SAVE_PATH', 'output/final_model.joblib')); final_model_save_path = f"{model_base}_{best_model_name}{model_ext}"; os.makedirs(os.path.dirname(final_model_save_path), exist_ok=True); joblib.dump(final_pipeline, final_model_save_path)
        print(f"\nFinal {best_model_name} pipeline saved to: {final_model_save_path}")
    except Exception as e: print(f"Error during final {best_model_name} training/saving/analysis: {e}"); traceback.print_exc()

    print("\n--- Experiment Finished ---")

if __name__ == "__main__":
    run_experiment()
# --- END OF MODIFIED main.py ---