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
PRECOMPUTED_MASK_DIR = getattr(config, 'MASK_DIR', 'hotspot_masks/slope_p01_focus5s_q99_env1_roi0')
THRESHOLD_ABS_CHANGE_FOR_AREA = getattr(config, 'FIXED_AREA_THRESHOLD', 0.5)
FPS_FOR_FEATURES = getattr(config, 'MASK_FPS', 5.0)
FOCUS_DURATION_SEC_FOR_FEATURES = getattr(config, 'MASK_FOCUS_DURATION_SEC', 5.0)
ENVIR_PARA = getattr(config, 'MASK_ENVIR_PARA', 1)

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
    'temp_max_overall_initial',
    # 'overall_std_deltaT', # You had this commented out in the provided main.py
]

NORMALIZE_AVG_RATE_INITIAL = True
NORMALIZE_AVG_MAG_INITIAL = True # Ensure these flags align with your feature construction
LOG_TRANSFORM_DELTA_T = True
LOG_TRANSFORM_AREA = True
# Add other normalization flags if used:
# NORMALIZE_PEAK_RATE_INITIAL = True
# NORMALIZE_PEAK_MAG_INITIAL = True
# NORMALIZE_STABILIZED_MEAN_DELTAT = True


def run_experiment():
    print("--- Starting Airflow Prediction Experiment (Regression) ---")
    # ... (print feature params, selected raw features, models) ...
    print(
        f"--- Feature Params: FPS={FPS_FOR_FEATURES}, InitialFocus={FOCUS_DURATION_SEC_FOR_FEATURES}s, AreaThresh={THRESHOLD_ABS_CHANGE_FOR_AREA}°C ---")
    print(
        f"--- RAW Features Selected from Mask: {SELECTED_FEATURE_NAMES_FROM_MASK} ---")
    
    available_models = modeling.get_regressors() # Get models once
    print(f"--- Models: {list(available_models.keys())} ---")


    # 1. Find Data Files and Corresponding Masks
    # ... (your existing file finding logic - ensure it uses config.DATASET_FOLDER) ...
    all_samples_info = []
    input_dataset_dir = config.DATASET_FOLDER
    if not input_dataset_dir or not os.path.isdir(input_dataset_dir):
        print(f"Error: Dataset folder not found: {input_dataset_dir}")
        return
    print(f"\nScanning for .mat files in: {input_dataset_dir}")
    file_count, skipped_meta_count, skipped_mask_count = 0, 0, 0
    for root, dirs, files in os.walk(input_dataset_dir): # Added dirs
        # --- MODIFICATION TO SKIP 'cooling' FOLDERS (if you want this permanently) ---
        if "cooling" in dirs:
            # print(f"  Found 'cooling' in {dirs} for root {root}, removing.")
            dirs.remove("cooling")
        # --- END OF MODIFICATION ---
        folder_name = os.path.basename(root)
        # No need to skip folder_name if it's "cooling" itself, as os.walk won't list .mat files there if we removed it from `dirs` of its parent.
        for mat_filename in sorted(fnmatch.filter(files, '*.mat')):
            # ... (rest of your sample info collection) ...
            file_count += 1
            mat_filepath = os.path.join(root, mat_filename)
            relative_dir = os.path.relpath(root, input_dataset_dir)
            if relative_dir == ".": relative_dir = "" 
            mat_filename_no_ext = os.path.splitext(mat_filename)[0]
            mask_filename = mat_filename_no_ext + '_mask.npy'
            
            mask_path = os.path.join(PRECOMPUTED_MASK_DIR, relative_dir, mask_filename)
            if not os.path.exists(mask_path):
                mask_path_alt = os.path.join(PRECOMPUTED_MASK_DIR, relative_dir, mat_filename_no_ext, mask_filename)
                if os.path.exists(mask_path_alt): mask_path = mask_path_alt
            
            if not os.path.exists(mask_path):
                skipped_mask_count += 1
                continue
            try:
                airflow_rate = data_utils.parse_airflow_rate(folder_name)
                delta_T = data_utils.parse_delta_T(mat_filename)
                if delta_T is None: skipped_meta_count += 1; continue
            except ValueError: skipped_meta_count += 1; continue
            except Exception: skipped_meta_count +=1; continue
            all_samples_info.append({"mat_filepath": mat_filepath, "mask_path": mask_path, "delta_T": float(delta_T), 
                                     "airflow_rate": float(airflow_rate), "source_folder_name": folder_name, 
                                     "mat_filename_no_ext": mat_filename_no_ext})


    print(f"\nFound {file_count} .mat files. Skipped Meta: {skipped_meta_count}, Skipped Mask: {skipped_mask_count}. Proceeding with {len(all_samples_info)} samples.")
    if not all_samples_info:
        print("Error: No valid samples found.")
        return

    # 2. Extract Features using Masks
    # ... (your existing feature extraction logic) ...
    feature_list = []
    print("\n--- Extracting Features ---")
    start_time = time.time()
    processed_samples, feature_error_count = 0, 0 # Renamed from processed_samples_feat
    for i, sample_info in enumerate(tqdm(all_samples_info, desc="Extracting Features", ncols=100)):
        # ... (call feature_engineering.extract_features_with_mask) ...
        # ... (populate combined_features and append to feature_list) ...
        try:
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
            combined_features = {name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES}
            combined_features["delta_T"] = sample_info["delta_T"]
            combined_features["airflow_rate"] = sample_info["airflow_rate"]

            if extracted_features_dict and isinstance(extracted_features_dict, dict):
                has_valid_selected_feature = False
                for key, value in extracted_features_dict.items():
                    if key in combined_features:
                        combined_features[key] = value
                        if key in SELECTED_FEATURE_NAMES_FROM_MASK and np.isfinite(value):
                            has_valid_selected_feature = True
                if not SELECTED_FEATURE_NAMES_FROM_MASK or has_valid_selected_feature: # If no specific selection, or if selected are valid
                    feature_list.append(combined_features)
                    processed_samples += 1
                else: feature_error_count += 1
            else: feature_error_count += 1
        except Exception as e_feat_main:
            print(f"  Critical error during feature extraction for {sample_info['mat_filename_no_ext']} in main: {e_feat_main}")
            feature_error_count +=1


    end_time = time.time()
    print(f"\nFeature extraction finished. Processed: {processed_samples}, Errors/Skipped: {feature_error_count}. Took: {end_time - start_time:.2f}s.")
    if not feature_list:
        print("Error: No feature rows generated.")
        return

    # 3. Create DataFrame `df` (this will be the one with all features before final X selection)
    df = pd.DataFrame(feature_list)
    df = df.infer_objects()
    print(f"\nDataFrame with ALL extracted features shape: {df.shape}")

    # --- Prepare y (Target Variable) ---
    # ... (your y preparation logic) ...
    if "airflow_rate" not in df.columns: print("Error: 'airflow_rate' column missing."); return
    if df["airflow_rate"].isnull().any():
        print(f"Warning: Found {df['airflow_rate'].isnull().sum()} NaN values in target. Dropping rows.")
        df = df.dropna(subset=["airflow_rate"])
    if df.empty: print("Error: All rows dropped after handling target NaNs."); return
    y = df["airflow_rate"].astype(float)


    # --- Construct Final X (DataFrame for model input) ---
    # ... (your X_final construction logic based on SELECTED_FEATURE_NAMES_FROM_MASK and transformations) ...
    print(f"\nConstructing final feature set for model input X...")
    X_final_for_model = pd.DataFrame(index=df.index) # Ensure X_final aligns with y and df
    final_feature_names_for_model_list = []

    # a) Process delta_T
    if 'delta_T' in df.columns:
        if LOG_TRANSFORM_DELTA_T:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                X_final_for_model['delta_T_log'] = np.log1p(df['delta_T'].astype(float))
            final_feature_names_for_model_list.append('delta_T_log')
            # print("  Added: delta_T_log to X")
        else:
            X_final_for_model['delta_T'] = df['delta_T']
            final_feature_names_for_model_list.append('delta_T')
            # print("  Added: delta_T (raw) to X")
    
    # b) Process hotspot_area
    if 'hotspot_area' in SELECTED_FEATURE_NAMES_FROM_MASK and 'hotspot_area' in df.columns:
        if LOG_TRANSFORM_AREA:
            X_final_for_model['hotspot_area_log'] = np.log1p(df['hotspot_area'].astype(float).clip(lower=0))
            final_feature_names_for_model_list.append('hotspot_area_log')
            # print("  Added: hotspot_area_log to X")
        else:
            X_final_for_model['hotspot_area'] = df['hotspot_area']
            final_feature_names_for_model_list.append('hotspot_area')
            # print("  Added: hotspot_area (raw) to X")

    # c) Process hotspot_avg_temp_change_rate_initial
    if 'hotspot_avg_temp_change_rate_initial' in SELECTED_FEATURE_NAMES_FROM_MASK and \
       'hotspot_avg_temp_change_rate_initial' in df.columns and 'delta_T' in df.columns:
        if NORMALIZE_AVG_RATE_INITIAL:
            X_final_for_model['hotspot_avg_temp_change_rate_initial_norm'] = df.apply(
                lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] 
                if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_rate_initial']) and np.isfinite(r['delta_T']) 
                else np.nan, axis=1)
            final_feature_names_for_model_list.append('hotspot_avg_temp_change_rate_initial_norm')
            # print("  Added: hotspot_avg_temp_change_rate_initial_norm to X")
        else:
            X_final_for_model['hotspot_avg_temp_change_rate_initial'] = df['hotspot_avg_temp_change_rate_initial']
            final_feature_names_for_model_list.append('hotspot_avg_temp_change_rate_initial')
            # print("  Added: hotspot_avg_temp_change_rate_initial (raw) to X")
    
    # d) Process other selected raw features
    direct_features_to_add_to_X = ['temp_max_overall_initial', 'overall_std_deltaT'] # Example
    for feat_name_direct in direct_features_to_add_to_X:
        if feat_name_direct in SELECTED_FEATURE_NAMES_FROM_MASK and feat_name_direct in df.columns:
            X_final_for_model[feat_name_direct] = df[feat_name_direct]
            final_feature_names_for_model_list.append(feat_name_direct)
            # print(f"  Added: {feat_name_direct} (raw) to X")
    
    unique_final_features_for_X = sorted(list(set(final_feature_names_for_model_list)))
    X = X_final_for_model[unique_final_features_for_X].copy()
    print(f"Final Features for Model (X): {list(X.columns)}")


    # --- Handle NaNs/Infs and Final Checks ---
    # ... (your existing checks for X and y) ...
    if X.isnull().values.any() or np.isinf(X.values).any():
        print(f"\nWarning: NaN/Inf detected in final X BEFORE imputation. Rows with NaNs will be affected by imputer.")
    # ... (rest of checks)
    if X.empty or y.empty or X.shape[1] == 0 or X.shape[0] != len(y):
        print("Error: Critical issue with final X or y shapes or content. Exiting."); return
    print(f"Final Feature matrix X shape: {X.shape}, Final Target vector y shape: {y.shape}")


    # --- SAVE FEATURES AND TARGET (using config paths) ---
    if not X.empty and not y.empty:
        try:
            os.makedirs(config.FEATURES_SAVE_DIR, exist_ok=True)
            x_save_path = os.path.join(config.FEATURES_SAVE_DIR, "model_input_features_X.csv")
            X.to_csv(x_save_path, index=False)
            print(f"\nModel input features (X) saved to: {x_save_path}")

            y_save_path = os.path.join(config.FEATURES_SAVE_DIR, "target_variable_y.csv")
            y.to_csv(y_save_path, header=['airflow_rate'], index=False)
            print(f"Target variable (y) saved to: {y_save_path}")

            # Save the DataFrame `df` which contains all raw extracted features and target for visualization flexibility
            all_features_df_save_path = os.path.join(config.FEATURES_SAVE_DIR, "all_extracted_features_for_viz_with_target.csv")
            df.to_csv(all_features_df_save_path, index=False) # df has all raw features from feature_list
            print(f"All extracted features (from feature_list) with target saved to: {all_features_df_save_path}")

        except Exception as e_save:
            print(f"Error saving features/target: {e_save}"); traceback.print_exc()
    # --- END OF SAVE FEATURES ---

    # --- Steps 4-8 (Tuning, Evaluation, Plotting, Saving the model) ---
    print("\n--- Running Hyperparameter Tuning ---")
    # Use available_models fetched earlier
    # print(f"Regressors for tuning: {list(available_models.keys())}")
    all_best_params, all_best_scores = tuning.run_grid_search_all_models(X, y)
    # ... (print tuning results summary) ...
    print("\n--- Tuning Results Summary ---")
    for name_model_tuned, score_tuned in all_best_scores.items():
        mse_tuned = -score_tuned if np.isfinite(score_tuned) else np.inf
        rmse_tuned = np.sqrt(mse_tuned) if np.isfinite(mse_tuned) else np.inf
        params_tuned = all_best_params.get(name_model_tuned, {})
        print(f"{name_model_tuned}: Best CV Score (neg_mse)={score_tuned:.4f} (MSE={mse_tuned:.4f}, RMSE={rmse_tuned:.4f}), Best Params={params_tuned}")


    # CV Setup
    # ... (your existing CV setup logic using config.CV_METHOD, config.K_FOLDS, config.RANDOM_STATE) ...
    num_samples_final = len(y)
    cv_strategy_main = None; cv_name_main = "Unknown"; n_splits_main = 0
    if config.CV_METHOD == 'LeaveOneOut':
        cv_strategy_main = LeaveOneOut(); n_splits_main = num_samples_final; cv_name_main = "LeaveOneOut"
    elif config.CV_METHOD == 'KFold':
        k_folds_val = getattr(config, 'K_FOLDS', 3)
        n_splits_main = min(k_folds_val, num_samples_final)
        if n_splits_main < 2:
            cv_strategy_main = LeaveOneOut(); n_splits_main = num_samples_final; cv_name_main = "LeaveOneOut (KFold Fallback)"
        else:
            cv_strategy_main = KFold(n_splits=n_splits_main, shuffle=True, random_state=config.RANDOM_STATE)
            cv_name_main = f"{n_splits_main}-Fold KFold"
    else:
        cv_strategy_main = LeaveOneOut(); n_splits_main = num_samples_final; cv_name_main = "LeaveOneOut (Default)"
    print(f"\nUsing {cv_name_main} Cross-Validation ({n_splits_main} splits) for generalization performance evaluation.")


    # Evaluation
    print(f"\n--- Evaluating ALL Models using Best Parameters (via Cross-Validation) ---")
    evaluation_results = {}
    # all_regressors_to_eval = modeling.get_regressors() # Already fetched as available_models

    for name, model_prototype_eval in available_models.items():
        print(f"\n===== Processing Model for CV: {name} =====")
        # It's crucial to get a fresh, untuned instance or clone the prototype for CV evaluation
        # If get_regressors() returns new instances each time, this is fine.
        # Otherwise, use sklearn.base.clone(model_prototype_eval)
        current_model_for_cv_eval = modeling.get_regressors()[name] # Get fresh instance
        
        tuned_params_for_cv = {k.replace('model__', ''): v for k, v in all_best_params.get(name, {}).items()}

        if tuned_params_for_cv and name in all_best_params and np.isfinite(all_best_scores.get(name, -np.inf)):
            print(f"  Applying Tuned Parameters for {name} (CV): {tuned_params_for_cv}")
            try: current_model_for_cv_eval.set_params(**tuned_params_for_cv)
            except ValueError as e_set_cv: print(f"  Warning: Error setting tuned params for {name} (CV): {e_set_cv}. Using defaults.")
        else: print(f"  Using default parameters for {name} (CV).")

        pipeline_model_for_cv = modeling.build_pipeline(current_model_for_cv_eval, pca_components=config.PCA_N_COMPONENTS)

        try:
            with warnings.catch_warnings():
                if isinstance(current_model_for_cv_eval, MLPRegressor):
                    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.neural_network._multilayer_perceptron")
                y_pred_cv_vals = cross_val_predict(pipeline_model_for_cv, X, y, cv=cv_strategy_main, n_jobs=-1)
            
            if np.isnan(y_pred_cv_vals).any():
                y_pred_cv_vals = np.nan_to_num(y_pred_cv_vals, nan=np.nanmean(y))
            
            mse_cv_val = mean_squared_error(y, y_pred_cv_vals)
            rmse_cv_val = np.sqrt(mse_cv_val)
            mae_cv_val = mean_absolute_error(y, y_pred_cv_vals)
            r2_cv_val = r2_score(y, y_pred_cv_vals)
            results_dict_cv_current = {'mse': mse_cv_val, 'rmse': rmse_cv_val, 'mae': mae_cv_val, 'r2': r2_cv_val, 'y_pred_cv': y_pred_cv_vals}
            evaluation_results[name] = results_dict_cv_current
            print(f"--- CV Results for {name} ---")
            print(f"MSE: {mse_cv_val:.4f} | RMSE: {rmse_cv_val:.4f} | MAE: {mae_cv_val:.4f} | R²: {r2_cv_val:.4f}")
        except Exception as e_cv_main:
            print(f"  Error during CV for {name}: {e_cv_main}"); traceback.print_exc()
            evaluation_results[name] = None; continue

        # Plot Actual vs. Predicted for CV
        if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True) and evaluation_results[name] is not None:
            # ... (your plotting logic for CV, using config.OUTPUT_DIR) ...
            plot_title_cv_main = f'Actual vs. Predicted - {name} (CV, Tuned*)'
            plot_save_path_cv_main = os.path.join(config.OUTPUT_DIR, f"actual_vs_predicted_CV_{name}.png")
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            print(f"Saving CV Actual vs Predicted plot for {name} to: {plot_save_path_cv_main}")
            try:
                plotting.plot_actual_vs_predicted(y, results_dict_cv_current['y_pred_cv'], plot_title_cv_main, plot_save_path_cv_main,
                                                  scatter_kwargs={'s': 50, 'alpha': 0.8}) # Example kwargs
            except Exception as e_plot_cv_main: print(f"  Error plotting CV Actual vs Predicted for {name}: {e_plot_cv_main}")


        # --- (Moved the Overfitting Check / Training Set Evaluation HERE, inside the loop) ---
        if config.EVALUATE_ON_FULL_TRAINING_SET:
            print(f"\n--- Training Set Performance for {name} (Overfitting Check) ---")
            try:
                # Use the same model instance (current_model_for_cv_eval) which already has tuned parameters set
                pipeline_for_train_eval = modeling.build_pipeline(current_model_for_cv_eval, pca_components=config.PCA_N_COMPONENTS)
                
                with warnings.catch_warnings():
                    if isinstance(current_model_for_cv_eval, MLPRegressor):
                        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.neural_network._multilayer_perceptron")
                    pipeline_for_train_eval.fit(X, y) # Fit on the full training set
                
                y_pred_train = pipeline_for_train_eval.predict(X)

                mse_train_set = mean_squared_error(y, y_pred_train)
                rmse_train_set = np.sqrt(mse_train_set)
                mae_train_set = mean_absolute_error(y, y_pred_train)
                r2_train_set = r2_score(y, y_pred_train)

                print(f"  R² (Training Set): {r2_train_set:.4f}  (vs CV R²: {r2_cv_val:.4f})")
                print(f"  RMSE (Training Set): {rmse_train_set:.4f} (vs CV RMSE: {rmse_cv_val:.4f})")
                # print(f"  MSE (Training Set): {mse_train_set:.4f}")
                # print(f"  MAE (Training Set): {mae_train_set:.4f}")

                if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True):
                    plot_title_train_set = f'Actual vs. Predicted - {name} (TRAINING_SET_FIT)'
                    train_set_plot_save_path = os.path.join(config.OUTPUT_DIR, f"actual_vs_predicted_TRAINING_SET_{name}.png")
                    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
                    plotting.plot_actual_vs_predicted(
                        y, y_pred_train, plot_title_train_set, train_set_plot_save_path,
                        scatter_kwargs={'s': 30, 'alpha': 0.6, 'edgecolors': 'gray'}, add_jitter=True, jitter_strength=0.005
                    )
                    print(f"  Training set fit plot saved to: {train_set_plot_save_path}")

            except Exception as e_train_set_eval:
                print(f"  Error during full training set evaluation for {name}: {e_train_set_eval}"); traceback.print_exc()
        # --- (End of Overfitting Check section) ---

        # Permutation Importance (based on the model fitted on the full training set during the overfitting check)
        if getattr(config, 'SAVE_PERMUTATION_IMPORTANCE_PLOT', True) and evaluation_results[name] is not None:
            if config.EVALUATE_ON_FULL_TRAINING_SET and 'pipeline_for_train_eval' in locals(): # Check if it was created
                print(f"\nCalculating Permutation Importance for {name} (using model from training set eval)...")
                try:
                    perm_result_main = permutation_importance(
                        pipeline_for_train_eval, X, y, n_repeats=15, random_state=config.RANDOM_STATE, n_jobs=-1, scoring='r2')
                    # ... (rest of your permutation importance plotting logic, using config.OUTPUT_DIR) ...
                    if hasattr(perm_result_main, 'importances_mean'):
                        sorted_idx_perm = perm_result_main.importances_mean.argsort()
                        importances_df_perm = pd.DataFrame(perm_result_main.importances[sorted_idx_perm].T, columns=X.columns[sorted_idx_perm])
                        plt.figure(figsize=(10, max(6, len(X.columns)*0.4)))
                        plt.boxplot(importances_df_perm.values, labels=importances_df_perm.columns, vert=False, whis=10)
                        plt.title(f"Permutation Importances for {name} (Fitted on All Data)")
                        plt.axvline(x=0, color="k", linestyle="--")
                        plt.xlabel("Decrease in R² score")
                        plt.tight_layout()
                        perm_plot_path_main = os.path.join(config.OUTPUT_DIR, f"permutation_importance_{name}.png")
                        plt.savefig(perm_plot_path_main, dpi=150); plt.close()
                        print(f"  Permutation importance plot for {name} saved to: {perm_plot_path_main}")
                        print(f"  Feature Importances (Mean Decrease in R²):")
                        for i_perm in sorted_idx_perm[::-1]: print(f"    {X.columns[i_perm]:<35}: {perm_result_main.importances_mean[i_perm]:.4f} +/- {perm_result_main.importances_std[i_perm]:.4f}")
                    else: print(f"  Warning: Permutation importance failed for {name}.")
                except Exception as e_perm_main: print(f"  Error in Permutation Importance for {name}: {e_perm_main}"); traceback.print_exc()
            else:
                print(f"  Skipping Permutation Importance for {name} as full training set evaluation was not run or pipeline not available.")


    # 7. Analyze Overall Best Model (based on CV R²)
    # ... (your existing logic for finding and printing best model) ...
    valid_results_final = {name_res: res_val for name_res, res_val in evaluation_results.items() if res_val is not None and 'r2' in res_val and np.isfinite(res_val['r2'])}
    if not valid_results_final: print("\nCRITICAL Error: All model CV evaluations failed."); return
    
    best_model_name_final = max(valid_results_final, key=lambda name_res: valid_results_final[name_res]['r2'])
    best_result_metrics_final = valid_results_final[best_model_name_final]
    print(f"\n--- OVERALL Best Model based on CV R²: {best_model_name_final} ---")
    print(f"CV MSE: {best_result_metrics_final['mse']:.4f} | CV RMSE: {best_result_metrics_final['rmse']:.4f} | CV MAE: {best_result_metrics_final['mae']:.4f} | CV R²: {best_result_metrics_final['r2']:.4f}")


    # 8. Final Training and Saving of the OVERALL BEST Model
    # ... (your existing logic for training and saving the best model, using config paths) ...
    print(f"\n--- Training final OVERALL BEST model ({best_model_name_final}) on all data for deployment/saving ---")
    try:
        final_model_instance_deploy = modeling.get_regressors()[best_model_name_final]
        best_params_deploy_raw = all_best_params.get(best_model_name_final, {})
        best_params_deploy_final = {k.replace('model__', ''): v for k, v in best_params_deploy_raw.items()}

        if best_params_deploy_final and best_model_name_final in all_best_params:
            print(f"  Applying Tuned Parameters: {best_params_deploy_final}")
            final_model_instance_deploy.set_params(**best_params_deploy_final)
        else: print(f"  Using default parameters for final {best_model_name_final} model.")

        final_pipeline_deploy = modeling.build_pipeline(final_model_instance_deploy, pca_components=config.PCA_N_COMPONENTS)
        with warnings.catch_warnings():
            if isinstance(final_model_instance_deploy, MLPRegressor):
                warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.neural_network._multilayer_perceptron")
            final_pipeline_deploy.fit(X, y)
        print("Overall best model final training complete.")

        if getattr(config, 'SAVE_LOSS_CURVE_PLOT', False) and isinstance(final_pipeline_deploy.named_steps.get('model'), MLPRegressor):
            mlp_final_step = final_pipeline_deploy.named_steps.get('model')
            if hasattr(mlp_final_step, 'loss_curve_'):
                print(f"\nPlotting Training Loss Curve for OVERALL BEST MLP ({best_model_name_final})...")
                loss_fname = f"final_loss_curve_{best_model_name_final}.png"
                plotting.plot_loss_curve(mlp_final_step.loss_curve_, f"MLP Loss ({best_model_name_final} - Final Fit)", config.OUTPUT_DIR, filename=loss_fname)
            else: print(f"  Note: Loss curve not available for {best_model_name_final} (final fit).")
        
        final_model_save_path_deploy = config.MODEL_SAVE_PATH.format(model_name=best_model_name_final)
        os.makedirs(os.path.dirname(final_model_save_path_deploy), exist_ok=True)
        joblib.dump(final_pipeline_deploy, final_model_save_path_deploy)
        print(f"\nFinal OVERALL BEST model ({best_model_name_final}) pipeline saved to: {final_model_save_path_deploy}")

    except Exception as e_final_deploy: print(f"  Error during final training/saving of {best_model_name_final}: {e_final_deploy}"); traceback.print_exc()


    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    run_experiment()