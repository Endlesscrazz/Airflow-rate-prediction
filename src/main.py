# main.py
"""
Main script to run airflow prediction for a COMBINED dataset of multiple materials,
using one-hot encoding for material type.
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
# import argparse 
# import sys     

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
# from sklearn.linear_model import LinearRegression, Ridge # Keep if used by get_regressors
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*SettingWithCopyWarning*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.neural_network._multilayer_perceptron") # Global for MLP

# --- Configuration Values from config.py ---
THRESHOLD_ABS_CHANGE_FOR_AREA = getattr(config, 'FIXED_AREA_THRESHOLD', 0.5)
FPS_FOR_FEATURES = getattr(config, 'MASK_FPS', 5.0)
FOCUS_DURATION_SEC_FOR_FEATURES = getattr(config, 'MASK_FOCUS_DURATION_SEC', 5.0)
ENVIR_PARA = getattr(config, 'MASK_ENVIR_PARA', 1) # Assumed globally consistent for now

ALL_POSSIBLE_FEATURE_NAMES = [
    'hotspot_area', 'hotspot_avg_temp_change_rate_initial', 
    'hotspot_avg_temp_change_magnitude_initial', 'peak_pixel_temp_change_rate_initial',
    'peak_pixel_temp_change_magnitude_initial', 'temp_mean_avg_initial', 
    'temp_std_avg_initial', 'temp_min_overall_initial', 'temp_max_overall_initial',
    'stabilized_mean_deltaT', 'overall_mean_deltaT', 'max_abs_mean_deltaT',
    'stabilized_std_deltaT', 'overall_std_deltaT', 'mean_area_significant_change',
    'stabilized_area_significant_change', 'max_area_significant_change',
]

# Define the feature set you want to use for the combined model
SELECTED_FEATURE_NAMES_FROM_MASK_COMBINED = [
    'hotspot_area',
    'hotspot_avg_temp_change_rate_initial',
    'temp_max_overall_initial',
    'temp_std_avg_initial',     
    'overall_std_deltaT'      
]

# Transformation Flags (ensure these are defined in config or here)
NORMALIZE_AVG_RATE_INITIAL = getattr(config, 'NORMALIZE_AVG_RATE_INITIAL', True)
LOG_TRANSFORM_DELTA_T = getattr(config, 'LOG_TRANSFORM_DELTA_T', True)
LOG_TRANSFORM_AREA = getattr(config, 'LOG_TRANSFORM_AREA', True)
# NORMALIZE_AVG_MAG_INITIAL = getattr(config, 'NORMALIZE_AVG_MAG_INITIAL', True) # If used


def run_combined_experiment():
    experiment_name = "combined_materials"
    combined_output_dir = os.path.join(config.BASE_OUTPUT_DIR, experiment_name)
    os.makedirs(combined_output_dir, exist_ok=True)
    combined_features_save_dir = os.path.join(combined_output_dir, "saved_features")
    os.makedirs(combined_features_save_dir, exist_ok=True)
    combined_trained_models_dir = os.path.join(config.BASE_TRAINED_MODELS_DIR, experiment_name)
    os.makedirs(combined_trained_models_dir, exist_ok=True)
    combined_best_model_dir = os.path.join(combined_trained_models_dir, "best_model")
    os.makedirs(combined_best_model_dir, exist_ok=True)

    print(f"--- Starting COMBINED Experiment for Multiple Materials ---")
    print(f"--- Parent Dataset Directory: {config.DATASET_PARENT_DIR} ---")
    print(f"--- Parent Mask Directory: {config.BASE_MASK_INPUT_DIR} ---")
    print(f"--- Outputs for this combined run: {combined_output_dir} ---")
    print(f"--- RAW Features Selected: {SELECTED_FEATURE_NAMES_FROM_MASK_COMBINED} ---")
    
    available_models = modeling.get_regressors()
    print(f"--- Models: {list(available_models.keys())} ---")

    all_samples_info_combined = []
    material_configs = {
        "gypsum": {
            "dataset_subfolder": "dataset_gypsum",
            "mask_sub_path": "slope_p01_focus5s_q99_env1_roi0_gypsum" # Specific mask subfolder for gypsum
        },
        "brick_cladding": {
            "dataset_subfolder": "dataset_brickcladding",
            "mask_sub_path": "slope_01_focus5s_q99_env1_roi10_mpOpenClose" # Specific mask subfolder for brick
        }
    }

    overall_file_count, overall_skipped_meta, overall_skipped_mask = 0, 0, 0
    for material_name, mat_conf in material_configs.items():
        print(f"\nScanning for material: {material_name}")
        current_dataset_dir = os.path.join(config.DATASET_PARENT_DIR, mat_conf["dataset_subfolder"])
        # Construct full path to the specific mask directory for this material configuration
        current_mask_root_dir = os.path.join(config.BASE_MASK_INPUT_DIR, material_name, mat_conf["mask_sub_path"])
        
        if not os.path.isdir(current_dataset_dir):
            print(f"  Warning: Dataset directory for {material_name} not found: {current_dataset_dir}. Skipping."); continue
        if not os.path.isdir(current_mask_root_dir):
            print(f"  CRITICAL Warning: Mask directory for {material_name} not found: {current_mask_root_dir}. Skipping this material."); continue

        file_count_mat, skipped_meta_mat, skipped_mask_mat = 0, 0, 0
        for root, dirs, files in os.walk(current_dataset_dir):
            if "cooling" in dirs: dirs.remove("cooling")
            folder_name = os.path.basename(root)
            for mat_filename in sorted(fnmatch.filter(files, '*.mat')):
                file_count_mat += 1; mat_filepath = os.path.join(root, mat_filename)
                relative_dir_for_mask_lookup = os.path.relpath(root, current_dataset_dir)
                if relative_dir_for_mask_lookup == ".": relative_dir_for_mask_lookup = ""
                
                mat_filename_no_ext = os.path.splitext(mat_filename)[0]
                mask_filename = mat_filename_no_ext + '_mask.npy'
                mask_path = os.path.join(current_mask_root_dir, relative_dir_for_mask_lookup, mask_filename)
                
                if not os.path.exists(mask_path): skipped_mask_mat += 1; continue
                try:
                    airflow_rate = data_utils.parse_airflow_rate(folder_name)
                    delta_T = data_utils.parse_delta_T(mat_filename)
                    if delta_T is None: skipped_meta_mat += 1; continue
                except ValueError: skipped_meta_mat += 1; continue
                except Exception: skipped_meta_mat +=1; continue
                
                all_samples_info_combined.append({
                    "mat_filepath": mat_filepath, "mask_path": mask_path, 
                    "delta_T": float(delta_T), "airflow_rate": float(airflow_rate),
                    "material": material_name, # ADDED MATERIAL TYPE
                    "source_folder_name": folder_name, "mat_filename_no_ext": mat_filename_no_ext
                })
        print(f"  Found {file_count_mat} .mat files for {material_name}. Meta skipped: {skipped_meta_mat}, Mask skipped: {skipped_mask_mat}.")
        overall_file_count += file_count_mat; overall_skipped_meta += skipped_meta_mat; overall_skipped_mask += skipped_mask_mat

    print(f"\nTotal .mat files scanned: {overall_file_count}. Total Meta skipped: {overall_skipped_meta}, Total Mask skipped: {overall_skipped_mask}.")
    print(f"Proceeding with {len(all_samples_info_combined)} samples for combined model.")
    if not all_samples_info_combined: print("Error: No valid samples found from any material."); return

    feature_list_combined = []
    print("\n--- Extracting Features for Combined Dataset ---")
    start_time = time.time()
    processed_samples_cb, feature_error_count_cb = 0,0
    for sample_info in tqdm(all_samples_info_combined, desc="Extracting Features (Combined)", ncols=100):
        try:
            extracted_features_dict = feature_engineering.extract_features_with_mask(
                frames_or_path=sample_info["mat_filepath"], mask_path=sample_info["mask_path"],
                fps=FPS_FOR_FEATURES, focus_duration_sec=FOCUS_DURATION_SEC_FOR_FEATURES,
                envir_para=ENVIR_PARA, threshold_abs_change=THRESHOLD_ABS_CHANGE_FOR_AREA,
                source_folder_name=sample_info["source_folder_name"], mat_filename_no_ext=sample_info["mat_filename_no_ext"]
            )
            current_combined_features = {name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES}
            current_combined_features["delta_T"] = sample_info["delta_T"]
            current_combined_features["airflow_rate"] = sample_info["airflow_rate"]
            current_combined_features["material"] = sample_info["material"]

            if extracted_features_dict:
                has_valid_sel = False
                for key, value in extracted_features_dict.items():
                    if key in current_combined_features: current_combined_features[key] = value
                    if key in SELECTED_FEATURE_NAMES_FROM_MASK_COMBINED and np.isfinite(value): has_valid_sel = True
                if not SELECTED_FEATURE_NAMES_FROM_MASK_COMBINED or has_valid_sel:
                    feature_list_combined.append(current_combined_features); processed_samples_cb += 1
                else: feature_error_count_cb += 1
            else: feature_error_count_cb += 1
        except Exception as e_feat_cb: feature_error_count_cb += 1; print(f"Err feat eng for {sample_info['mat_filename_no_ext']}: {e_feat_cb}"); traceback.print_exc()
    end_time = time.time()
    print(f"\nCombined feature extraction. Processed: {processed_samples_cb}, Errors/Skipped: {feature_error_count_cb}. Took: {end_time - start_time:.2f}s.")
    if not feature_list_combined: print("Error: No feature rows for combined dataset."); return
    
    df_combined = pd.DataFrame(feature_list_combined); df_combined = df_combined.infer_objects()
    print(f"Combined DataFrame shape (before target NaN drop): {df_combined.shape}")

    if "airflow_rate" not in df_combined.columns: print("Error: 'airflow_rate' missing."); return
    if df_combined["airflow_rate"].isnull().any(): df_combined.dropna(subset=["airflow_rate"], inplace=True)
    if df_combined.empty: print("Error: Combined DataFrame empty after target NaN drop."); return
    y_combined = df_combined["airflow_rate"].astype(float)

    print(f"\nConstructing final feature set X for combined model...")
    X_final_combined = pd.DataFrame(index=df_combined.index) # Use index from df_combined AFTER potential NaN drop
    final_feature_names_for_combined_model = []

    # a) Process delta_T
    if 'delta_T' in df_combined.columns:
        if LOG_TRANSFORM_DELTA_T:
            with warnings.catch_warnings(): warnings.simplefilter("ignore", category=RuntimeWarning)
            X_final_combined['delta_T_log'] = np.log1p(df_combined['delta_T'].astype(float).clip(lower=0)) # Clip to handle potential negatives if any
            final_feature_names_for_combined_model.append('delta_T_log')
        else: X_final_combined['delta_T'] = df_combined['delta_T']; final_feature_names_for_combined_model.append('delta_T')
    
    # b) Process SELECTED_FEATURE_NAMES_FROM_MASK_COMBINED
    for feat_name in SELECTED_FEATURE_NAMES_FROM_MASK_COMBINED:
        if feat_name in df_combined.columns:
            # Handle specific transformations if needed for these selected features
            if feat_name == 'hotspot_area' and LOG_TRANSFORM_AREA:
                X_final_combined['hotspot_area_log'] = np.log1p(df_combined['hotspot_area'].astype(float).clip(lower=0))
                final_feature_names_for_combined_model.append('hotspot_area_log')
            elif feat_name == 'hotspot_avg_temp_change_rate_initial' and NORMALIZE_AVG_RATE_INITIAL and 'delta_T' in df_combined.columns:
                X_final_combined['hotspot_avg_temp_change_rate_initial_norm'] = df_combined.apply(
                    lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T']!=0 and np.isfinite(r['hotspot_avg_temp_change_rate_initial']) and np.isfinite(r['delta_T']) else np.nan, axis=1)
                final_feature_names_for_combined_model.append('hotspot_avg_temp_change_rate_initial_norm')
            # Add other specific transformations here if a selected feature needs it
            # else, add raw
            elif feat_name not in X_final_combined.columns: # Avoid re-adding if transformed version was already added
                X_final_combined[feat_name] = df_combined[feat_name]
                final_feature_names_for_combined_model.append(feat_name)
        else:
            print(f"Warning: Selected feature '{feat_name}' not found in extracted df_combined columns.")


    # c) Add One-Hot Encoded Material Feature
    if 'material' in df_combined.columns:
        material_dummies = pd.get_dummies(df_combined['material'], prefix='material', dtype=int)
        # Ensure indices match before concat
        X_final_combined = pd.concat([X_final_combined, material_dummies.set_index(X_final_combined.index)], axis=1)
        for dummy_col in material_dummies.columns:
            final_feature_names_for_combined_model.append(dummy_col)
        print(f"  Added One-Hot Encoded material features: {list(material_dummies.columns)}")

    unique_final_features_cb = sorted(list(set(final_feature_names_for_combined_model)))
    if not unique_final_features_cb: print("Error: No features selected for combined model X."); return
    X_combined = X_final_combined[unique_final_features_cb].copy()
    print(f"Final Features for Combined Model (X): {list(X_combined.columns)}")
    
    if X_combined.isnull().values.any(): print(f"\nWarning: NaNs found in X_combined BEFORE imputation. Count: {X_combined.isnull().sum().sum()}")
    if X_combined.empty or y_combined.empty or X_combined.shape[1]==0 or X_combined.shape[0]!=len(y_combined):
        print("Error with combined X/y shapes or content. Exiting."); return
    print(f"Combined X shape: {X_combined.shape}, Combined y shape: {y_combined.shape}")

    X_combined.to_csv(os.path.join(combined_features_save_dir, "model_input_features_X_COMBINED.csv"), index=False)
    y_combined.to_csv(os.path.join(combined_features_save_dir, "target_variable_y_COMBINED.csv"), header=['airflow_rate'], index=False)
    df_combined.to_csv(os.path.join(combined_features_save_dir, "all_extracted_features_COMBINED_for_viz.csv"), index=False)
    print(f"\nCombined features and target saved to: {combined_features_save_dir}")

    print(f"\n--- Running Hyperparameter Tuning for Combined Dataset ---")
    all_best_params_cb, all_best_scores_cb = tuning.run_grid_search_all_models(X_combined, y_combined)
    print(f"\n--- Tuning Results Summary for Combined Dataset ---")
    # ... (print tuning summary) ...

    # CV Setup
    # ... (CV setup logic as before, using X_combined.shape[0] for num_samples) ...
    num_samples_cb_final = len(y_combined); cv_strategy_cb = None; cv_name_cb = "Unknown"; n_splits_cb = 0
    if config.CV_METHOD == 'LeaveOneOut': cv_strategy_cb = LeaveOneOut(); n_splits_cb = num_samples_cb_final; cv_name_cb = "LeaveOneOut"
    # ... (rest of CV setup logic) ...
    elif config.CV_METHOD == 'KFold':
        k_folds_cb = getattr(config, 'K_FOLDS', 3); n_splits_cb = min(k_folds_cb, num_samples_cb_final)
        if n_splits_cb < 2: cv_strategy_cb = LeaveOneOut(); n_splits_cb = num_samples_cb_final; cv_name_cb = "LeaveOneOut (KFold Fallback)"
        else: cv_strategy_cb = KFold(n_splits=n_splits_cb, shuffle=True, random_state=config.RANDOM_STATE); cv_name_cb = f"{n_splits_cb}-Fold KFold"
    else: cv_strategy_cb = LeaveOneOut(); n_splits_cb = num_samples_cb_final; cv_name_cb = "LeaveOneOut (Default)"
    print(f"\nUsing {cv_name_cb} CV ({n_splits_cb} splits) for combined dataset.")


    print(f"\n--- Evaluating ALL Models for Combined Dataset ---")
    evaluation_results_cb = {}
    for name_cb_eval, model_prototype_cb_eval in available_models.items():
        print(f"\n===== Processing Model: {name_cb_eval} (Combined) =====")
        current_model_instance_cb_eval = modeling.get_regressors()[name_cb_eval] # Fresh instance
        tuned_params_cb_eval = {k.replace('model__', ''): v for k,v in all_best_params_cb.get(name_cb_eval, {}).items()}
        if tuned_params_cb_eval: current_model_instance_cb_eval.set_params(**tuned_params_cb_eval)
        
        pipeline_cv_cb_eval = modeling.build_pipeline(current_model_instance_cb_eval, pca_components=config.PCA_N_COMPONENTS)
        
        # CV Evaluation
        try:
            y_pred_cv_cb_vals = cross_val_predict(pipeline_cv_cb_eval, X_combined, y_combined, cv=cv_strategy_cb, n_jobs=-1)
            # ... (metrics calculation) ...
            if np.isnan(y_pred_cv_cb_vals).any(): y_pred_cv_cb_vals = np.nan_to_num(y_pred_cv_cb_vals, nan=np.nanmean(y_combined))
            mse_cb_cv = mean_squared_error(y_combined, y_pred_cv_cb_vals); rmse_cb_cv = np.sqrt(mse_cb_cv)
            mae_cb_cv = mean_absolute_error(y_combined, y_pred_cv_cb_vals); r2_cb_cv = r2_score(y_combined, y_pred_cv_cb_vals)
            evaluation_results_cb[name_cb_eval] = {'mse': mse_cb_cv, 'rmse': rmse_cb_cv, 'mae': mae_cb_cv, 'r2': r2_cb_cv, 'y_pred_cv': y_pred_cv_cb_vals}
            print(f"  CV Results: MSE={mse_cb_cv:.4f} | RMSE={rmse_cb_cv:.4f} | MAE={mae_cb_cv:.4f} | R²={r2_cb_cv:.4f}")
            if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True):
                plotting.plot_actual_vs_predicted(y_combined, y_pred_cv_cb_vals, f'Actual vs. Predicted - {name_cb_eval} (Combined CV)', 
                                                  os.path.join(combined_output_dir, f"actual_vs_predicted_CV_{name_cb_eval}_COMBINED.png"),
                                                  scatter_kwargs={'s':50, 'alpha':0.7})
        except Exception as e_cv_cb: print(f"  Error CV for {name_cb_eval} (Combined): {e_cv_cb}"); evaluation_results_cb[name_cb_eval]=None; continue

        # Training Set Evaluation (Overfitting Check)
        pipeline_fitted_all_combined = None # For permutation importance and saving individual model
        if config.EVALUATE_ON_FULL_TRAINING_SET:
            print(f"  Training Set Performance (Combined):")
            try:
                pipeline_fitted_all_combined = modeling.build_pipeline(current_model_instance_cb_eval, pca_components=config.PCA_N_COMPONENTS)
                pipeline_fitted_all_combined.fit(X_combined, y_combined)
                y_pred_train_cb = pipeline_fitted_all_combined.predict(X_combined)
                r2_train_cb = r2_score(y_combined, y_pred_train_cb); rmse_train_cb = np.sqrt(mean_squared_error(y_combined, y_pred_train_cb))
                print(f"    R² (Train): {r2_train_cb:.4f} (vs CV R²: {evaluation_results_cb.get(name_cb_eval,{}).get('r2',np.nan):.4f})")
                print(f"    RMSE (Train): {rmse_train_cb:.4f} (vs CV RMSE: {evaluation_results_cb.get(name_cb_eval,{}).get('rmse',np.nan):.4f})")
                if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True):
                    plotting.plot_actual_vs_predicted(y_combined, y_pred_train_cb, f'Actual vs. Predicted - {name_cb_eval} (Combined TRAIN_FIT)',
                                                      os.path.join(combined_output_dir, f"actual_vs_predicted_TRAIN_SET_{name_cb_eval}_COMBINED.png"),
                                                      scatter_kwargs={'s':30, 'alpha':0.6}, add_jitter=True)
            except Exception as e_train_cb: print(f"    Error training set eval for {name_cb_eval} (Combined): {e_train_cb}")

        # Save this specific tuned model (fitted on all combined data)
        if pipeline_fitted_all_combined: # If model was fitted (e.g. during training set eval)
            indiv_model_path_cb_save = os.path.join(combined_trained_models_dir, f"{name_cb_eval}_tuned_COMBINED.joblib")
            joblib.dump(pipeline_fitted_all_combined, indiv_model_path_cb_save)
            print(f"  Tuned model '{name_cb_eval}' (Combined) saved to: {indiv_model_path_cb_save}")

        # Permutation Importance
        if getattr(config, 'SAVE_PERMUTATION_IMPORTANCE_PLOT', True) and pipeline_fitted_all_combined:
            print(f"  Permutation Importance (Combined):")
            # ... (Permutation importance logic and plotting) ...
            try:
                perm_res_cb = permutation_importance(pipeline_fitted_all_combined, X_combined, y_combined, n_repeats=15, random_state=config.RANDOM_STATE, n_jobs=-1, scoring='r2')
                if hasattr(perm_res_cb, 'importances_mean'):
                    # ... (plotting logic as before) ...
                    plt_path_perm_cb = os.path.join(combined_output_dir, f"permutation_importance_{name_cb_eval}_COMBINED.png")
                    # ... (save plot) ...
                    print(f"    Permutation plot saved to {plt_path_perm_cb}")
            except Exception as e_perm_cb: print(f"    Error Permutation Importance for {name_cb_eval} (Combined): {e_perm_cb}")


    # Analyze Overall Best Model for Combined Dataset
    valid_results_cb_final = {n:r for n,r in evaluation_results_cb.items() if r and 'r2' in r and np.isfinite(r['r2'])}
    if not valid_results_cb_final: print("\nCRITICAL Error: All model CV evals failed for combined data."); return
    best_model_name_cb_final = max(valid_results_cb_final, key=lambda n: valid_results_cb_final[n]['r2'])
    best_metrics_cb_final = valid_results_cb_final[best_model_name_cb_final]
    print(f"\n--- OVERALL Best Model for Combined Dataset (based on CV R²): {best_model_name_cb_final} ---")
    print(f"CV MSE: {best_metrics_cb_final['mse']:.4f} | CV RMSE: {best_metrics_cb_final['rmse']:.4f} | CV MAE: {best_metrics_cb_final['mae']:.4f} | CV R²: {best_metrics_cb_final['r2']:.4f}")

    # Final Training and Saving of OVERALL BEST Model for Combined Dataset
    print(f"\n--- Saving OVERALL BEST model ({best_model_name_cb_final}) for Combined Dataset ---")
    try:
        final_best_model_inst_cb = modeling.get_regressors()[best_model_name_cb_final]
        best_overall_params_cb = {k.replace('model__',''):v for k,v in all_best_params_cb.get(best_model_name_cb_final,{}).items()}
        if best_overall_params_cb: final_best_model_inst_cb.set_params(**best_overall_params_cb)
        
        final_best_pipeline_cb_save = modeling.build_pipeline(final_best_model_inst_cb, pca_components=config.PCA_N_COMPONENTS)
        final_best_pipeline_cb_save.fit(X_combined, y_combined) # Fit on all combined data
        print(f"  Overall best model ({best_model_name_cb_final}) for combined data fitted.")

        overall_best_model_save_path_cb = os.path.join(combined_best_model_dir, f"BEST_{best_model_name_cb_final}_final_COMBINED.joblib")
        joblib.dump(final_best_pipeline_cb_save, overall_best_model_save_path_cb)
        print(f"  Final OVERALL BEST model for combined data saved to: {overall_best_model_save_path_cb}")

        if getattr(config, 'SAVE_LOSS_CURVE_PLOT', False) and isinstance(final_best_pipeline_cb_save.named_steps.get('model'), MLPRegressor):
            # ... (Plot loss curve for this best combined MLP) ...
            pass
    except Exception as e_final_save_cb: print(f"  Error saving BEST combined model: {e_final_save_cb}")

    print(f"\n--- COMBINED Experiment Finished ---")


if __name__ == "__main__":
    run_combined_experiment()