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
    config, 'MASK_DIR', 'hotspot_masks/slope_p01_focus5s_q99_env1_roi0')
THRESHOLD_ABS_CHANGE_FOR_AREA = getattr(config, 'FIXED_AREA_THRESHOLD', 0.5)
FPS_FOR_FEATURES = getattr(config, 'MASK_FPS', 5.0)
FOCUS_DURATION_SEC_FOR_FEATURES = getattr(
    config, 'MASK_FOCUS_DURATION_SEC', 5.0)
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
]

NORMALIZE_AVG_RATE_INITIAL = True
NORMALIZE_AVG_MAG_INITIAL = True
LOG_TRANSFORM_DELTA_T = True
LOG_TRANSFORM_AREA = True

# NORMALIZE_PEAK_RATE_INITIAL = True
# NORMALIZE_PEAK_MAG_INITIAL = True
# NORMALIZE_STABILIZED_MEAN_DELTAT = True


def run_experiment():
    print("--- Starting Airflow Prediction Experiment (Regression) ---")
    print(
        f"--- Feature Params: FPS={FPS_FOR_FEATURES}, InitialFocus={FOCUS_DURATION_SEC_FOR_FEATURES}s, AreaThresh={THRESHOLD_ABS_CHANGE_FOR_AREA}°C ---")
    print(
        f"--- RAW Features Selected from Mask: {SELECTED_FEATURE_NAMES_FROM_MASK} ---")

    available_models = modeling.get_regressors()  # Get models once
    print(f"--- Models: {list(available_models.keys())} ---")

    # 1. Find Data Files and Corresponding Masks
    all_samples_info = []
    input_dataset_dir = config.DATASET_FOLDER
    if not input_dataset_dir or not os.path.isdir(input_dataset_dir):
        print(f"Error: Dataset folder not found: {input_dataset_dir}")
        return
    print(f"\nScanning for .mat files in: {input_dataset_dir}")
    file_count, skipped_meta_count, skipped_mask_count = 0, 0, 0
    for root, dirs, files in os.walk(input_dataset_dir):
        if "cooling" in dirs:
            dirs.remove("cooling")
        folder_name = os.path.basename(root)
        for mat_filename in sorted(fnmatch.filter(files, '*.mat')):
            file_count += 1
            mat_filepath = os.path.join(root, mat_filename)
            relative_dir = os.path.relpath(root, input_dataset_dir)
            if relative_dir == ".":
                relative_dir = ""
            mat_filename_no_ext = os.path.splitext(mat_filename)[0]
            mask_filename = mat_filename_no_ext + '_mask.npy'
            mask_path = os.path.join(
                PRECOMPUTED_MASK_DIR, relative_dir, mask_filename)
            if not os.path.exists(mask_path):
                mask_path_alt = os.path.join(
                    PRECOMPUTED_MASK_DIR, relative_dir, mat_filename_no_ext, mask_filename)
                if os.path.exists(mask_path_alt):
                    mask_path = mask_path_alt
            if not os.path.exists(mask_path):
                skipped_mask_count += 1
                continue
            try:
                airflow_rate = data_utils.parse_airflow_rate(folder_name)
                delta_T = data_utils.parse_delta_T(mat_filename)
                if delta_T is None:
                    skipped_meta_count += 1
                    continue
            except ValueError:
                skipped_meta_count += 1
                continue
            except Exception:
                skipped_meta_count += 1
                continue
            all_samples_info.append({"mat_filepath": mat_filepath, "mask_path": mask_path, "delta_T": float(delta_T),
                                     "airflow_rate": float(airflow_rate), "source_folder_name": folder_name,
                                     "mat_filename_no_ext": mat_filename_no_ext})
    print(f"\nFound {file_count} .mat files. Skipped Meta: {skipped_meta_count}, Skipped Mask: {skipped_mask_count}. Proceeding with {len(all_samples_info)} samples.")
    if not all_samples_info:
        print("Error: No valid samples found.")
        return

    # 2. Extract Features
    feature_list = []
    print("\n--- Extracting Features ---")
    start_time = time.time()
    processed_samples, feature_error_count = 0, 0
    for i, sample_info in enumerate(tqdm(all_samples_info, desc="Extracting Features", ncols=100)):
        try:
            extracted_features_dict = feature_engineering.extract_features_with_mask(
                frames_or_path=sample_info["mat_filepath"], mask_path=sample_info["mask_path"],
                fps=FPS_FOR_FEATURES, focus_duration_sec=FOCUS_DURATION_SEC_FOR_FEATURES,
                envir_para=ENVIR_PARA, threshold_abs_change=THRESHOLD_ABS_CHANGE_FOR_AREA,
                source_folder_name=sample_info["source_folder_name"], mat_filename_no_ext=sample_info["mat_filename_no_ext"]
            )
            combined_features = {
                name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES}
            combined_features["delta_T"] = sample_info["delta_T"]
            combined_features["airflow_rate"] = sample_info["airflow_rate"]
            if extracted_features_dict:
                has_valid_selected = False
                for key, value in extracted_features_dict.items():
                    if key in combined_features:
                        combined_features[key] = value
                    if key in SELECTED_FEATURE_NAMES_FROM_MASK and np.isfinite(value):
                        has_valid_selected = True
                if not SELECTED_FEATURE_NAMES_FROM_MASK or has_valid_selected:
                    feature_list.append(combined_features)
                    processed_samples += 1
                else:
                    feature_error_count += 1
            else:
                feature_error_count += 1
        except Exception:
            feature_error_count += 1
            traceback.print_exc()
    end_time = time.time()
    print(
        f"\nFeature extraction. Processed: {processed_samples}, Errors/Skipped: {feature_error_count}. Took: {end_time - start_time:.2f}s.")
    if not feature_list:
        print("Error: No feature rows generated.")
        return
    df = pd.DataFrame(feature_list)
    df = df.infer_objects()
    print(f"DataFrame with ALL extracted features shape: {df.shape}")

    # 3. Prepare X and y
    if "airflow_rate" not in df.columns:
        print("Error: 'airflow_rate' column missing.")
        return
    if df["airflow_rate"].isnull().any():
        df = df.dropna(subset=["airflow_rate"])
    if df.empty:
        print("Error: DataFrame empty after target NaN drop.")
        return
    y = df["airflow_rate"].astype(float)
    X_final_for_model = pd.DataFrame(index=df.index)
    final_feature_names_for_model_list = []
    if 'delta_T' in df.columns:
        if LOG_TRANSFORM_DELTA_T:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
            X_final_for_model['delta_T_log'] = np.log1p(
                df['delta_T'].astype(float))
            final_feature_names_for_model_list.append('delta_T_log')
        else:
            X_final_for_model['delta_T'] = df['delta_T']
            final_feature_names_for_model_list.append('delta_T')
    if 'hotspot_area' in SELECTED_FEATURE_NAMES_FROM_MASK and 'hotspot_area' in df.columns:
        if LOG_TRANSFORM_AREA:
            X_final_for_model['hotspot_area_log'] = np.log1p(
                df['hotspot_area'].astype(float).clip(lower=0))
            final_feature_names_for_model_list.append('hotspot_area_log')
        else:
            X_final_for_model['hotspot_area'] = df['hotspot_area']
            final_feature_names_for_model_list.append('hotspot_area')
    if 'hotspot_avg_temp_change_rate_initial' in SELECTED_FEATURE_NAMES_FROM_MASK and \
       'hotspot_avg_temp_change_rate_initial' in df.columns and 'delta_T' in df.columns:
        if NORMALIZE_AVG_RATE_INITIAL:
            X_final_for_model['hotspot_avg_temp_change_rate_initial_norm'] = df.apply(
                lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_rate_initial']) else np.nan, axis=1)
            final_feature_names_for_model_list.append(
                'hotspot_avg_temp_change_rate_initial_norm')
        else:
            X_final_for_model['hotspot_avg_temp_change_rate_initial'] = df['hotspot_avg_temp_change_rate_initial']
            final_feature_names_for_model_list.append(
                'hotspot_avg_temp_change_rate_initial')
    direct_features_to_add_to_X = [
        'temp_max_overall_initial', 'overall_std_deltaT']
    for feat_name_direct in direct_features_to_add_to_X:
        if feat_name_direct in SELECTED_FEATURE_NAMES_FROM_MASK and feat_name_direct in df.columns:
            X_final_for_model[feat_name_direct] = df[feat_name_direct]
            final_feature_names_for_model_list.append(feat_name_direct)
    unique_final_features_for_X = sorted(
        list(set(final_feature_names_for_model_list)))
    X = X_final_for_model[unique_final_features_for_X].copy()
    print(f"Final Features for Model (X): {list(X.columns)}")
    if X.empty or y.empty or X.shape[1] == 0 or X.shape[0] != len(y):
        print("Error: Critical issue with final X or y. Exiting.")
        return
    print(
        f"Final Feature matrix X shape: {X.shape}, Final Target vector y shape: {y.shape}")

    # --- SAVE FEATURES AND TARGET ---
    if not X.empty and not y.empty:
        try:
            os.makedirs(config.FEATURES_SAVE_DIR, exist_ok=True)
            X.to_csv(os.path.join(config.FEATURES_SAVE_DIR,
                     "model_input_features_X.csv"), index=False)
            y.to_csv(os.path.join(config.FEATURES_SAVE_DIR, "target_variable_y.csv"), header=[
                     'airflow_rate'], index=False)
            df.to_csv(os.path.join(config.FEATURES_SAVE_DIR,
                      "all_extracted_features_for_viz_with_target.csv"), index=False)
            print(
                f"\nFeatures and target saved to: {config.FEATURES_SAVE_DIR}")
        except Exception as e_save:
            print(f"Error saving features/target: {e_save}")
            traceback.print_exc()

    # --- Hyperparameter Tuning ---
    print("\n--- Running Hyperparameter Tuning ---")
    all_best_params, all_best_scores = tuning.run_grid_search_all_models(
        X, y)  # tuning.py uses config for CV
    print("\n--- Tuning Results Summary ---")
    for name_model_tuned, score_tuned in all_best_scores.items():
        mse_tuned = -score_tuned if np.isfinite(score_tuned) else np.inf
        rmse_tuned = np.sqrt(mse_tuned) if np.isfinite(mse_tuned) else np.inf
        params_tuned = all_best_params.get(name_model_tuned, {})
        print(f"{name_model_tuned}: Best CV Score (neg_mse)={score_tuned:.4f} (MSE={mse_tuned:.4f}, RMSE={rmse_tuned:.4f}), Best Params={params_tuned}")

    # CV Setup for Evaluation
    num_samples_final = len(y)
    cv_strategy_main = None
    cv_name_main = "Unknown"
    n_splits_main = 0
    if config.CV_METHOD == 'LeaveOneOut':
        cv_strategy_main = LeaveOneOut()
        n_splits_main = num_samples_final
        cv_name_main = "LeaveOneOut"
    elif config.CV_METHOD == 'KFold':
        k_folds_val = getattr(config, 'K_FOLDS', 3)
        n_splits_main = min(k_folds_val, num_samples_final)
        if n_splits_main < 2:
            cv_strategy_main = LeaveOneOut()
            n_splits_main = num_samples_final
            cv_name_main = "LeaveOneOut (KFold Fallback)"
        else:
            cv_strategy_main = KFold(
                n_splits=n_splits_main, shuffle=True, random_state=config.RANDOM_STATE)
            cv_name_main = f"{n_splits_main}-Fold KFold"
    else:
        cv_strategy_main = LeaveOneOut()
        n_splits_main = num_samples_final
        cv_name_main = "LeaveOneOut (Default)"
    print(
        f"\nUsing {cv_name_main} CV ({n_splits_main} splits) for generalization performance evaluation.")

    # Evaluation Loop
    print(f"\n--- Evaluating ALL Models (CV, Training Set, Permutation Importance) ---")
    evaluation_results = {}

    for name, model_prototype_eval_loop in available_models.items():
        print(f"\n===== Processing Model: {name} =====")
        # Fresh instance for this iteration
        current_model_instance = modeling.get_regressors()[name]

        tuned_model_params = {
            k.replace('model__', ''): v for k, v in all_best_params.get(name, {}).items()}
        if tuned_model_params and name in all_best_params and np.isfinite(all_best_scores.get(name, -np.inf)):
            print(
                f"  Applying Tuned Parameters for {name}: {tuned_model_params}")
            try:
                current_model_instance.set_params(**tuned_model_params)
            except ValueError as e_set_p:
                print(
                    f"  Warning: Error setting tuned params for {name}: {e_set_p}. Using defaults.")
        else:
            print(f"  Using default parameters for {name}.")

        # CV Evaluation
        pipeline_for_cv = modeling.build_pipeline(
            current_model_instance, pca_components=config.PCA_N_COMPONENTS)
        try:
            with warnings.catch_warnings():
                if isinstance(current_model_instance, MLPRegressor):
                    warnings.filterwarnings(
                        "ignore", category=ConvergenceWarning)
                y_pred_cv_loop = cross_val_predict(
                    pipeline_for_cv, X, y, cv=cv_strategy_main, n_jobs=-1)
            if np.isnan(y_pred_cv_loop).any():
                y_pred_cv_loop = np.nan_to_num(
                    y_pred_cv_loop, nan=np.nanmean(y))
            mse_cv_loop = mean_squared_error(y, y_pred_cv_loop)
            rmse_cv_loop = np.sqrt(mse_cv_loop)
            mae_cv_loop = mean_absolute_error(y, y_pred_cv_loop)
            r2_cv_loop = r2_score(y, y_pred_cv_loop)
            evaluation_results[name] = {'mse': mse_cv_loop, 'rmse': rmse_cv_loop,
                                        'mae': mae_cv_loop, 'r2': r2_cv_loop, 'y_pred_cv': y_pred_cv_loop}
            print(f"--- CV Results for {name} ---")
            print(
                f"MSE: {mse_cv_loop:.4f} | RMSE: {rmse_cv_loop:.4f} | MAE: {mae_cv_loop:.4f} | R²: {r2_cv_loop:.4f}")
        except Exception as e_cv_loop:
            print(f"  Error during CV for {name}: {e_cv_loop}")
            traceback.print_exc()
            evaluation_results[name] = None
            continue

        if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True) and evaluation_results[name]:
            # ... (Plot CV actual vs predicted) ...
            plot_title_cv_current = f'Actual vs. Predicted - {name} (CV, Tuned*)'
            plot_save_path_cv_current = os.path.join(
                config.OUTPUT_DIR, f"actual_vs_predicted_CV_{name}.png")
            try:
                plotting.plot_actual_vs_predicted(
                    y, evaluation_results[name]['y_pred_cv'], plot_title_cv_current, plot_save_path_cv_current, scatter_kwargs={'s': 50, 'alpha': 0.8})
            except Exception as e_plot_cv_curr:
                print(
                    f"  Error plotting CV Actual vs Predicted for {name}: {e_plot_cv_curr}")

        # Training Set Evaluation (Overfitting Check) & Saving this specific tuned model
        pipeline_fitted_on_all_data = None  # To store the pipeline fitted on all data
        if config.EVALUATE_ON_FULL_TRAINING_SET:
            print(
                f"\n--- Training Set Performance for {name} (Overfitting Check) ---")
            try:
                # current_model_instance already has tuned params
                pipeline_fitted_on_all_data = modeling.build_pipeline(
                    current_model_instance, pca_components=config.PCA_N_COMPONENTS)
                with warnings.catch_warnings():
                    if isinstance(current_model_instance, MLPRegressor):
                        warnings.filterwarnings(
                            "ignore", category=ConvergenceWarning)
                    pipeline_fitted_on_all_data.fit(X, y)
                y_pred_train_set = pipeline_fitted_on_all_data.predict(X)
                r2_train_set_val = r2_score(y, y_pred_train_set)
                rmse_train_set_val = np.sqrt(
                    mean_squared_error(y, y_pred_train_set))
                print(
                    f"  R² (Training Set): {r2_train_set_val:.4f}  (vs CV R²: {r2_cv_loop:.4f})")
                print(
                    f"  RMSE (Training Set): {rmse_train_set_val:.4f} (vs CV RMSE: {rmse_cv_loop:.4f})")

                if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True):
                    # ... (Plot training set actual vs predicted) ...
                    plot_title_train_current = f'Actual vs. Predicted - {name} (TRAINING_SET_FIT)'
                    train_set_plot_save_path_current = os.path.join(
                        config.OUTPUT_DIR, f"actual_vs_predicted_TRAINING_SET_{name}.png")
                    plotting.plot_actual_vs_predicted(y, y_pred_train_set, plot_title_train_current, train_set_plot_save_path_current,
                                                      scatter_kwargs={'s': 30, 'alpha': 0.6, 'edgecolors': 'gray'}, add_jitter=True, jitter_strength=0.005)

                # --- SAVE THIS SPECIFIC TUNED MODEL (FITTED ON ALL DATA) ---
                individual_model_save_path = config.INDIVIDUAL_MODEL_SAVE_PATH_TEMPLATE.format(
                    model_name=name)
                os.makedirs(os.path.dirname(
                    individual_model_save_path), exist_ok=True)
                joblib.dump(pipeline_fitted_on_all_data,
                            individual_model_save_path)
                print(
                    f"  Tuned model '{name}' (fitted on all data) saved to: {individual_model_save_path}")

            except Exception as e_train_set_main:
                print(
                    f"  Error during full training set evaluation for {name}: {e_train_set_main}")
                traceback.print_exc()

        # Permutation Importance
        if getattr(config, 'SAVE_PERMUTATION_IMPORTANCE_PLOT', True) and evaluation_results[name]:
            if pipeline_fitted_on_all_data:
                print(f"\nCalculating Permutation Importance for {name} ...")
                try:
                    perm_result_loop = permutation_importance(
                        pipeline_fitted_on_all_data, X, y, n_repeats=15, random_state=config.RANDOM_STATE, n_jobs=-1, scoring='r2')
                    if hasattr(perm_result_loop, 'importances_mean'):
                        sorted_idx_perm_loop = perm_result_loop.importances_mean.argsort()
                        importances_df_perm_loop = pd.DataFrame(
                            perm_result_loop.importances[sorted_idx_perm_loop].T, columns=X.columns[sorted_idx_perm_loop])
                        plt.figure(figsize=(10, max(6, len(X.columns)*0.4)))
                        try:  # Matplotlib 3.9+ uses tick_labels
                            plt.boxplot(importances_df_perm_loop.values,
                                        tick_labels=importances_df_perm_loop.columns, vert=False, whis=10)
                        except TypeError:  # Fallback for older Matplotlib
                            plt.boxplot(importances_df_perm_loop.values,
                                        labels=importances_df_perm_loop.columns, vert=False, whis=10)
                        plt.title(f"Permutation Importances for {name}")
                        plt.axvline(x=0, color="k", linestyle="--")
                        plt.xlabel("Decrease in R² score")
                        plt.tight_layout()
                        perm_plot_path_loop = os.path.join(
                            config.OUTPUT_DIR, f"permutation_importance_{name}.png")
                        plt.savefig(perm_plot_path_loop, dpi=150)
                        plt.close()
                        print(
                            f"  Permutation importance plot for {name} saved to: {perm_plot_path_loop}")
                        print(f"  Feature Importances (Mean Decrease in R²):")
                        for i_p_loop in sorted_idx_perm_loop[::-1]:
                            print(
                                f"    {X.columns[i_p_loop]:<35}: {perm_result_loop.importances_mean[i_p_loop]:.4f} +/- {perm_result_loop.importances_std[i_p_loop]:.4f}")
                    else:
                        print(
                            f"  Warning: Permutation importance failed for {name} (no importances_mean).")
                except Exception as e_perm_loop:
                    print(
                        f"  Error in Permutation Importance for {name}: {e_perm_loop}")
                    traceback.print_exc()
            else:
                print(
                    f"  Skipping Permutation Importance for {name} as model was not fitted on all data (EVALUATE_ON_FULL_TRAINING_SET might be False).")

    # 7. Analyze Overall Best Model
    valid_results_final_check = {name_res_chk: res_val_chk for name_res_chk, res_val_chk in evaluation_results.items(
    ) if res_val_chk is not None and 'r2' in res_val_chk and np.isfinite(res_val_chk['r2'])}
    if not valid_results_final_check:
        print("\nCRITICAL Error: All model CV evaluations failed.")
        return
    best_model_name_overall_final = max(
        valid_results_final_check, key=lambda name_res_chk: valid_results_final_check[name_res_chk]['r2'])
    best_result_metrics_overall_final = valid_results_final_check[best_model_name_overall_final]
    print(
        f"\n--- OVERALL Best Model based on CV R²: {best_model_name_overall_final} ---")
    print(
        f"CV MSE: {best_result_metrics_overall_final['mse']:.4f} | CV RMSE: {best_result_metrics_overall_final['rmse']:.4f} | CV MAE: {best_result_metrics_overall_final['mae']:.4f} | CV R²: {best_result_metrics_overall_final['r2']:.4f}")

    # 8. Final Training and Saving of the OVERALL BEST Model
    print(
        f"\n--- Preparing to save OVERALL BEST model ({best_model_name_overall_final}) ---")
    try:

        # Get a fresh instance of the best model prototype and set its tuned params
        final_best_model_instance = modeling.get_regressors()[
            best_model_name_overall_final]
        best_overall_model_params_raw = all_best_params.get(
            best_model_name_overall_final, {})
        best_overall_model_params_final = {
            k.replace('model__', ''): v for k, v in best_overall_model_params_raw.items()}

        if best_overall_model_params_final and best_model_name_overall_final in all_best_params:
            print(
                f"  Applying Tuned Parameters for BEST model ({best_model_name_overall_final}): {best_overall_model_params_final}")
            final_best_model_instance.set_params(
                **best_overall_model_params_final)
        else:
            print(
                f"  Using default parameters for final BEST model ({best_model_name_overall_final}).")

        final_best_pipeline_to_save = modeling.build_pipeline(
            final_best_model_instance, pca_components=config.PCA_N_COMPONENTS)
        print(
            f"  Fitting the final best model ({best_model_name_overall_final}) on all data X, y...")
        with warnings.catch_warnings():
            if isinstance(final_best_model_instance, MLPRegressor):
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            final_best_pipeline_to_save.fit(X, y)
        print("  Overall best model final training complete.")

        # Save the OVERALL BEST model to the specific "best_models" subfolder
        overall_best_model_save_path = config.OVERALL_BEST_MODEL_SAVE_PATH_TEMPLATE.format(
            model_name=best_model_name_overall_final)
        os.makedirs(os.path.dirname(overall_best_model_save_path),
                    exist_ok=True)  # Ensure BEST_MODELS_DIR exists
        joblib.dump(final_best_pipeline_to_save, overall_best_model_save_path)
        print(
            f"\nFinal OVERALL BEST model ({best_model_name_overall_final}) pipeline saved to: {overall_best_model_save_path}")

        # Plot loss curve for the overall best model if it's an MLP and flag is True
        if getattr(config, 'SAVE_LOSS_CURVE_PLOT', False) and isinstance(final_best_pipeline_to_save.named_steps.get('model'), MLPRegressor):
            mlp_final_best_step = final_best_pipeline_to_save.named_steps.get(
                'model')
            if hasattr(mlp_final_best_step, 'loss_curve_'):
                print(
                    f"\nPlotting Training Loss Curve for OVERALL BEST MLP ({best_model_name_overall_final})...")
                loss_final_fname = f"final_loss_curve_OVERALL_BEST_{best_model_name_overall_final}.png"
                plotting.plot_loss_curve(mlp_final_best_step.loss_curve_,
                                         f"MLP Loss ({best_model_name_overall_final} - OVERALL BEST Final Fit)",
                                         config.OUTPUT_DIR, filename=loss_final_fname)

    except Exception as e_final_best_save:
        print(
            f"  Error during final training/saving of BEST model {best_model_name_overall_final}: {e_final_best_save}")
        traceback.print_exc()

    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    run_experiment()
