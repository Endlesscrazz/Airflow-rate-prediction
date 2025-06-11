# temp_main_for_nested_cv.py
"""
Temporary main script to test:
1. Hold-out set creation.
2. Nested Cross-Validation for hyperparameter tuning and performance estimation.
3. Per-material performance reporting.
4. Learning curve plotting.
Outputs are saved to 'output_temp' with a dynamically generated experiment folder.
Logs all parameters and terminal output.
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
import argparse  # Added for CLI arguments
import sys
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, RepeatedKFold, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

# Import project modules
import config
import data_utils
import feature_engineering
import modeling
import plotting
import tuning
import evaluation_utils

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*SettingWithCopyWarning*")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Global Configuration ---
BASE_OUTPUT_DIR_TEMP = "output_temp"

# --- Feature selection and transformation flags ---
ALL_POSSIBLE_FEATURE_NAMES = [
    'hotspot_area', 'hotspot_avg_temp_change_rate_initial',
    'hotspot_avg_temp_change_magnitude_initial', 'peak_pixel_temp_change_rate_initial',
    'peak_pixel_temp_change_magnitude_initial', 'temp_mean_avg_initial',
    'temp_std_avg_initial', 'temp_min_overall_initial', 'temp_max_overall_initial',
    'stabilized_mean_deltaT', 'overall_mean_deltaT', 'max_abs_mean_deltaT',
    'stabilized_std_deltaT', 'overall_std_deltaT', 'mean_area_significant_change',
    'stabilized_area_significant_change', 'max_area_significant_change',
]
SELECTED_RAW_FEATURES_TO_EXTRACT = [
    'hotspot_area',
    'hotspot_avg_temp_change_rate_initial',
    'temp_max_overall_initial',
    'temp_std_avg_initial',
    'overall_std_deltaT'
]
LOG_TRANSFORM_DELTA_T = getattr(config, 'LOG_TRANSFORM_DELTA_T', True)
LOG_TRANSFORM_AREA = getattr(config, 'LOG_TRANSFORM_AREA', True)
NORMALIZE_AVG_RATE_INITIAL = getattr(
    config, 'NORMALIZE_AVG_RATE_INITIAL', True)

# --- Class to redirect stdout and stderr to a log file ---
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
        except (IOError, ValueError):  # Handle potential errors during writing
            pass

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)


def log_parameters(param_file_path, args, feature_set):
    """Writes all relevant parameters to a text file."""
    with open(param_file_path, 'w') as f:
        f.write("--- Experiment Parameters ---\n\n")
        f.write("Command Line Arguments:\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"  {arg}: {value}\n")

        f.write("\nConfiguration from config.py:\n")
        config_vars = {k: v for k, v in config.__dict__.items() if not k.startswith(
            '_') and isinstance(v, (str, int, float, bool, list, tuple, dict))}
        for name, value in sorted(config_vars.items()):
            f.write(f"  config.{name}: {value}\n")

        f.write("\nFeature Engineering:\n")
        f.write(
            f"  SELECTED_RAW_FEATURES_TO_EXTRACT: {SELECTED_RAW_FEATURES_TO_EXTRACT}\n")
        f.write(f"  LOG_TRANSFORM_DELTA_T: {LOG_TRANSFORM_DELTA_T}\n")
        f.write(f"  LOG_TRANSFORM_AREA: {LOG_TRANSFORM_AREA}\n")
        f.write(
            f"  NORMALIZE_AVG_RATE_INITIAL: {NORMALIZE_AVG_RATE_INITIAL}\n")

        f.write(
            "\nFinal Feature Set Used for Model X (after transformation & one-hot encoding):\n")
        for i, feature in enumerate(feature_set):
            f.write(f"  {i+1}. {feature}\n")


def run_full_evaluation_pipeline(exp_number):
    material_configs_map = {
        "gypsum": {"dataset_subfolder": "dataset_gypsum", "mask_sub_path": "SP_Blur3_slope_p01_focus5s_q99_env1_roi0_mpOpenClose"},
        "brick_cladding": {"dataset_subfolder": "dataset_brickcladding", "mask_sub_path": "SP_Blur3_slope_p01_focus5s_q99_env1_roi0_mpOpenClose"}
    }
    mask_config_name = next(iter(material_configs_map.values()))[
        'mask_sub_path']
    experiment_name = f"exp-{exp_number}"

    base_output_dir = os.path.join(BASE_OUTPUT_DIR_TEMP, mask_config_name)
    current_output_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(current_output_dir, exist_ok=True)
    current_features_save_dir = os.path.join(
        current_output_dir, "saved_features")
    os.makedirs(current_features_save_dir, exist_ok=True)
    current_trained_models_dir = os.path.join(
        current_output_dir, "trained_models")
    os.makedirs(current_trained_models_dir, exist_ok=True)
    current_best_model_dir = os.path.join(
        current_trained_models_dir, "best_model_from_dev_set")
    os.makedirs(current_best_model_dir, exist_ok=True)

    log_file_path = os.path.join(current_output_dir, "output.log")
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout

    print(f"--- Starting Full Evaluation Pipeline ---")
    print(f"--- Experiment ID: {experiment_name} ---")
    print(f"--- All console output is being logged to: {log_file_path} ---")
    print(
        f"--- All other artifacts will be saved in: {current_output_dir} ---")

    # --- Data Loading and Preparation ---
    all_samples_info_list = []
    for material_name_load, mat_conf_load in material_configs_map.items():
        # ... (Data loading logic as before) ...
        dataset_path_load = os.path.join(
            config.DATASET_PARENT_DIR, mat_conf_load["dataset_subfolder"])
        mask_root_path_load = os.path.join(
            config.BASE_MASK_INPUT_DIR, material_name_load, mat_conf_load["mask_sub_path"])
        if not os.path.isdir(dataset_path_load) or not os.path.isdir(mask_root_path_load):
            continue
        for root_load, dirs_load, files_load in os.walk(dataset_path_load):
            if "cooling" in dirs_load:
                dirs_load.remove("cooling")
            folder_name_load = os.path.basename(root_load)
            for mat_filename_load in fnmatch.filter(files_load, '*.mat'):
                mat_filepath_load = os.path.join(root_load, mat_filename_load)
                rel_dir_mask_load = os.path.relpath(
                    root_load, dataset_path_load)
                if rel_dir_mask_load == ".":
                    rel_dir_mask_load = ""
                mask_path_load = os.path.join(mask_root_path_load, rel_dir_mask_load, os.path.splitext(
                    mat_filename_load)[0] + '_mask.npy')
                if not os.path.exists(mask_path_load):
                    continue
                try:
                    airflow_load = data_utils.parse_airflow_rate(
                        folder_name_load)
                    delta_t_load = data_utils.parse_delta_T(mat_filename_load)
                    if delta_t_load is None:
                        continue
                    all_samples_info_list.append({"mat_filepath": mat_filepath_load, "mask_path": mask_path_load, "delta_T": float(delta_t_load),
                                                 "airflow_rate": float(airflow_load), "material": material_name_load})
                except Exception:
                    continue
    if not all_samples_info_list:
        print("No samples loaded. Exiting.")
        return
    df_all_samples_raw_info = pd.DataFrame(all_samples_info_list)
    print(f"Total samples loaded: {len(df_all_samples_raw_info)}")

    # --- Hold-out Set Creation ---
    dev_df_info, hold_out_df_info = train_test_split(df_all_samples_raw_info, test_size=0.15, random_state=config.RANDOM_STATE, stratify=df_all_samples_raw_info['material']) if len(
        df_all_samples_raw_info) >= 10 else (df_all_samples_raw_info, pd.DataFrame())
    print(
        f"\nCreated hold-out set ({len(hold_out_df_info)} samples). Using {len(dev_df_info)} for development.")
    if not hold_out_df_info.empty:
        hold_out_df_info.to_csv(os.path.join(
            current_features_save_dir, "hold_out_set_INFO.csv"), index=False)
    df_to_process_for_features = dev_df_info.copy()

    # --- Feature Extraction for Development Set ---
    dev_feature_list_extracted = []
    for _, sample_row_dev in tqdm(df_to_process_for_features.iterrows(), total=len(df_to_process_for_features), desc="Extracting Features (Dev Set)"):
        extracted_dict_dev = feature_engineering.extract_features_with_mask(
            sample_row_dev["mat_filepath"], sample_row_dev["mask_path"], fps=config.MASK_FPS,
            focus_duration_sec=config.MASK_FOCUS_DURATION_SEC, envir_para=config.MASK_ENVIR_PARA,
            threshold_abs_change=config.FIXED_AREA_THRESHOLD)
        current_feats_dev = {
            name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES}
        current_feats_dev.update(
            {"delta_T": sample_row_dev["delta_T"], "airflow_rate": sample_row_dev["airflow_rate"], "material": sample_row_dev["material"]})
        if extracted_dict_dev:
            current_feats_dev.update(extracted_dict_dev)
        dev_feature_list_extracted.append(current_feats_dev)
    df_dev_features_extracted_raw = pd.DataFrame(dev_feature_list_extracted)
    if df_dev_features_extracted_raw.empty:
        print("No features extracted for dev set. Exiting.")
        return

    # --- Create X_dev, y_dev ---
    y_dev = df_dev_features_extracted_raw["airflow_rate"].astype(float)
    material_labels_dev = df_dev_features_extracted_raw["material"]
    X_dev_transformed = pd.DataFrame(index=df_dev_features_extracted_raw.index)
    dev_feature_names_list = []
    # Transformation Logic
    if LOG_TRANSFORM_DELTA_T:
        X_dev_transformed['delta_T_log'] = np.log1p(
            df_dev_features_extracted_raw['delta_T'].astype(float).clip(lower=0))
        dev_feature_names_list.append('delta_T_log')
    else:
        X_dev_transformed['delta_T'] = df_dev_features_extracted_raw['delta_T']
        dev_feature_names_list.append('delta_T')
    for f_name in SELECTED_RAW_FEATURES_TO_EXTRACT:
        if f_name == 'hotspot_area' and LOG_TRANSFORM_AREA:
            X_dev_transformed['hotspot_area_log'] = np.log1p(
                df_dev_features_extracted_raw[f_name].astype(float).clip(lower=0))
            dev_feature_names_list.append('hotspot_area_log')
        elif f_name == 'hotspot_avg_temp_change_rate_initial' and NORMALIZE_AVG_RATE_INITIAL:
            X_dev_transformed['hotspot_avg_temp_change_rate_initial_norm'] = df_dev_features_extracted_raw.apply(
                lambda r: r[f_name]/r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r[f_name]) else np.nan, axis=1)
            dev_feature_names_list.append(
                'hotspot_avg_temp_change_rate_initial_norm')
        elif f_name in df_dev_features_extracted_raw.columns and f_name not in X_dev_transformed.columns:
            X_dev_transformed[f_name] = df_dev_features_extracted_raw[f_name]
            dev_feature_names_list.append(f_name)
    dev_mat_dummies = pd.get_dummies(
        df_dev_features_extracted_raw['material'], prefix='material', dtype=int)
    X_dev_transformed = pd.concat(
        [X_dev_transformed, dev_mat_dummies.set_index(X_dev_transformed.index)], axis=1)
    for d_col in dev_mat_dummies.columns:
        dev_feature_names_list.append(d_col)
    X_dev = X_dev_transformed[sorted(list(set(dev_feature_names_list)))].copy()

    # Log parameters to file now that final feature set is known
    param_file_path = os.path.join(current_output_dir, "parameters.txt")
    log_parameters(param_file_path, args, X_dev.columns.tolist())
    print(f"\nExperiment parameters logged to: {param_file_path}")
    print(
        f"Development X_dev (shape {X_dev.shape}) features and y_dev (shape {y_dev.shape}) prepared.")
    print(f"Final feature set : {X_dev.columns.to_list()}")

    # --- Nested Cross-Validation ---
    print("\n--- Running Nested Cross-Validation on Development Set ---")

    available_models_prototypes = modeling.get_regressors()
    nested_cv_aggregated_scores = {}
    all_models_outer_fold_predictions = {}
    all_models_best_hyperparams_from_outer_folds = {}
    for model_name_ncv, model_proto_ncv in available_models_prototypes.items():
        if model_name_ncv not in tuning.param_grids:
            continue
        outer_scores, best_params_outer, y_true_outer_agg, y_pred_outer_agg, material_labels_outer_agg = \
            tuning.run_nested_cv_for_model_type(X_dev, y_dev, material_labels_dev, model_name_ncv, model_proto_ncv, tuning.param_grids[
                                                model_name_ncv], n_outer_folds=config.NESTED_CV_N_OUTER_FOLDS, n_repeats_outer=config.NESTED_CV_N_REPEATS, n_inner_folds=config.NESTED_CV_N_INNER_FOLDS, random_state=config.RANDOM_STATE)
        nested_cv_aggregated_scores[model_name_ncv] = evaluation_utils.aggregate_nested_cv_scores(
            model_name_ncv, outer_scores)
        all_models_outer_fold_predictions[model_name_ncv] = {
            'true': y_true_outer_agg, 'pred': y_pred_outer_agg, 'material': material_labels_outer_agg}
        all_models_best_hyperparams_from_outer_folds[model_name_ncv] = best_params_outer
        if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True) and y_true_outer_agg.size > 0:
            plotting.plot_actual_vs_predicted(y_true_outer_agg, y_pred_outer_agg, f"Actual vs. Pred - {model_name_ncv} ({config.NESTED_CV_N_REPEATS}x{config.NESTED_CV_N_OUTER_FOLDS}-Fold Nested CV)", os.path.join(
                current_output_dir, f"actual_vs_predicted_NESTED_CV_AGG_{model_name_ncv}.png"), scatter_kwargs={'s': 30, 'alpha': 0.6}, material_labels=material_labels_outer_agg, material_colors={"gypsum": "blue", "brick_cladding": "green"})

    # --- Select Best Model & Final Tuning ---

    if not nested_cv_aggregated_scores:
        print("No models evaluated. Exiting.")
        return
    best_model_type_ncv = max(nested_cv_aggregated_scores, key=lambda name: nested_cv_aggregated_scores[name].get(
        'mean_r2', -float('inf'))) if nested_cv_aggregated_scores else None
    if not best_model_type_ncv:
        print("Could not determine best model from Nested CV. Exiting.")
        return
    print(
        f"\n--- Best Model Type from Nested CV: {best_model_type_ncv} (Mean R²: {nested_cv_aggregated_scores[best_model_type_ncv].get('mean_r2'):.4f}) ---")
    evaluation_utils.analyze_hyperparameter_frequency(
        best_model_type_ncv, all_models_best_hyperparams_from_outer_folds.get(best_model_type_ncv, []))
    print(
        f"\n--- Final Hyperparameter Tuning for {best_model_type_ncv} on Full Development Set ---")
    final_tuning_cv_strategy = KFold(n_splits=config.K_FOLDS_FOR_FINAL_TUNING, shuffle=True, random_state=config.RANDOM_STATE) if len(
        X_dev) >= config.K_FOLDS_FOR_FINAL_TUNING else LeaveOneOut()
    final_best_params_for_dev_model, _ = tuning.run_grid_search_for_final_tuning(
        X_dev, y_dev, available_models_prototypes[best_model_type_ncv], tuning.param_grids[best_model_type_ncv], cv_strategy=final_tuning_cv_strategy)

    # --- Train Final Model & Save ---

    final_dev_model_proto = clone(
        available_models_prototypes[best_model_type_ncv])
    final_dev_model_proto.set_params(
        **{k.replace('model__', ''): v for k, v in final_best_params_for_dev_model.items()})
    pipeline_final_dev_model = modeling.build_pipeline(final_dev_model_proto)
    print(f"Fitting final {best_model_type_ncv} model on dev set...")
    pipeline_final_dev_model.fit(X_dev, y_dev)
    print("  Final model fitting complete.")
    dev_model_save_path = os.path.join(
        current_best_model_dir, f"FINAL_DEV_MODEL_{best_model_type_ncv}.joblib")
    joblib.dump(pipeline_final_dev_model, dev_model_save_path)
    print(
        f"  Final model (trained on full dev set) saved to: {dev_model_save_path}")

    # --- Learning Curve ---

    if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True):
        lc_save_path = os.path.join(
            current_output_dir, f"learning_curve_{best_model_type_ncv}_COMBINED.png")
        print(f"\nPlotting learning curve for {best_model_type_ncv}...")
        lc_cv_strategy = KFold(n_splits=config.LEARNING_CURVE_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE) if len(
            X_dev) >= config.LEARNING_CURVE_CV_FOLDS*2 else None
        if lc_cv_strategy:
            plotting.plot_learning_curves_custom(clone(pipeline_final_dev_model), f"Learning Curve ({best_model_type_ncv}, Combined Dev Data)",
                                                 X_dev, y_dev, cv=lc_cv_strategy, scoring='r2', save_path=lc_save_path, train_sizes=config.LEARNING_CURVE_TRAIN_SIZES)

    # --- Evaluate on Hold-Out Set ---
    if not hold_out_df_info.empty:
        print("\n--- Evaluating Final Model on Hold-Out Set ---")

        ho_feature_list = []
        for _, ho_sample_row in tqdm(hold_out_df_info.iterrows(), total=len(hold_out_df_info), desc="Extracting Features (Hold-Out)"):
            ho_extracted = feature_engineering.extract_features_with_mask(ho_sample_row["mat_filepath"], ho_sample_row["mask_path"], fps=config.MASK_FPS,
                                                                          focus_duration_sec=config.MASK_FOCUS_DURATION_SEC, envir_para=config.MASK_ENVIR_PARA, threshold_abs_change=config.FIXED_AREA_THRESHOLD)
            ho_feats = {name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES}
            ho_feats.update(
                {"delta_T": ho_sample_row["delta_T"], "airflow_rate": ho_sample_row["airflow_rate"], "material": ho_sample_row["material"]})
            if ho_extracted:
                ho_feats.update(ho_extracted)
            ho_feature_list.append(ho_feats)
        df_ho_features_raw = pd.DataFrame(ho_feature_list)

        if not df_ho_features_raw.empty:
            y_ho = df_ho_features_raw["airflow_rate"].astype(float)
            material_labels_ho = df_ho_features_raw["material"]
            X_ho_transformed = pd.DataFrame(index=df_ho_features_raw.index)
            # Reconstruct X_ho with same columns as X_dev
            for col_name_ho in X_dev.columns:
                if col_name_ho.startswith("material_"):
                    X_ho_transformed[col_name_ho] = (
                        df_ho_features_raw['material'] == col_name_ho.split("material_")[-1]).astype(int)
                elif col_name_ho == 'delta_T_log' and LOG_TRANSFORM_DELTA_T:
                    X_ho_transformed[col_name_ho] = np.log1p(
                        df_ho_features_raw['delta_T'].astype(float).clip(lower=0))
                elif col_name_ho == 'hotspot_area_log' and 'hotspot_area' in SELECTED_RAW_FEATURES_TO_EXTRACT and LOG_TRANSFORM_AREA:
                    X_ho_transformed[col_name_ho] = np.log1p(
                        df_ho_features_raw['hotspot_area'].astype(float).clip(lower=0))
                elif col_name_ho == 'hotspot_avg_temp_change_rate_initial_norm' and 'hotspot_avg_temp_change_rate_initial' in SELECTED_RAW_FEATURES_TO_EXTRACT and NORMALIZE_AVG_RATE_INITIAL:
                    X_ho_transformed[col_name_ho] = df_ho_features_raw.apply(
                        lambda r: r['hotspot_avg_temp_change_rate_initial']/r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_rate_initial']) else np.nan, axis=1)
                elif col_name_ho in df_ho_features_raw.columns:
                    X_ho_transformed[col_name_ho] = df_ho_features_raw[col_name_ho]
                else:
                    X_ho_transformed[col_name_ho] = np.nan
            X_ho = X_ho_transformed[X_dev.columns].copy()
            y_pred_ho = pipeline_final_dev_model.predict(X_ho)
            r2_ho = r2_score(y_ho, y_pred_ho)
            rmse_ho = np.sqrt(mean_squared_error(y_ho, y_pred_ho))
            mae_ho = mean_absolute_error(y_ho, y_pred_ho)
            print(
                f"  Hold-Out Performance ({best_model_type_ncv}): R²={r2_ho:.4f}, RMSE={rmse_ho:.4f}, MAE={mae_ho:.4f}")
            if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True):
                plotting.plot_actual_vs_predicted(y_ho, y_pred_ho, f"Hold-Out Set - {best_model_type_ncv}", os.path.join(current_output_dir, f"actual_vs_predicted_HOLD_OUT_{best_model_type_ncv}.png"),
                                                  scatter_kwargs={'s': 70, 'alpha': 0.9, 'edgecolors': 'black'}, material_labels=material_labels_ho, material_colors={"gypsum": "blue", "brick_cladding": "green"})

            # Per-Material Performance on Hold-Out
            print("\n--- Per-Material Performance on Hold-Out Set ---")
            for mat_type_ho in df_ho_features_raw['material'].unique():
                ho_mat_mask = (df_ho_features_raw['material'] == mat_type_ho)
                y_true_ho_mat = y_ho[ho_mat_mask]
                y_pred_ho_mat = y_pred_ho[ho_mat_mask]
                if len(y_true_ho_mat) > 0:
                    r2_ho_m = r2_score(y_true_ho_mat, y_pred_ho_mat)

                    rmse_ho_m = np.sqrt(mean_squared_error(
                        y_true_ho_mat, y_pred_ho_mat))
                    print(
                        f"  Material: {mat_type_ho} (N={len(y_true_ho_mat)}): Hold-Out R²={r2_ho_m:.4f}, RMSE={rmse_ho_m:.4f}")

    print(f"\n--- Full Evaluation Pipeline Finished for {experiment_name} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full evaluation pipeline for combined materials.")
    parser.add_argument("exp_number", type=int,
                        help="An integer number for this experiment run (e.g., 1, 2, 3). Used for creating the output folder 'exp-001'.")

    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    run_full_evaluation_pipeline(exp_number=args.exp_number)
