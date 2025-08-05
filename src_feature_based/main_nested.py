# temp_main_for_nested_cv.py
"""
(FINAL VERSION: AGGREGATE FEATURES, GROUP-AWARE CV, CORRECT AUGMENTATION)
Main script to test and evaluate airflow prediction models.
- Handles multiple source datasets (single-hole, two-hole, gypsum, brick cladding).
- Uses "aggregate-features-per-video" strategy (one sample per unique video).
- Implements Group-Aware Nested Cross-Validation (GroupKFold).
- Implements data augmentation CORRECTLY on inner training folds only.
- Extracts and uses the full, powerful feature set.
- Generates comprehensive diagnostic plots.
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
import argparse
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
warnings.filterwarnings("ignore", category=UserWarning, message=".*SettingWithCopyWarning*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning) 

# --- Global Configuration ---
BASE_OUTPUT_DIR_TEMP = "output_temp"

ALL_POSSIBLE_FEATURE_NAMES = feature_engineering.ALL_POSSIBLE_FEATURE_NAMES
SELECTED_RAW_FEATURES_TO_EXTRACT = [
    'delta_T_log',
    'hotspot_area_log',
    'hotspot_avg_temp_change_rate_initial',
    'overall_std_deltaT',
    'temp_max_overall_initial',
    'temp_std_avg_initial',
] 

LOG_TRANSFORM_DELTA_T = getattr(config, 'LOG_TRANSFORM_DELTA_T', True)
LOG_TRANSFORM_AREA = getattr(config, 'LOG_TRANSFORM_AREA', True)
NORMALIZE_AVG_RATE_INITIAL = getattr(config, 'NORMALIZE_AVG_RATE_INITIAL', True)

# --- Class to redirect stdout and stderr to a log file ---
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
        except (IOError, ValueError): pass
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

def log_parameters(param_file_path, args, feature_set, material_to_run, dataset_paths, mask_paths):
    with open(param_file_path, 'w') as f:
        f.write(f"--- Experiment Parameters for Run: {material_to_run} | Exp ID: {args.exp_number} ---\n\n")
        f.write("Command Line Arguments:\n" + "-" * 25 + "\n")
        for arg, value in sorted(vars(args).items()): f.write(f"  {arg}: {value}\n")
        f.write("\nConfiguration from config.py:\n" + "-" * 25 + "\n")
        config_vars = {k: v for k, v in config.__dict__.items() if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, tuple, dict))}
        for name, value in sorted(config_vars.items()): f.write(f"  config.{name}: {value}\n")
        f.write("\nData Loading Paths:\n" + "-" * 25 + "\n")
        f.write(f"  Dataset Paths: {dataset_paths}\n")
        f.write(f"  Mask Paths: {mask_paths}\n")
        f.write("\nFeature Engineering:\n" + "-" * 25 + "\n")
        f.write(f"  SELECTED_RAW_FEATURES_TO_EXTRACT: {SELECTED_RAW_FEATURES_TO_EXTRACT}\n")
        f.write(f"  LOG_TRANSFORM_DELTA_T: {LOG_TRANSFORM_DELTA_T}\n")
        f.write(f"  LOG_TRANSFORM_AREA: {LOG_TRANSFORM_AREA}\n")
        f.write(f"  NORMALIZE_AVG_RATE_INITIAL: {NORMALIZE_AVG_RATE_INITIAL}\n")
        f.write("\nFinal Feature Set Used for Model X:\n" + "-" * 25 + "\n")
        for i, feature in enumerate(feature_set): f.write(f"  {i+1}. {feature}\n")


def run_full_evaluation_pipeline(material_to_run, exp_number):
    
    # --- Data Source Configuration ---
    # This defines where original video data and SAM masks are located.
    # mask_subfolder points to the base directory under output_SAM/datasets/
    DATASET_CONFIGS = {
        "gypsum_single_hole": {"material": "gypsum", "dataset_subfolder": "dataset_gypsum", "mask_subfolder": "dataset_gypsum"},
        "gypsum_single_hole2": {"material": "gypsum", "dataset_subfolder": "dataset_gypsum2", "mask_subfolder": "dataset_gypsum2"},
        #"brick_cladding_single_hole": {"material": "brick_cladding", "dataset_subfolder": "dataset_brickcladding", "mask_subfolder": "dataset_brickcladding"},
        #"brick_cladding_two_holes": {"material": "brick_cladding", "dataset_subfolder": "dataset_two_holes_brickcladding", "mask_subfolder": "dataset_two_holes_brickcladding"}
    }

    # Determine which datasets to load based on 'material_to_run' argument
    datasets_to_load_keys = []
    if material_to_run == "combined": datasets_to_load_keys = list(DATASET_CONFIGS.keys())
    elif material_to_run == "gypsum": datasets_to_load_keys = ["gypsum_single_hole", "gypsum_single_hole2"]
    elif material_to_run == "brick_cladding": datasets_to_load_keys = ["brick_cladding_single_hole", "brick_cladding_two_holes"]
    else: print(f"Error: Material '{material_to_run}' not recognized."); return

    # --- Experiment Output Directory Setup ---
    output_subfolder_name = f"output-2G-1BC_{material_to_run}" # to change output folder name 
    experiment_name = f"exp-{exp_number}"
    current_output_dir = os.path.join(BASE_OUTPUT_DIR_TEMP, output_subfolder_name, experiment_name)
    os.makedirs(current_output_dir, exist_ok=True)
    
    current_features_save_dir = os.path.join(current_output_dir, "saved_features")
    os.makedirs(current_features_save_dir, exist_ok=True)
    current_trained_models_dir = os.path.join(current_output_dir, "trained_models")
    os.makedirs(current_trained_models_dir, exist_ok=True)
    current_best_model_dir = os.path.join(current_trained_models_dir, "best_model_from_dev_set")
    os.makedirs(current_best_model_dir, exist_ok=True)

    # --- Setup Logging ---
    log_file_path = os.path.join(current_output_dir, "output.log")
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout
    print(f"--- Starting Full Evaluation Pipeline for Run: '{material_to_run}' | Exp ID: {exp_number} ---")
    print(f"--- Console output logged to: {log_file_path} ---")

    # --- Step 1: Data Loading (Aggregate-Features per Video) ---
    all_samples_info_list = [] # This will hold one dictionary per unique video
    dataset_paths_log = []; mask_paths_log = []
    video_to_masks_map = {} # Temporarily store grouped masks

    for d_key in datasets_to_load_keys:
        d_config = DATASET_CONFIGS[d_key]
        print(f"\nScanning data from dataset: '{d_key}' (Material: {d_config['material']})")
        dataset_path_load = os.path.join(config.DATASET_PARENT_DIR, d_config["dataset_subfolder"])
        mask_root_path_load = os.path.join(config.BASE_MASK_INPUT_DIR, d_config["mask_subfolder"])
        
        dataset_paths_log.append(dataset_path_load)
        mask_paths_log.append(mask_root_path_load)
        
        if not os.path.isdir(dataset_path_load) or not os.path.isdir(mask_root_path_load):
            print(f"  Warning: Path not found. Dataset: '{dataset_path_load}', Masks: '{mask_root_path_load}'. Skipping."); continue

        for root_load, dirs_load, files_load in os.walk(dataset_path_load):
            if "cooling" in dirs_load: dirs_load.remove("cooling") # Skip cooling folders
            
            for mat_filename_load in fnmatch.filter(files_load, '*.mat'):
                mat_filepath_load = os.path.join(root_load, mat_filename_load)
                video_id = os.path.splitext(mat_filename_load)[0] # Unique ID for this video

                # Initialize sample entry if not seen yet (important for videos with multiple masks)
                if video_id not in video_to_masks_map:
                    try:
                        # Extract metadata once per video
                        folder_name_load = os.path.basename(os.path.dirname(mat_filepath_load))
                        airflow_load = data_utils.parse_airflow_rate(folder_name_load)
                        delta_t_load = data_utils.parse_delta_T(mat_filename_load)
                        if delta_t_load is None: continue # Skip video if metadata invalid
                        
                        video_to_masks_map[video_id] = {
                            "video_id": video_id, # Store video_id here
                            "mat_filepath": mat_filepath_load,
                            "mask_paths": [], # This will store a list of all mask paths for this video
                            "delta_T": float(delta_t_load),
                            "airflow_rate": float(airflow_load),
                            "material": d_config["material"]
                        }
                    except Exception as e:
                        print(f"  Warning: Metadata parsing failed for {mat_filename_load}. Error: {e}. Skipping video.")
                        continue

                # Find all masks for this video and add to its list
                mat_basename = os.path.splitext(mat_filename_load)[0]
                relative_path_part = os.path.relpath(root_load, dataset_path_load)
                mask_search_dir = os.path.join(mask_root_path_load, relative_path_part, mat_basename)
                
                if os.path.isdir(mask_search_dir):
                    mask_files_found = []
                    # Check for individually saved masks (e.g., _mask_0.npy, _mask_1.npy)
                    mask_files_found.extend(fnmatch.filter(os.listdir(mask_search_dir), f"{mat_basename}_mask_*.npy"))
                    # Check for single combined mask (e.g., _sam_mask.npy)
                    mask_files_found.extend(fnmatch.filter(os.listdir(mask_search_dir), f"{mat_basename}_sam_mask.npy"))

                    for mask_filename in mask_files_found:
                        full_mask_path = os.path.join(mask_search_dir, mask_filename)
                        video_to_masks_map[video_id]["mask_paths"].append(full_mask_path)
    
    # Finalize all_samples_info_list from the map (ensuring only videos with masks are included)
    all_samples_info_list = [v_data for v_data in video_to_masks_map.values() if v_data["mask_paths"]]

    if not all_samples_info_list: print("No video samples with masks loaded. Exiting."); return
    df_all_samples_raw_info = pd.DataFrame(all_samples_info_list)
    print(f"\nTotal UNIQUE VIDEO samples loaded: {len(df_all_samples_raw_info)}")
    print("Sample counts per material:\n", df_all_samples_raw_info['material'].value_counts())

    # --- Step 2: Create Hold-Out Set (on ORIGINAL, UNIQUE video samples) ---
    hold_out_df_info = pd.DataFrame() 
    MIN_SAMPLES_FOR_HOLDOUT = 30 
    
    if len(df_all_samples_raw_info) >= MIN_SAMPLES_FOR_HOLDOUT:
        print(f"\nCreating hold-out set from {len(df_all_samples_raw_info)} ORIGINAL video samples.")
        
        # Create bins from the continuous airflow_rate for stratification
        original_airflow_bins = pd.cut(df_all_samples_raw_info['airflow_rate'], bins=5, labels=False, duplicates='drop')
        
        dev_df_info, hold_out_df_info = train_test_split(
            df_all_samples_raw_info, 
            test_size=0.2, # Use 20% for hold-out, as per config for a small dataset
            random_state=config.RANDOM_STATE, 
            stratify=original_airflow_bins # Stratify the hold-out split by airflow rate bins
        )
        print(f"  Created hold-out set ({len(hold_out_df_info)} ORIGINAL samples). Using {len(dev_df_info)} for development.")
        if not hold_out_df_info.empty:
            hold_out_df_info.to_csv(os.path.join(current_features_save_dir, "hold_out_set_INFO.csv"), index=False)
        df_to_process_for_features = dev_df_info.copy()
    else:
        print(f"\nWarning: Dataset size ({len(df_all_samples_raw_info)}) is small. Using all data for development (no hold-out).")
        df_to_process_for_features = df_all_samples_raw_info.copy()

    # --- Step 3: Extract Features (for ORIGINAL development/hold-out samples) ---
    dev_feature_list_extracted = []
    for _, sample_row_dev in tqdm(df_to_process_for_features.iterrows(), total=len(df_to_process_for_features), desc="Extracting Features (Dev Set)"):
        # Call extract_aggregate_features with a list of mask paths
        extracted_dict_dev = feature_engineering.extract_aggregate_features(
            sample_row_dev["mat_filepath"],
            sample_row_dev["mask_paths"], # Pass the LIST of mask paths
            fps=config.MASK_FPS,
            focus_duration_sec=config.MASK_FOCUS_DURATION_SEC, 
            envir_para=config.MASK_ENVIR_PARA,
            threshold_abs_change=config.FIXED_AREA_THRESHOLD
        )
        current_feats_dev = {name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES}
        current_feats_dev.update({"video_id": sample_row_dev["video_id"], "delta_T": sample_row_dev["delta_T"], "airflow_rate": sample_row_dev["airflow_rate"], "material": sample_row_dev["material"]})
        if extracted_dict_dev: current_feats_dev.update(extracted_dict_dev)
        dev_feature_list_extracted.append(current_feats_dev)
    df_dev_features_extracted_raw = pd.DataFrame(dev_feature_list_extracted)
    if df_dev_features_extracted_raw.empty: print("No features extracted for dev set. Exiting."); return

    # --- Step 4: Prepare X_dev, y_dev (from ORIGINAL development samples) ---
    y_dev = df_dev_features_extracted_raw["airflow_rate"].astype(float)
    material_labels_dev = df_dev_features_extracted_raw["material"]
    video_id_groups_dev = df_dev_features_extracted_raw["video_id"] # Groups for GroupKFold based on ORIGINAL videos
    X_dev_transformed = pd.DataFrame(index=df_dev_features_extracted_raw.index)
    dev_feature_names_list = []

    # Handle feature transformations
    if 'delta_T' in df_dev_features_extracted_raw.columns:
        if LOG_TRANSFORM_DELTA_T: X_dev_transformed['delta_T_log'] = np.log1p(df_dev_features_extracted_raw['delta_T'].astype(float).clip(lower=0))
        else: X_dev_transformed['delta_T'] = df_dev_features_extracted_raw['delta_T']
        dev_feature_names_list.append('delta_T_log' if LOG_TRANSFORM_DELTA_T else 'delta_T')

    if 'hotspot_area' in SELECTED_RAW_FEATURES_TO_EXTRACT and 'hotspot_area' in df_dev_features_extracted_raw.columns:
        if LOG_TRANSFORM_AREA: X_dev_transformed['hotspot_area_log'] = np.log1p(df_dev_features_extracted_raw['hotspot_area'].astype(float).clip(lower=0))
        else: X_dev_transformed['hotspot_area'] = df_dev_features_extracted_raw['hotspot_area']
        dev_feature_names_list.append('hotspot_area_log' if LOG_TRANSFORM_AREA else 'hotspot_area')
            
    if 'hotspot_avg_temp_change_rate_initial' in SELECTED_RAW_FEATURES_TO_EXTRACT and 'hotspot_avg_temp_change_rate_initial' in df_dev_features_extracted_raw.columns:
        if NORMALIZE_AVG_RATE_INITIAL: X_dev_transformed['hotspot_avg_temp_change_rate_initial_norm'] = df_dev_features_extracted_raw.apply(lambda r: r['hotspot_avg_temp_change_rate_initial']/r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_rate_initial']) else np.nan, axis=1)
        else: X_dev_transformed['hotspot_avg_temp_change_rate_initial'] = df_dev_features_extracted_raw['hotspot_avg_temp_change_rate_initial']
        dev_feature_names_list.append('hotspot_avg_temp_change_rate_initial_norm' if NORMALIZE_AVG_RATE_INITIAL else 'hotspot_avg_temp_change_rate_initial')

    # Add all OTHER selected raw features that haven't been specially handled by transformation
    special_transformed_features = ['hotspot_area', 'hotspot_avg_temp_change_rate_initial', 'delta_T']
    for f_name in SELECTED_RAW_FEATURES_TO_EXTRACT:
        if f_name not in special_transformed_features and f_name in df_dev_features_extracted_raw.columns:
            X_dev_transformed[f_name] = df_dev_features_extracted_raw[f_name]
            dev_feature_names_list.append(f_name)

    # One-hot encode material
    if material_to_run == "combined" or df_all_samples_raw_info['material'].nunique() > 1:
        print("\nApplying one-hot encoding for 'material' feature.")
        if 'material' in df_dev_features_extracted_raw.columns:
            all_possible_materials = df_all_samples_raw_info['material'].unique().tolist()
            df_dev_features_extracted_raw['material'] = pd.Categorical(df_dev_features_extracted_raw['material'], categories=all_possible_materials)
            dev_mat_dummies = pd.get_dummies(df_dev_features_extracted_raw['material'], prefix='material', dtype=int)
            X_dev_transformed = pd.concat([X_dev_transformed, dev_mat_dummies.set_index(X_dev_transformed.index)], axis=1)
            dev_feature_names_list.extend(dev_mat_dummies.columns)

    X_dev = X_dev_transformed[sorted(list(set(dev_feature_names_list)))].copy()
    
    log_parameters(os.path.join(current_output_dir, "parameters.txt"), args, X_dev.columns.tolist(), material_to_run, dataset_paths_log, mask_paths_log)
    print(f"\nDevelopment X_dev (shape {X_dev.shape}) and y_dev (shape {y_dev.shape}) prepared.")
    print(f"Feature set: {X_dev.columns.tolist()}")

    # --- Step 5: Run Nested CV (Augmentation happens INSIDE tuning.py) ---
    print("\n--- Running Nested Cross-Validation on Development Set ---")
    available_models_prototypes = modeling.get_regressors()
    nested_cv_aggregated_scores = {}
    all_models_outer_fold_predictions = {}
    all_models_best_hyperparams_from_outer_folds = {}
    for model_name_ncv, model_proto_ncv in available_models_prototypes.items():
        if model_name_ncv not in tuning.param_grids: continue
        outer_scores, best_params_outer, y_true_outer, y_pred_outer, material_labels_outer, video_ids_outer = \
            tuning.run_nested_cv_for_model_type(X_dev, y_dev, material_labels_dev, video_id_groups_dev,
                                                model_name_ncv, model_proto_ncv, tuning.param_grids[model_name_ncv],
                                                n_outer_folds=config.NESTED_CV_N_OUTER_FOLDS,
                                                n_inner_folds=config.NESTED_CV_N_INNER_FOLDS, random_state=config.RANDOM_STATE)
        nested_cv_aggregated_scores[model_name_ncv] = evaluation_utils.aggregate_nested_cv_scores(model_name_ncv, outer_scores)
        all_models_outer_fold_predictions[model_name_ncv] = {'true': y_true_outer, 'pred': y_pred_outer, 'material': material_labels_outer, 'video_id': video_ids_outer}
        all_models_best_hyperparams_from_outer_folds[model_name_ncv] = best_params_outer
        if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True) and y_true_outer.size > 0:
            plotting.plot_actual_vs_predicted(y_true_outer, y_pred_outer, f"Actual vs. Pred - {model_name_ncv} (GroupKFold Nested CV)", os.path.join(current_output_dir, f"actual_vs_predicted_NESTED_CV_AGG_{model_name_ncv}.png"), material_labels=material_labels_outer)

    if not nested_cv_aggregated_scores: print("No models evaluated. Exiting."); return
    best_model_type_ncv = max(nested_cv_aggregated_scores, key=lambda n: nested_cv_aggregated_scores[n].get('mean_r2', -float('inf')))
    if not best_model_type_ncv: print("Could not determine best model. Exiting."); return
    print(f"\n--- Best Model Type from Nested CV: {best_model_type_ncv} (Mean R²: {nested_cv_aggregated_scores[best_model_type_ncv].get('mean_r2'):.4f}) ---")
    evaluation_utils.analyze_hyperparameter_frequency(best_model_type_ncv, all_models_best_hyperparams_from_outer_folds.get(best_model_type_ncv, []))

    # --- Step 6: Final Tuning and Model Saving ---
    print(f"\n--- Final Hyperparameter Tuning for {best_model_type_ncv} on Full Development Set ---")
    # For final tuning, KFold is typically used as we don't have groups to split here
    final_tuning_cv = KFold(n_splits=config.K_FOLDS_FOR_FINAL_TUNING, shuffle=True, random_state=config.RANDOM_STATE) if len(X_dev) >= config.K_FOLDS_FOR_FINAL_TUNING else LeaveOneOut()
    final_best_params, _ = tuning.run_grid_search_for_final_tuning(X_dev, y_dev, available_models_prototypes[best_model_type_ncv], tuning.param_grids[best_model_type_ncv], cv_strategy=final_tuning_cv)
    
    final_dev_model_proto = clone(available_models_prototypes[best_model_type_ncv])
    final_dev_model_proto.set_params(**{k.replace('model__', ''): v for k, v in final_best_params.items()})
    pipeline_final_dev_model = modeling.build_pipeline(final_dev_model_proto)
    pipeline_final_dev_model.fit(X_dev, y_dev) # Fit on ORIGINAL development set
    joblib.dump(pipeline_final_dev_model, os.path.join(current_best_model_dir, f"FINAL_DEV_MODEL_{best_model_type_ncv}.joblib"))

    if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True):
        lc_cv = KFold(n_splits=config.LEARNING_CURVE_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE) if len(X_dev) >= config.LEARNING_CURVE_CV_FOLDS*2 else None
        if lc_cv: plotting.plot_learning_curves_custom(clone(pipeline_final_dev_model), f"Learning Curve ({best_model_type_ncv})", X_dev, y_dev, cv=lc_cv, scoring='r2', save_path=os.path.join(current_output_dir, f"learning_curve_{best_model_type_ncv}.png"))

    # --- Step 7: Evaluate on Hold-Out Set (ORIGINAL samples only) ---
    if not hold_out_df_info.empty:
        print("\n--- Evaluating Final Model on Hold-Out Set ---")
        ho_feature_list = []
        for _, ho_sample_row in tqdm(hold_out_df_info.iterrows(), total=len(hold_out_df_info), desc="Extracting Features (Hold-Out)"):
            ho_extracted = feature_engineering.extract_aggregate_features(
                ho_sample_row["mat_filepath"], ho_sample_row["mask_paths"], 
                fps=config.MASK_FPS, focus_duration_sec=config.MASK_FOCUS_DURATION_SEC, 
                envir_para=config.MASK_ENVIR_PARA, threshold_abs_change=config.FIXED_AREA_THRESHOLD)
            
            ho_feats = {name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES}
            ho_feats.update({"video_id": ho_sample_row["video_id"], "delta_T": ho_sample_row["delta_T"], "airflow_rate": ho_sample_row["airflow_rate"], "material": ho_sample_row["material"]})
            if ho_extracted: ho_feats.update(ho_extracted)
            ho_feature_list.append(ho_feats)
        
        df_ho_features_raw = pd.DataFrame(ho_feature_list)

        if not df_ho_features_raw.empty:
            y_ho = df_ho_features_raw["airflow_rate"].astype(float)
            # Filter X_dev columns as features may not be in holdout if no mask found
            common_cols = list(set(X_dev.columns) & set(ALL_POSSIBLE_FEATURE_NAMES + ['delta_T_log'] + [col for col in X_dev.columns if col.startswith('material_')]))
            X_ho_transformed = pd.DataFrame(index=df_ho_features_raw.index)
            # Reconstruct X_ho with SAME columns as X_dev
            for col_name_ho in X_dev.columns: # Iterate through the columns of X_dev to ensure X_ho matches
                if col_name_ho.startswith("material_"):
                    # For material columns, ensure it's set to 1 if material matches, 0 otherwise
                    mat_type_ho = col_name_ho.replace("material_", "")
                    X_ho_transformed[col_name_ho] = (df_ho_features_raw['material'] == mat_type_ho).astype(int)
                elif col_name_ho == 'delta_T_log':
                    X_ho_transformed[col_name_ho] = np.log1p(df_ho_features_raw['delta_T'].astype(float).clip(lower=0))
                elif col_name_ho == 'hotspot_area_log': # Ensure this is covered
                    X_ho_transformed[col_name_ho] = np.log1p(df_ho_features_raw.get('hotspot_area', 0).astype(float).clip(lower=0)) # Use .get for safety
                elif col_name_ho == 'hotspot_avg_temp_change_rate_initial_norm':
                    X_ho_transformed[col_name_ho] = df_ho_features_raw.apply(lambda r: r.get('hotspot_avg_temp_change_rate_initial', np.nan)/r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r.get('hotspot_avg_temp_change_rate_initial', np.nan)) else np.nan, axis=1)
                elif col_name_ho in df_ho_features_raw.columns:
                    X_ho_transformed[col_name_ho] = df_ho_features_raw[col_name_ho]
                else: # Feature not in raw holdout features, set to NaN
                    X_ho_transformed[col_name_ho] = np.nan # Or 0 if appropriate, but NaN is safer for imputer
            X_ho = X_ho_transformed[X_dev.columns].copy() # Ensure order and presence of all columns
            
            y_pred_ho = pipeline_final_dev_model.predict(X_ho)
            r2_ho = r2_score(y_ho, y_pred_ho)
            rmse_ho = np.sqrt(mean_squared_error(y_ho, y_pred_ho))
            print(f"  Hold-Out Performance ({best_model_type_ncv}): R²={r2_ho:.4f}, RMSE={rmse_ho:.4f}")
            if getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', True):
                plotting.plot_actual_vs_predicted(y_ho, y_pred_ho, f"Hold-Out Set - {best_model_type_ncv}", os.path.join(current_output_dir, f"actual_vs_predicted_HOLD_OUT_{best_model_type_ncv}.png"), material_labels=df_ho_features_raw["material"])

            print("\n--- Generating Diagnostic Plots for Best Model ---")
    
            # 1. Permutation Importance Plot (on the final model trained on the dev set)
            perm_importance_path = os.path.join(current_output_dir, f"permutation_importance_{best_model_type_ncv}.png")
            try:
                plotting.plot_permutation_importance(
                    pipeline_final_dev_model, X_dev, y_dev, 
                    feature_names=X_dev.columns.tolist(),
                    save_path=perm_importance_path
                )
                print(f"  Saved permutation importance plot to: {perm_importance_path}")
            except Exception as e:
                print(f"  Error generating permutation importance plot: {e}")

            # 2. Residuals vs. Fitted Plot (using the robust Nested CV predictions)
            best_model_preds = all_models_outer_fold_predictions.get(best_model_type_ncv)
            if best_model_preds:
                residuals_path = os.path.join(current_output_dir, f"residuals_vs_fitted_{best_model_type_ncv}.png")
                try:
                    plotting.plot_residuals_vs_fitted(
                        best_model_preds['true'], 
                        best_model_preds['pred'], 
                        best_model_preds['material'],
                        save_path=residuals_path
                    )
                    print(f"  Saved residuals plot to: {residuals_path}")
                except Exception as e:
                    print(f"  Error generating residuals plot: {e}")

    else:
        print("\n--- No Hold-Out Set was created. Skipping final hold-out evaluation. ---")
    print(f"\n--- Full Evaluation Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full evaluation pipeline for a specific material or combined.")
    parser.add_argument("material_name", type=str,
                        help="Name of the material to run (e.g., 'gypsum', 'brick_cladding') or 'combined'.")
    parser.add_argument("exp_number", type=int,
                        help="An integer for this experiment run.")
    args = parser.parse_args()
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)
    run_full_evaluation_pipeline(
        material_to_run=args.material_name, exp_number=args.exp_number)