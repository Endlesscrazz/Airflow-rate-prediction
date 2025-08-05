# select_features.py
"""
(Corrected and Integrated Version)
A script to perform automated feature selection using RFECV.
- Uses a dedicated preprocess_features function for cleaning the data.
- Imports the master feature list from feature_engineering.py.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import fnmatch

# --- Scikit-learn Imports ---
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# --- Import Your Project Modules ---
try:
    import config
    import data_utils
    import feature_engineering
except ImportError:
    print("Error: Could not import project modules.", file=sys.stderr)
    sys.exit(1)

# ========================================================================================
# --- HELPER FUNCTIONS ---
# ========================================================================================


def preprocess_features(df: pd.DataFrame, corr_thresh: float = 0.95) -> (np.ndarray, list): # type: ignore
    """
    1) (Optional) drop one of any pair of features whose |corr| > corr_thresh
    2) Impute any missing values
    3) Scale everything to zero-mean/unit-var
    Returns: transformed numpy array, list of kept column names
    """
    X = df.copy()
    
    # --- Drop highly-correlated features ---
    if corr_thresh < 1.0:
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_thresh)]
        if to_drop:
            print(f"Dropping {len(to_drop)} highly-correlated features: {to_drop}")
            X = X.drop(columns=to_drop)
    
    kept_columns = X.columns.tolist()

    # --- Impute + Scale pipeline ---
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    
    print("Applying imputer and scaler...")
    X_processed = preprocessor.fit_transform(X)

    return X_processed, kept_columns

def load_and_process_all_data():
    """
    Loads all datasets, applies the aggregate-features strategy, and extracts
    every possible feature for the selection process.
    """
    DATASET_CONFIGS = {
        "gypsum_single_hole": {"material": "gypsum", "dataset_subfolder": "dataset_gypsum", "mask_subfolder": "dataset_gypsum"},
        "gypsum_single_hole2": {"material": "gypsum", "dataset_subfolder": "dataset_gypsum2", "mask_subfolder": "dataset_gypsum2"},
        "brick_cladding_single_hole": {"material": "brick_cladding", "dataset_subfolder": "dataset_brickcladding", "mask_subfolder": "dataset_brickcladding"},
        "brick_cladding_two_holes": {"material": "brick_cladding", "dataset_subfolder": "dataset_two_holes_brickcladding", "mask_subfolder": "dataset_two_holes_brickcladding"}
    }
    video_to_masks_map = {}
    for d_name, d_config in DATASET_CONFIGS.items():
        print(f"\nScanning data from dataset: '{d_name}'...")
        dataset_path_load = os.path.join(config.DATASET_PARENT_DIR, d_config["dataset_subfolder"])
        mask_root_path_load = os.path.join(config.BASE_MASK_INPUT_DIR, d_config["mask_subfolder"])
        if not os.path.isdir(dataset_path_load) or not os.path.isdir(mask_root_path_load):
            print(f"  Warning: Path not found. Skipping."); continue
        for root_load, dirs_load, files_load in os.walk(dataset_path_load):
            if "cooling" in dirs_load: dirs_load.remove("cooling")
            for mat_filename_load in fnmatch.filter(files_load, '*.mat'):
                mat_filepath_load = os.path.join(root_load, mat_filename_load)
                video_id = os.path.splitext(mat_filename_load)[0]
                if video_id not in video_to_masks_map:
                    try:
                        folder_name_load = os.path.basename(os.path.dirname(mat_filepath_load))
                        airflow_load = data_utils.parse_airflow_rate(folder_name_load)
                        delta_t_load = data_utils.parse_delta_T(mat_filename_load)
                        if delta_t_load is None: continue
                        video_to_masks_map[video_id] = {"mat_filepath": mat_filepath_load, "mask_paths": [], "delta_T": float(delta_t_load), "airflow_rate": float(airflow_load), "material": d_config["material"]}
                    except Exception: continue
                mat_basename = os.path.splitext(mat_filename_load)[0]
                relative_path_part = os.path.relpath(root_load, dataset_path_load)
                mask_search_dir = os.path.join(mask_root_path_load, relative_path_part, mat_basename)
                if os.path.isdir(mask_search_dir):
                    mask_files = fnmatch.filter(os.listdir(mask_search_dir), f"{mat_basename}_mask_*.npy")
                    mask_files.extend(fnmatch.filter(os.listdir(mask_search_dir), f"{mat_basename}_sam_mask.npy"))
                    for mask_filename in mask_files:
                        video_to_masks_map[video_id]["mask_paths"].append(os.path.join(mask_search_dir, mask_filename))
    
    all_samples_info_list = [v for v in video_to_masks_map.values() if v["mask_paths"]]
    df_all_samples_raw_info = pd.DataFrame(all_samples_info_list)
    print(f"\nTotal video samples loaded: {len(df_all_samples_raw_info)}")
    
    feature_list = []
    for _, row in tqdm(df_all_samples_raw_info.iterrows(), total=len(df_all_samples_raw_info), desc="Extracting All Features"):
        extracted_dict = feature_engineering.extract_aggregate_features(
            row["mat_filepath"], row["mask_paths"], fps=config.MASK_FPS,
            focus_duration_sec=config.MASK_FOCUS_DURATION_SEC, envir_para=config.MASK_ENVIR_PARA,
            threshold_abs_change=config.FIXED_AREA_THRESHOLD)
        current_feats = {"delta_T": row["delta_T"], "airflow_rate": row["airflow_rate"], "material": row["material"]}
        if extracted_dict: current_feats.update(extracted_dict)
        feature_list.append(current_feats)
        
    df_features_raw = pd.DataFrame(feature_list)
    y = df_features_raw["airflow_rate"].astype(float)
    X_transformed = pd.DataFrame(index=df_features_raw.index)
    
    # Use the master list from the feature_engineering module
    all_feature_names = feature_engineering.ALL_POSSIBLE_FEATURE_NAMES
    
    X_transformed['delta_T_log'] = np.log1p(df_features_raw['delta_T'].astype(float).clip(lower=0))
    X_transformed['hotspot_area_log'] = np.log1p(df_features_raw.get('hotspot_area', 0).astype(float).clip(lower=0))
    X_transformed['hotspot_avg_temp_change_rate_initial_norm'] = df_features_raw.apply(
        lambda r: r.get('hotspot_avg_temp_change_rate_initial', np.nan)/r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r.get('hotspot_avg_temp_change_rate_initial', np.nan)) else np.nan, axis=1)
    
    special_features = ['hotspot_area', 'hotspot_avg_temp_change_rate_initial', 'delta_T']
    for f_name in all_feature_names:
        if f_name not in special_features and f_name in df_features_raw.columns:
            X_transformed[f_name] = df_features_raw[f_name]
            
    mat_dummies = pd.get_dummies(df_features_raw['material'], prefix='material', dtype=int)
    X_transformed = pd.concat([X_transformed, mat_dummies], axis=1)
    
    return X_transformed, y

# ========================================================================================
# --- MAIN EXECUTION ---
# ========================================================================================

if __name__ == "__main__":
    print("--- Automatic Feature Selection using RFECV ---")

    X_full, y_full = load_and_process_all_data()
    print(f"\nPrepared full feature matrix with shape: {X_full.shape}")

    # --- CORRECTED INTEGRATION ---
    # 1. Use your new function to preprocess the full feature set
    X_prepared, kept_feature_names = preprocess_features(X_full, corr_thresh=0.95)
    
    # 2. Define the estimator as the raw XGBRegressor model
    estimator = XGBRegressor(
        n_estimators=100, 
        max_depth=5,
        random_state=config.RANDOM_STATE, 
        n_jobs=-1
    )
    # --- END OF CORRECTION ---

    airflow_bins = pd.cut(y_full, bins=5, labels=False, duplicates='drop')
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    min_features_to_select = 3

    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=cv_splitter,
        scoring='r2',
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
        verbose=1
    )

    print("\nRunning RFECV... (This may take several minutes)")
    selector.fit(X_prepared, airflow_bins)

    print("\n--- RFECV Results ---")
    print(f"Optimal number of features found: {selector.n_features_}")
    
    # Get the selected feature names from the list of columns that were kept after correlation filtering
    selected_features = np.array(kept_feature_names)[selector.support_].tolist()
    print("\nOptimal feature set:")
    for i, f in enumerate(selected_features):
        print(f"  {i+1}. {f}")

    # (Plotting logic remains the same)
    n_scores = len(selector.cv_results_["mean_test_score"])
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (R²)")
    plt.plot(range(min_features_to_select, n_scores + min_features_to_select), selector.cv_results_["mean_test_score"])
    optimal_n = selector.n_features_
    optimal_score_idx = optimal_n - min_features_to_select
    if 0 <= optimal_score_idx < len(selector.cv_results_["mean_test_score"]):
        optimal_score = selector.cv_results_["mean_test_score"][optimal_score_idx]
        plt.plot(optimal_n, optimal_score, 'ro', markersize=8,
                 label=f'Optimal Point ({optimal_n} features, R²={optimal_score:.3f})')
    plt.title("RFECV Performance vs. Number of Features")
    plt.legend()
    plt.grid()
    output_plot_path = "feature_selection_rfecv_results.png"
    plt.savefig(output_plot_path)
    print(f"\nSaved RFECV results plot to: {output_plot_path}")
    plt.show()