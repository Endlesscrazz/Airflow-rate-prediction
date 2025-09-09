# scripts/select_features.py
"""
Performs automated feature selection using RFECV on a pre-computed master
feature CSV file. This version includes an expanded set of interaction features.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src_feature_based import config as cfg

def preprocess_features(df: pd.DataFrame, corr_thresh: float = 0.95) -> (pd.DataFrame, list): # type: ignore
    """Cleans the feature dataframe."""
    X = df.copy()
    
    if corr_thresh < 1.0:
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        corr_matrix = X[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_thresh)]
        if to_drop:
            print(f"Dropping {len(to_drop)} highly-correlated features: {to_drop}")
            X = X.drop(columns=to_drop)
    
    kept_columns = X.columns.tolist()
    
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    
    print("Applying imputer and scaler...")
    X_processed = preprocessor.fit_transform(X)
    return pd.DataFrame(X_processed, columns=kept_columns, index=X.index), kept_columns

if __name__ == "__main__":
    print("--- Automatic Feature Selection using RFECV (with Interaction Features) ---")

    MASTER_CSV_PATH = os.path.join(cfg.OUTPUT_DIR, "master_features.csv")
    if not os.path.exists(MASTER_CSV_PATH):
        print(f"Error: Master feature file not found at '{MASTER_CSV_PATH}'.")
        print("Please run 'python -m scripts.generate_features' first.")
        sys.exit(1)
        
    df_master = pd.read_csv(MASTER_CSV_PATH)
    
    y_full = df_master['airflow_rate']

    X_full = df_master.drop(columns=['airflow_rate', 'video_id', 'hole_id','voltage','pressure_Pa'])
    
    # --- Feature Transformation ---
    X_transformed = pd.DataFrame(index=X_full.index)
    
    if 'delta_T' in X_full.columns:
        X_transformed['delta_T_log'] = np.log1p(X_full['delta_T'])
    if 'hotspot_area' in X_full.columns and cfg.LOG_TRANSFORM_AREA:
        X_transformed['hotspot_area_log'] = np.log1p(X_full['hotspot_area'])
    if 'hotspot_avg_temp_change_rate_initial' in X_full.columns and cfg.NORMALIZE_AVG_RATE_INITIAL:
        X_transformed['hotspot_avg_temp_change_rate_initial_norm'] = X_full.apply(
            lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and pd.notna(r.get('hotspot_avg_temp_change_rate_initial')) else np.nan, axis=1
        )
    
    special_raw_features = ['delta_T', 'hotspot_area', 'hotspot_avg_temp_change_rate_initial', 'material']
    other_features = [col for col in X_full.columns if col not in special_raw_features]
    X_transformed = pd.concat([X_transformed, X_full[other_features]], axis=1)
    
    X_transformed = pd.concat([X_transformed, pd.get_dummies(X_full['material'], prefix='material', dtype=int)], axis=1)
    
    print(f"\nPrepared base feature matrix with shape: {X_transformed.shape}")

    # --- CREATE INTERACTION FEATURES ---
    print("\nCreating interaction features for selection...")
    # EXPANDED LIST of physically-meaningful features to interact with material type
    features_to_interact = [
        'hotspot_avg_temp_change_rate_initial_norm',
        'hotspot_avg_temp_change_magnitude_initial', 
        'temperature_kurtosis', 
        'temp_std_avg_initial', 
        #'mean_area_significant_change',
        'hotspot_area_log', 
        'hotspot_solidity', 
        'temperature_skewness'
    ]
    material_cols = [col for col in X_transformed.columns if col.startswith('material_')]
    
    interaction_count = 0
    for feature in features_to_interact:
        if feature in X_transformed.columns:
            for material_col in material_cols:
                interaction_col_name = f"{feature}_x_{material_col.split('_')[-1]}"
                X_transformed[interaction_col_name] = X_transformed[feature] * X_transformed[material_col]
                interaction_count += 1
    
    if interaction_count > 0:
        print(f"Added {interaction_count} new interaction features.")
        print(f"Full feature matrix shape is now: {X_transformed.shape}")
    # --- END OF INTERACTION BLOCK ---

    X_transformed.fillna(X_transformed.median(numeric_only=True), inplace=True)
    
    X_prepared, kept_feature_names = preprocess_features(X_transformed, corr_thresh=0.95)
    
    estimator = XGBRegressor(n_estimators=100, max_depth=5, random_state=cfg.RANDOM_STATE, n_jobs=-1)
    
    airflow_bins = pd.cut(y_full, bins=5, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    cv_iterator = list(skf.split(X_prepared, airflow_bins))

    min_features_to_select = 3
    selector = RFECV(
        estimator=estimator, step=1, cv=cv_iterator, scoring='r2',
        min_features_to_select=min_features_to_select, n_jobs=-1, verbose=1
    )

    print("\nRunning RFECV... (This may take several minutes)")
    selector.fit(X_prepared.values, y_full.values)

    print("\n--- RFECV Results ---")
    print(f"Optimal number of features found: {selector.n_features_}")
    
    selected_features = np.array(kept_feature_names)[selector.support_].tolist()
    print("\nOptimal feature set (COPY THIS LIST TO YOUR config.py):")
    print("SELECTED_FEATURES = [")
    for f in selected_features:
        print(f"    '{f}',")
    print("]")

    # --- Plotting Results ---
    n_scores = len(selector.cv_results_["mean_test_score"])
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (R²)")
    plt.plot(range(min_features_to_select, n_scores + min_features_to_select), selector.cv_results_["mean_test_score"])
    
    optimal_n = selector.n_features_
    if min_features_to_select <= optimal_n <= n_scores + min_features_to_select -1:
        optimal_score_idx = optimal_n - min_features_to_select
        optimal_score = selector.cv_results_["mean_test_score"][optimal_score_idx]
        plt.plot(optimal_n, optimal_score, 'ro', markersize=8,
                 label=f'Optimal Point ({optimal_n} features, R²={optimal_score:.3f})')
        
    plt.title("RFECV Performance vs. Number of Features")
    plt.legend()
    plt.grid()
    output_plot_path = os.path.join(project_root, "feature_selection_rfecv_results.png")
    plt.savefig(output_plot_path)
    print(f"\nSaved RFECV results plot to: {output_plot_path}")
    plt.show()

# python -m scripts.select_features