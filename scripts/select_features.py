# scripts/select_features.py
"""
Performs automated feature selection using RFECV on a pre-computed master
feature CSV file.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Add project root to path for imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Scikit-learn Imports ---
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# --- Import Your Project Modules ---
from src_cnn import config as cfg

def preprocess_features(df: pd.DataFrame, corr_thresh: float = 0.95) -> (pd.DataFrame, list): # type: ignore
    """Cleans the feature dataframe."""
    X = df.copy()
    
    if corr_thresh < 1.0:
        corr_matrix = X.corr(numeric_only=True).abs()
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
    print("--- Automatic Feature Selection using RFECV ---")

    MASTER_CSV_PATH = os.path.join(project_root, "master_features.csv")
    if not os.path.exists(MASTER_CSV_PATH):
        print(f"Error: Master feature file not found at '{MASTER_CSV_PATH}'.")
        print("Please run 'python -m scripts.generate_master_features' first.")
        sys.exit(1)
        
    df_master = pd.read_csv(MASTER_CSV_PATH)
    
    y_full = df_master['airflow_rate']
    X_full = df_master.drop(columns=['airflow_rate', 'video_id', 'source_dataset'])
    
    # --- Feature Transformation ---
    X_transformed = pd.DataFrame(index=X_full.index)
    
    if 'delta_T' in X_full.columns: X_transformed['delta_T_log'] = np.log1p(X_full['delta_T'])
    if 'hotspot_area' in X_full.columns and cfg.LOG_TRANSFORM_AREA: X_transformed['hotspot_area_log'] = np.log1p(X_full['hotspot_area'])
    if 'hotspot_avg_temp_change_rate_initial' in X_full.columns and cfg.NORMALIZE_AVG_RATE_INITIAL:
        X_transformed['hotspot_avg_temp_change_rate_initial_norm'] = X_full.apply(
            lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 else np.nan, axis=1
        ).fillna(0)
    
    special_raw_features = ['delta_T', 'hotspot_area', 'hotspot_avg_temp_change_rate_initial', 'material']
    other_features = [col for col in X_full.columns if col not in special_raw_features]
    X_transformed = pd.concat([X_transformed, X_full[other_features]], axis=1)
    X_transformed = pd.concat([X_transformed, pd.get_dummies(X_full['material'], prefix='material', dtype=int)], axis=1)
    
    print(f"\nPrepared full feature matrix with shape: {X_transformed.shape}")

    X_prepared, kept_feature_names = preprocess_features(X_transformed, corr_thresh=0.95)
    
    estimator = XGBRegressor(n_estimators=100, max_depth=5, random_state=cfg.RANDOM_STATE, n_jobs=-1)
    
    # --- START OF FIX ---
    # 1. Create discrete bins from the continuous target for stratification.
    airflow_bins = pd.cut(y_full, bins=5, labels=False, duplicates='drop')
    
    # 2. Manually create the stratified folds using the bins.
    #    This generates a list of (train_indices, test_indices).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    cv_iterator = list(skf.split(X_prepared, airflow_bins))

    min_features_to_select = 3

    # 3. Pass the pre-computed list of folds directly to RFECV via the 'cv' parameter.
    selector = RFECV(
        estimator=estimator, step=1, cv=cv_iterator, scoring='r2',
        min_features_to_select=min_features_to_select, n_jobs=-1, verbose=1
    )

    print("\nRunning RFECV... (This may take several minutes)")
    # 4. Fit the selector. Now it will use our custom folds and train on the continuous y_full.
    selector.fit(X_prepared.values, y_full.values)
    # --- END OF FIX ---

    print("\n--- RFECV Results ---")
    print(f"Optimal number of features found: {selector.n_features_}")
    
    selected_features = np.array(kept_feature_names)[selector.support_].tolist()
    print("\nOptimal feature set:")
    for i, f in enumerate(selected_features):
        print(f"  {i+1}. {f}")

    # --- Plotting Results ---
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
    output_plot_path = os.path.join(project_root, "feature_selection_rfecv_results.png")
    plt.savefig(output_plot_path)
    print(f"\nSaved RFECV results plot to: {output_plot_path}")
    plt.show()