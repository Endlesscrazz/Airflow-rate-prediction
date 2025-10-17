# scripts/select_features.py
"""
Performs automated feature selection on the master_features.csv to find the
most predictive set of handcrafted features.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

from src_cnn import config as cfg

# --- NEW: Centralized Feature Preparation Function ---
# For perfect code, this function should be in a new file like `src_cnn/data_utils.py`
# and imported here. For now, defining it here solves the consistency problem.
def prepare_tabular_features(df_master, X_raw):
    """Applies all required transformations to the raw feature data."""
    X_transformed = pd.DataFrame(index=X_raw.index)
    
    # Log transform delta_T, which is a base feature
    if 'delta_T' in df_master.columns:
        X_transformed['delta_T_log'] = np.log1p(df_master['delta_T'])
    
    # Apply log transform for area
    if cfg.LOG_TRANSFORM_AREA and 'hotspot_area' in X_raw.columns:
        X_transformed['hotspot_area_log'] = np.log1p(X_raw['hotspot_area'])
        X_raw = X_raw.drop(columns=['hotspot_area'], errors='ignore')

    # Apply normalization for initial rate
    if cfg.NORMALIZE_AVG_RATE_INITIAL and 'hotspot_avg_temp_change_rate_initial' in X_raw.columns:
        # Use df_master to access delta_T for normalization
        X_transformed['hotspot_avg_temp_change_rate_initial_norm'] = X_raw.apply(
            lambda r, dt=df_master['delta_T']: r['hotspot_avg_temp_change_rate_initial'] / dt[r.name] if dt[r.name] != 0 else 0, axis=1
        )
        X_raw = X_raw.drop(columns=['hotspot_avg_temp_change_rate_initial'], errors='ignore')
        
    # Apply normalization for cumulative features
    if cfg.NORMALIZE_CUMULATIVE_FEATURES:
        features_to_normalize = [
            'cumulative_raw_delta_sum', 'cumulative_abs_delta_sum',
            'auc_mean_temp_delta', 'mean_pixel_volatility'
        ]
        for feature in features_to_normalize:
            if feature in X_raw.columns:
                new_col_name = f"{feature}_norm"
                # Use df_master to access delta_T for normalization
                X_transformed[new_col_name] = X_raw.apply(
                    lambda r, dt=df_master['delta_T']: r[feature] / dt[r.name] if dt[r.name] != 0 and pd.notna(r.get(feature)) else np.nan, axis=1
                )
                X_raw = X_raw.drop(columns=[feature], errors='ignore')
    
    # Combine transformed features with the rest of the raw features
    X_final = pd.concat([X_transformed, X_raw], axis=1)
    
    # Add material dummies
    if 'material' in df_master.columns:
        material_dummies = pd.get_dummies(df_master['material'], prefix='material', dtype=int)
        X_final = pd.concat([X_final, material_dummies], axis=1)
        
    return X_final

def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Loads master_features.csv and prepares X and y for selection."""
    MASTER_CSV_PATH = cfg.MASTER_FEATURES_PATH
    if not os.path.exists(MASTER_CSV_PATH):
        sys.exit(f"Error: Master feature file not found at '{MASTER_CSV_PATH}'. Run generate_master_features.py first.")
        
    df_master = pd.read_csv(MASTER_CSV_PATH)
    
    y = df_master['airflow_rate']
    
    # --- FIX: Added 'session' to the list of columns to drop ---
    cols_to_drop = [
        'sample_id', 'video_id', 'hole_id', 'airflow_rate', 
        'material', 'voltage', 'pressure_Pa', 'session'
    ]
    # Drop columns that exist, ignore if they don't
    X_raw = df_master.drop(columns=[col for col in cols_to_drop if col in df_master.columns])
    
    # Use the centralized function to perform all transformations
    X_final = prepare_tabular_features(df_master, X_raw)
    
    print(f"\nFull feature matrix shape for selection: {X_final.shape}")
    
    return X_final, y

def select_with_kbest(X: pd.DataFrame, y: pd.Series, k: int) -> Tuple[List[str], pd.DataFrame]:
    """Selects the top K features using univariate F-test regression."""
    print(f"\nSelecting top {k} features using SelectKBest (f_regression)...")
    
    # Impute NaNs that may have been created during normalization
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_regression, k=k))
    ])
    
    pipeline.fit(X_imputed, y)
    
    selector = pipeline.named_steps['selector']
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'f_score': selector.scores_
    }).sort_values(by='f_score', ascending=False, na_position='last').reset_index(drop=True)
    
    return selected_features, feature_importance_df

def main():
    parser = argparse.ArgumentParser(description="Run automated feature selection.")
    parser.add_argument("--k", type=int, default=15, help="Number of top features to select.")
    args = parser.parse_args()

    print(f"--- Running Feature Selection ---")

    X_final, y = load_and_prepare_data()
    
    selected_features, feature_importance_df = select_with_kbest(X_final, y, k=args.k)
    
    print("\n--- Top 25 Feature Scores (Higher is Better) ---")
    print(feature_importance_df.head(25).to_string())

    print(f"\n\n--- Optimal feature set with {len(selected_features)} features (COPY THIS TO config.py) ---")
    print("\nDYNAMIC_FEATURES = [")
    for f in selected_features: 
        # Exclude context features from the dynamic list
        if f not in cfg.CONTEXT_FEATURES and not f.startswith('material_'):
            print(f"    '{f}',")
    print("]")

    plt.figure(figsize=(10, 8))
    top_20 = feature_importance_df.head(20)
    plt.barh(top_20['feature'], top_20['f_score'])
    plt.xlabel("F-Score (Importance)")
    plt.ylabel("Feature")
    plt.title(f"Top 20 Features from SelectKBest")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    output_plot_path = os.path.join(cfg.OUTPUT_DIR, "feature_importance.png")
    plt.savefig(output_plot_path)
    print(f"\nSaved feature importance plot to: {output_plot_path}")
    plt.close()

if __name__ == "__main__":
    main()

# python src_cnn/select_features.py 