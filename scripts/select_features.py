# scripts/select_features.py
"""
Performs an automated feature selection process using one of three robust,
cross-validated methods to combat overfitting on small datasets.

METHOD 1: 'xgboost' (Original)
- Uses the average feature importance from a cross-validated XGBoost model.
- Powerful but can be prone to overfitting the feature selection process itself on small N.

METHOD 2: 'lasso' (Strategy 2 - Regularization)
- Uses LassoCV, which performs L1 regularization to shrink the coefficients of
  unimportant features to exactly zero.
- Excellent for creating sparse, simple models and is inherently resistant to overfitting.
- The "importance" is the magnitude of the final coefficients.

METHOD 3: 'rfecv' (Strategy 3 - Recursive Feature Elimination)
- Uses Recursive Feature Elimination with Cross-Validation to find the optimal
  number of features by repeatedly training a model and removing the weakest one.
- Very robust and provides a diagnostic plot showing performance vs. number of features.
- Uses a simple Ridge estimator to keep the process stable.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple
import sklearn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

from src_feature_based import config as cfg

# This is required to pass the 'groups' parameter to nested estimators in a pipeline.
sklearn.set_config(enable_metadata_routing=True)

def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Loads the master features CSV and prepares the full feature matrix (X)
       and target vector (y), applying all necessary transformations."""
    MASTER_CSV_PATH = os.path.join(cfg.OUTPUT_DIR, "master_features.csv")
    if not os.path.exists(MASTER_CSV_PATH):
        sys.exit(f"Error: Master feature file not found. Run generate_features.py first.")
        
    df_master = pd.read_csv(MASTER_CSV_PATH)
    y_full = df_master['airflow_rate']
    groups = df_master['video_id'].apply(lambda x: x.split('_hole_')[0])
    X_full = df_master.drop(columns=['airflow_rate', 'video_id', 'hole_id','voltage','pressure_Pa'])
    
    # --- Create a Comprehensive Feature Set ---
    X_transformed = pd.DataFrame(index=X_full.index)
    
    # Perform transformations as defined in config
    if 'delta_T' in X_full.columns:
        X_transformed['delta_T_log'] = np.log1p(X_full['delta_T'])
    if 'hotspot_area' in X_full.columns and cfg.LOG_TRANSFORM_AREA:
        X_transformed['hotspot_area_log'] = np.log1p(X_full['hotspot_area'])
    if 'hotspot_avg_temp_change_rate_initial' in X_full.columns and cfg.NORMALIZE_AVG_RATE_INITIAL:
        X_transformed['hotspot_avg_temp_change_rate_initial_norm'] = X_full.apply(
            lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and pd.notna(r.get('hotspot_avg_temp_change_rate_initial')) else np.nan, axis=1
        )
        
    if cfg.NORMALIZE_CUMULATIVE_FEATURES:
        print("Normalizing new cumulative features by delta_T...")
        features_to_normalize = [
            'cumulative_raw_delta_sum', 'cumulative_abs_delta_sum',
            'auc_mean_temp_delta', 'mean_pixel_volatility'
        ]
        for feature in features_to_normalize:
            if feature in X_full.columns:
                new_col_name = f"{feature}_norm"
                X_transformed[new_col_name] = X_full.apply(
                    lambda r: r[feature] / r['delta_T'] if r['delta_T'] != 0 and pd.notna(r[feature]) else np.nan, axis=1
                )

    # Define raw columns to exclude
    special_raw = ['delta_T', 'hotspot_area', 'hotspot_avg_temp_change_rate_initial', 'material', 'session',
                   'cumulative_raw_delta_sum', 'cumulative_abs_delta_sum', 'auc_mean_temp_delta', 'mean_pixel_volatility']
    other_features = [col for col in X_full.columns if col not in special_raw]
    X_transformed = pd.concat([X_transformed, X_full[other_features]], axis=1)
    
    # Create dummies
    material_dummies = pd.get_dummies(X_full['material'], prefix='material', dtype=int)
    X_transformed = pd.concat([X_transformed, material_dummies], axis=1)
    # if 'session' in X_full.columns:
    #     session_dummies = pd.get_dummies(X_full['session'], prefix='session', dtype=int)
    #     X_transformed = pd.concat([X_transformed, session_dummies], axis=1)
    
    print(f"\nFull feature matrix shape after transformations: {X_transformed.shape}")
    X_transformed.fillna(X_transformed.median(numeric_only=True), inplace=True)
    
    return X_transformed, y_full, groups

def select_with_xgboost(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv_splitter) -> Tuple[List[str], pd.DataFrame]:
    """Selects features based on average importance from a cross-validated XGBoost model."""
    print("\nGetting feature importances using XGBoost with StratifiedGroupKFold CV...")
    
    all_feature_importances_df = pd.DataFrame(index=X.columns)
    airflow_bins = pd.cut(y, bins=5, labels=False, duplicates='drop')

    for fold, (train_idx, _) in enumerate(cv_splitter.split(X, airflow_bins, groups)):
        print(f"  - Processing Fold {fold+1}/{cv_splitter.get_n_splits()}...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

        preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler",  StandardScaler())])
        X_train_prep = preprocessor.fit_transform(X_train)
        
        xgb_selector = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=cfg.RANDOM_STATE, n_jobs=-1)
        xgb_selector.fit(X_train_prep, np.log1p(y_train))
        
        fold_importances = pd.Series(xgb_selector.feature_importances_, index=X.columns, name=f"fold_{fold}")
        all_feature_importances_df = pd.concat([all_feature_importances_df, fold_importances], axis=1)

    feature_importance_df = pd.DataFrame({
        'feature': all_feature_importances_df.index,
        'importance': all_feature_importances_df.mean(axis=1)
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    N_TOP_FEATURES = 15
    selected_features = feature_importance_df.head(N_TOP_FEATURES)['feature'].tolist()
    return selected_features, feature_importance_df

def select_with_lasso(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv_splitter) -> Tuple[List[str], pd.DataFrame]:
    """Selects features using LassoCV, which shrinks unimportant coefficients to zero."""
    print("\nSelecting features using LassoCV with GroupKFold CV...")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=cv_splitter, random_state=cfg.RANDOM_STATE, n_jobs=-1, max_iter=5000))
    ])
    
    # With metadata routing enabled, we can now pass 'groups' directly to fit.
    pipeline.fit(X, np.log1p(y), groups=groups)
    
    coefficients = pipeline.named_steps['lasso'].coef_
    importance = np.abs(coefficients)
    
    selected_mask = importance > 1e-5
    selected_features = X.columns[selected_mask].tolist()

    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    print(f"\nLassoCV selected {len(selected_features)} features with non-zero coefficients.")
    return selected_features, feature_importance_df

def select_with_rfecv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv_splitter) -> Tuple[List[str], pd.DataFrame]:
    """Selects features using Recursive Feature Elimination with Cross-Validation."""
    print("\nSelecting features using RFECV with GroupKFold CV...")

    # Use RidgeCV as the estimator. This allows the model to find the best
    # regularization strength (alpha) at each step of the elimination,
    estimator = RidgeCV(alphas=np.logspace(-3, 3, 7))
    
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv_splitter,
        scoring='r2',
        min_features_to_select=5, 
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rfecv", rfecv)
    ])

    pipeline.fit(X, np.log1p(y), groups=groups)
    
    num_selected = pipeline.named_steps['rfecv'].n_features_
    print(f"\nRFECV selected {num_selected} as the optimal number of features.")

    # Plot the RFECV performance curve
    n_scores = len(pipeline.named_steps['rfecv'].cv_results_['mean_test_score'])
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (R²)")
    min_features = pipeline.named_steps['rfecv'].min_features_to_select
    # The x-axis should represent the number of features at each step of the elimination
    x_axis = np.arange(X.shape[1], X.shape[1] - n_scores, -1)[::-1]
    # Ensure x_axis and scores have the same length
    scores = pipeline.named_steps['rfecv'].cv_results_['mean_test_score']
    if len(x_axis) > len(scores):
        x_axis = x_axis[:len(scores)]

    plt.plot(x_axis, scores)
    # Add a vertical line at the optimal number of features
    plt.axvline(x=num_selected, color='r', linestyle='--', label=f'Optimal features: {num_selected}')
    plt.title("RFECV: Performance vs. Number of Features")
    plt.legend()
    plt.grid()
    rfecv_plot_path = os.path.join(project_root, "rfecv_performance_curve.png")
    plt.savefig(rfecv_plot_path)
    print(f"Saved RFECV performance curve to: {rfecv_plot_path}")
    plt.close()

    # --- CORRECTED IMPORTANCE CALCULATION ---
    selected_mask = pipeline.named_steps['rfecv'].support_
    selected_features = X.columns[selected_mask].tolist()

    # To get a meaningful importance score, we fit the final estimator
    # one last time on only the selected features.
    final_estimator = pipeline.named_steps['rfecv'].estimator_
    X_selected = X[selected_features]
    
    # Create a simple pipeline for the final fit
    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", final_estimator)
    ])
    final_pipeline.fit(X_selected, np.log1p(y))

    # The importance is the absolute coefficient from this final model
    importance_values = np.abs(final_pipeline.named_steps['model'].coef_)
    
    # Create a full importance series, with 0 for non-selected features
    full_importance = pd.Series(0.0, index=X.columns)
    full_importance[selected_features] = importance_values

    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': full_importance
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    return selected_features, feature_importance_df

def main():
    parser = argparse.ArgumentParser(description="Run cross-validated feature selection.")
    parser.add_argument(
        "--method", 
        type=str, 
        default="lasso", 
        choices=['xgboost', 'lasso', 'rfecv'],
        help="The feature selection method to use."
    )
    args = parser.parse_args()

    print(f"--- Running Feature Selection with Method: '{args.method.upper()}' ---")

    X_transformed, y_full, groups = load_and_prepare_data()
    
    selected_features = []
    feature_importance_df = pd.DataFrame()

    if args.method == 'xgboost':
        cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
        selected_features, feature_importance_df = select_with_xgboost(X_transformed, y_full, groups, cv_splitter)
    elif args.method == 'lasso':
        cv_splitter = GroupKFold(n_splits=5)
        selected_features, feature_importance_df = select_with_lasso(X_transformed, y_full, groups, cv_splitter)
    elif args.method == 'rfecv':
        cv_splitter = GroupKFold(n_splits=5)
        selected_features, feature_importance_df = select_with_rfecv(X_transformed, y_full, groups, cv_splitter)

    # --- Print and Plot Final Results ---
    print("\n--- Top 20 Most Important Features ---")
    print(feature_importance_df.head(20).to_string())

    ordered_selected_features = feature_importance_df[feature_importance_df['importance'] > 1e-5]['feature'].tolist()
    
    print(f"\n\nOptimal feature set with {len(ordered_selected_features)} features (COPY THIS TO config.py):")
    print("SELECTED_FEATURES = [")
    for f in ordered_selected_features: 
        print(f"    '{f}',")
    print("]")

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['feature'].head(20), feature_importance_df['importance'].head(20))
    plt.xlabel(f"Feature Importance ({args.method.capitalize()} method)")
    plt.ylabel("Feature")
    plt.title(f"Top 20 Features Discovered via {args.method.capitalize()} CV")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    output_plot_path = os.path.join(project_root, f"feature_importance_selection_{args.method}.png")
    plt.savefig(output_plot_path)
    print(f"\nSaved feature importance plot to: {output_plot_path}")
    plt.show()

if __name__ == "__main__":
    main()

# python -m scripts.select_features --method rfecv
