# scripts/split_data.py
"""
Splits the master feature CSV into a development (train) and hold-out (test) set.

This script uses StratifiedGroupKFold to create a balanced split based on
binned airflow rates, while ensuring that all samples from the same original
video (e.g., different holes) are kept together in the same set to prevent
data leakage.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_feature_based import config as cfg

def main():
    print("--- Creating STRATIFIED & GROUPED Train/Hold-Out Split ---")
    
    MASTER_CSV_PATH = os.path.join(cfg.OUTPUT_DIR, "master_features.csv")
    TRAIN_CSV_PATH = os.path.join(cfg.OUTPUT_DIR, "train_features.csv")
    HOLDOUT_CSV_PATH = os.path.join(cfg.OUTPUT_DIR, "holdout_features.csv")
    
    if not os.path.exists(MASTER_CSV_PATH):
        print(f"FATAL ERROR: Master feature file not found at '{MASTER_CSV_PATH}'")
        print("Please run 'python -m scripts.generate_features' first.")
        sys.exit(1)
        
    df = pd.read_csv(MASTER_CSV_PATH)

    # --- Stratification & Grouping  ---
    # 1.Discrete bins from the continuous airflow_rate for stratification.
    airflow_bins = pd.cut(df['airflow_rate'], bins=5, labels=False, duplicates='drop')
    
    # 2.Group labels from the original video ID.
    df['original_video_id'] = df['video_id'].apply(lambda x: x.split('_hole_')[0])
    groups = df['original_video_id']
    
    # StratifiedGroupKFold to respect both stratification and groups.
    n_splits = int(np.round(1 / cfg.HOLDOUT_SIZE))
    if n_splits < 2:
        raise ValueError("HOLDOUT_SIZE is too large to create a valid split.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=41)

    try:
        dev_idx, holdout_idx = next(sgkf.split(X=df, y=airflow_bins, groups=groups))
    except ValueError as e:
        print(f"\nCRITICAL ERROR: Could not create a stratified split.")
        print(f"This often happens if a group ('video_id') contains samples from multiple stratification classes, or if a class has fewer members than n_splits ({n_splits}).")
        print(f"Scikit-learn error: {e}")
        return

    dev_df = df.iloc[dev_idx]
    holdout_df = df.iloc[holdout_idx]

    # Dropping the temporary grouping column before saving
    dev_df = dev_df.drop(columns=['original_video_id'])
    holdout_df = holdout_df.drop(columns=['original_video_id'])

    dev_df.to_csv(TRAIN_CSV_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_CSV_PATH, index=False)

    print(f"\nOriginal dataset size: {len(df)} samples")
    print(f"Development (train) set size: {len(dev_df)} samples")
    print(f"Hold-out set size: {len(holdout_df)} samples")
    print(f"\nSaved development features to: {TRAIN_CSV_PATH}")
    print(f"Saved hold-out features to: {HOLDOUT_CSV_PATH}")

    # --- Verification Step ---
    print("\n--- Verifying Split Distribution ---")
    dev_counts = dev_df['airflow_rate'].value_counts().sort_index()
    holdout_counts = holdout_df['airflow_rate'].value_counts().sort_index()
    
    verification_df = pd.DataFrame({
        'dev_set_count': dev_counts,
        'holdout_set_count': holdout_counts
    }).fillna(0).astype(int)
    
    print("\nOverall distribution of samples per airflow rate:")
    print(verification_df)

if __name__ == "__main__":
    main()

# python -m src_feature_based.split_data