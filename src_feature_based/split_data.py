# scripts/split_data.py
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold # <-- This is the correct splitter
import os
import sys
import numpy as np
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_feature_based import config as cfg
from src_feature_based.utils import setup_logging

def main(args):
    setup_logging(output_dir=cfg.OUTPUT_DIR, script_name="split_data")
    
    print("--- Creating STRATIFIED & GROUPED Train/Hold-Out Split ---")
    print(f"Using Random State: {args.random_state}")
    
    MASTER_CSV_PATH = os.path.join(cfg.OUTPUT_DIR, "master_features.csv")
    TRAIN_CSV_PATH = os.path.join(cfg.OUTPUT_DIR, "train_features.csv")
    HOLDOUT_CSV_PATH = os.path.join(cfg.OUTPUT_DIR, "holdout_features.csv")
    
    if not os.path.exists(MASTER_CSV_PATH):
        sys.exit(f"FATAL ERROR: Master feature file not found at '{MASTER_CSV_PATH}'")
        
    df = pd.read_csv(MASTER_CSV_PATH)

    # --- Stratification & Grouping ---
    # Bin the continuous airflow_rate for stratification
    airflow_bins = pd.cut(df['airflow_rate'], bins=5, labels=False, duplicates='drop')
    
    # This ensures that both holes from a single video recording are kept together
    # in either the train or holdout set, preventing data leakage.
    df['base_video_id'] = df['video_id'].apply(lambda x: x.split('_hole_')[0])
    groups = df['base_video_id']
    print("\nUsing 'base_video_id' for grouping to keep holes from the same video together.")
    
    n_splits = int(np.round(1 / cfg.HOLDOUT_SIZE))
    if n_splits < 2:
        raise ValueError("HOLDOUT_SIZE is too large to create a valid split.")

    # Use StratifiedGroupKFold to balance airflow rates while respecting groups
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)

    try:
        # This split will prioritize keeping airflow_bins balanced across the split
        dev_idx, holdout_idx = next(sgkf.split(X=df, y=airflow_bins, groups=groups))
    except ValueError as e:
        print(f"\nCRITICAL ERROR: Could not create a stratified split.")
        print(f"Scikit-learn error: {e}")
        return

    dev_df = df.iloc[dev_idx]
    holdout_df = df.iloc[holdout_idx]

    dev_df = dev_df.drop(columns=['base_video_id'])
    holdout_df = holdout_df.drop(columns=['base_video_id'])

    dev_df.to_csv(TRAIN_CSV_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_CSV_PATH, index=False)

    print(f"\nOriginal dataset size: {len(df)} samples")
    print(f"Development (train) set size: {len(dev_df)} samples")
    print(f"Hold-out set size: {len(holdout_df)} samples")
    print(f"\nSaved development features to: {TRAIN_CSV_PATH}")
    print(f"Saved hold-out features to: {HOLDOUT_CSV_PATH}")

    # --- Verification Step ---
    print("\n--- Verifying Split Distribution ---")

    # 1. The original raw count table
    dev_counts = dev_df['airflow_rate'].value_counts().sort_index()
    holdout_counts = holdout_df['airflow_rate'].value_counts().sort_index()
    
    verification_raw_df = pd.DataFrame({
        'dev_set_count': dev_counts,
        'holdout_set_count': holdout_counts
    }).fillna(0).astype(int)
    
    # Use .to_string() to ensure the full DataFrame is printed without truncation
    print("\nOverall distribution of samples per airflow rate (raw counts):")
    print(f"\n{verification_raw_df.to_string()}")

    # 2. Verify the distribution of the STRATIFICATION BINS
    print("\nDistribution of airflow_rate bins (proportions):")
    dev_bin_counts = airflow_bins.iloc[dev_idx].value_counts(normalize=True).sort_index()
    holdout_bin_counts = airflow_bins.iloc[holdout_idx].value_counts(normalize=True).sort_index()

    verification_bins_df = pd.DataFrame({
        'dev_set_proportion': dev_bin_counts,
        'holdout_set_proportion': holdout_bin_counts
    }).fillna(0)
    print(f"\n{verification_bins_df.to_string()}")

    # 3. Verify the distribution of MATERIALS
    print("\nDistribution of materials (proportions):")
    dev_material_counts = dev_df['material'].value_counts(normalize=True).sort_index()
    holdout_material_counts = holdout_df['material'].value_counts(normalize=True).sort_index()

    verification_material_df = pd.DataFrame({
        'dev_set_proportion': dev_material_counts,
        'holdout_set_proportion': holdout_material_counts
    }).fillna(0)
    print(f"\n{verification_material_df.to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a stratified and grouped train/hold-out split.")
    parser.add_argument('--random-state', type=int, default=42, help='The random state seed for the split.')
    args = parser.parse_args()
    main(args)

# python -m src_feature_based.split_data --random-state 42