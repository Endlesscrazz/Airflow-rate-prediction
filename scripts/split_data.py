# scripts/split_data.py
"""
Splits a main metadata.csv into a development (train) and hold-out (test) set.
Uses StratifiedGroupKFold to create a balanced split.
All configurations are imported from src_cnn.config.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os
import sys
import numpy as np
import argparse

# --- Import project modules ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn import config as cfg

def main():
    parser = argparse.ArgumentParser(description="Create a stratified train/hold-out split for a dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the processed dataset directory (e.g., 'CNN_dataset/dataset_1ch_thermal').")
    parser.add_argument("--holdout_size", type=float, default=0.2,
                        help="The approximate fraction of the data to use for the hold-out set.")
    args = parser.parse_args()

    METADATA_PATH = os.path.join(args.dataset_dir, "metadata.csv")
    TRAIN_METADATA_PATH = os.path.join(args.dataset_dir, "train_metadata.csv")
    HOLDOUT_METADATA_PATH = os.path.join(args.dataset_dir, "holdout_metadata.csv")

    print("--- Creating STRATIFIED Train/Hold-Out Split ---")
    
    if not os.path.exists(METADATA_PATH):
        print(f"FATAL ERROR: Metadata file not found at {METADATA_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(METADATA_PATH)

    # Create a combined column for stratification to balance by airflow and material.
    df['stratify_group'] = df['airflow_rate'].astype(str) + '_' + df['material']
    groups = df['video_id']
    y = df['stratify_group']

    n_splits = int(np.round(1 / args.holdout_size))
    if n_splits < 2:
        raise ValueError("holdout_size is too large to create a valid split.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_STATE)

    try:
        dev_idx, holdout_idx = next(sgkf.split(X=df, y=y, groups=groups))
    except ValueError as e:
        print(f"\nCRITICAL ERROR: Could not create a stratified split.")
        print(f"This often happens if a group ('video_id') contains samples from multiple stratification classes, or if a class has fewer members than n_splits ({n_splits}).")
        print(f"Scikit-learn error: {e}")
        return

    dev_df = df.iloc[dev_idx].drop(columns=['stratify_group'])
    holdout_df = df.iloc[holdout_idx].drop(columns=['stratify_group'])

    dev_df.to_csv(TRAIN_METADATA_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_METADATA_PATH, index=False)

    print(f"\nOriginal dataset size: {len(df)}")
    print(f"Development (train) set size: {len(dev_df)}")
    print(f"Hold-out set size: {len(holdout_df)}")
    print(f"Saved development metadata to: {TRAIN_METADATA_PATH}")
    print(f"Saved hold-out metadata to: {HOLDOUT_METADATA_PATH}")

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

    unique_dev = set(dev_df['airflow_rate'].unique())
    unique_holdout = set(holdout_df['airflow_rate'].unique())
    missing_in_dev = unique_holdout - unique_dev
    
    if missing_in_dev:
        print(f"\nWARNING: The following airflow rates are in the hold-out set but NOT in the development set:")
        print(sorted(list(missing_in_dev)))
    else:
        print("\nSUCCESS: All airflow rates in the hold-out set are represented in the development set.")

if __name__ == "__main__":
    main()

"""
python -m scripts.split_data --dataset_dir "CNN_dataset/dataset_2ch_thermal_masked_f10s"
"""