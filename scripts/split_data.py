# scripts/split_data.py
"""
Splits a master metadata/feature CSV into a development (train) and hold-out (test) set.

This single, robust script can handle multiple scenarios:
- CNN pipeline metadata or feature-based pipeline master feature files.
- Datasets with single or multiple material types.
- Correctly groups samples from the same original video (e.g., multi-hole videos).

It uses StratifiedGroupKFold to create the most balanced and leakage-free split possible.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os
import sys
import numpy as np
import argparse

# --- Add project root to path for imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import configs from both pipelines to get RANDOM_STATE
from src_cnn import config as cnn_cfg
from src_feature_based import config as feat_cfg

def main():
    parser = argparse.ArgumentParser(
        description="Create a stratified train/hold-out split for any dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the input metadata.csv or master_features.csv file to be split.")
    parser.add_argument("--holdout_size", type=float, default=0.2,
                        help="The approximate fraction of the data for the hold-out set.")
    parser.add_argument("--random_state", type=int, default=cnn_cfg.RANDOM_STATE,
                        help="Seed for the random number generator for reproducibility.")
    args = parser.parse_args()

    # Determine output file paths based on the input path
    input_dir = os.path.dirname(args.input_csv)
    TRAIN_CSV_PATH = os.path.join(input_dir, "train_metadata.csv")
    HOLDOUT_CSV_PATH = os.path.join(input_dir, "holdout_metadata.csv")
    
    # If the input is master_features.csv, name the outputs accordingly
    if "master_features.csv" in args.input_csv:
        TRAIN_CSV_PATH = os.path.join(input_dir, "train_features.csv")
        HOLDOUT_CSV_PATH = os.path.join(input_dir, "holdout_features.csv")

    print("--- Creating STRATIFIED & GROUPED Train/Hold-Out Split ---")
    
    if not os.path.exists(args.input_csv):
        print(f"FATAL ERROR: Input file not found at '{args.input_csv}'")
        sys.exit(1)
        
    df = pd.read_csv(args.input_csv)

    # --- Intelligent Stratification Strategy ---
    print("\nDetermining stratification strategy...")
    # Always stratify by airflow rate by creating discrete bins
    df['stratify_bins'] = pd.cut(df['airflow_rate'], bins=5, labels=False, duplicates='drop')
    
    # Check if a 'material' column exists and has more than one unique value
    if 'material' in df.columns and df['material'].nunique() > 1:
        print("Multiple materials detected. Stratifying by 'airflow_rate' AND 'material'.")
        df['stratify_group'] = df['stratify_bins'].astype(str) + '_' + df['material']
    else:
        print("Single material detected (or no material column). Stratifying by 'airflow_rate' only.")
        df['stratify_group'] = df['stratify_bins'].astype(str)
        
    # Create group labels from the original video ID to prevent data leakage
    df['original_video_id'] = df['video_id'].apply(lambda x: x.split('_hole_')[0])
    groups = df['original_video_id']
    y_stratify = df['stratify_group']
    
    n_splits = int(np.round(1 / args.holdout_size))
    if n_splits < 2:
        raise ValueError("holdout_size is too large to create a valid split.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)

    try:
        dev_idx, holdout_idx = next(sgkf.split(X=df, y=y_stratify, groups=groups))
    except ValueError as e:
        print(f"\nCRITICAL ERROR: Could not create a stratified split.")
        print(f"This can happen if a group ('video_id') contains samples from multiple stratification classes, or if a class has fewer members than n_splits ({n_splits}).")
        print(f"Scikit-learn error: {e}")
        return

    dev_df = df.iloc[dev_idx]
    holdout_df = df.iloc[holdout_idx]

    # Drop the temporary columns before saving
    dev_df = dev_df.drop(columns=['stratify_bins', 'stratify_group', 'original_video_id'])
    holdout_df = holdout_df.drop(columns=['stratify_bins', 'stratify_group', 'original_video_id'])

    dev_df.to_csv(TRAIN_CSV_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_CSV_PATH, index=False)

    print(f"\nOriginal dataset size: {len(df)} samples")
    print(f"Development set size: {len(dev_df)} samples")
    print(f"Hold-out set size: {len(holdout_df)} samples")
    print(f"\nSaved development data to: {TRAIN_CSV_PATH}")
    print(f"Saved hold-out data to: {HOLDOUT_CSV_PATH}")

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

"""
python -m scripts.split_data --input_csv CNN_dataset/dataset_2ch_thermal_masked_f10s/metadata.csv --random_state 43
python -m scripts.split_data --input_csv output_feature_based/master_features.csv

"""