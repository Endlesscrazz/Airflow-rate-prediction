# scripts/split_data.py
"""
Splits a master metadata/feature CSV into a development (train) and hold-out (test) set.
Uses StratifiedGroupKFold to create the most balanced and leakage-free split possible.
Includes a detailed verification step to check the distribution of key variables.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os
import sys
import numpy as np
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(
        description="Create a stratified train/hold-out split for any dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the input metadata.csv or master_features.csv file to be split.")
    parser.add_argument("--holdout_size", type=float, default=0.2,
                        help="The approximate fraction of the data for the hold-out set.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Seed for the random number generator for reproducibility.")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_csv)
    TRAIN_CSV_PATH = os.path.join(input_dir, "train_features.csv")
    HOLDOUT_CSV_PATH = os.path.join(input_dir, "holdout_features.csv")
    if "metadata.csv" in args.input_csv:
        TRAIN_CSV_PATH = os.path.join(input_dir, "train_metadata.csv")
        HOLDOUT_CSV_PATH = os.path.join(input_dir, "holdout_metadata.csv")

    print("--- Creating STRATIFIED & GROUPED Train/Hold-Out Split ---")
    
    if not os.path.exists(args.input_csv):
        print(f"FATAL ERROR: Input file not found at '{args.input_csv}'")
        sys.exit(1)
        
    df = pd.read_csv(args.input_csv)

    print("\nDetermining stratification strategy...")
    df['stratify_bins'] = pd.cut(df['airflow_rate'], bins=5, labels=False, duplicates='drop')
    
    if 'material' in df.columns and df['material'].nunique() > 1:
        print("Multiple materials detected. Stratifying by 'airflow_rate' AND 'material'.")
        df['stratify_group'] = df['stratify_bins'].astype(str) + '_' + df['material']
    else:
        print("Single material detected. Stratifying by 'airflow_rate' only.")
        df['stratify_group'] = df['stratify_bins'].astype(str)
        
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

    dev_df = dev_df.drop(columns=['stratify_bins', 'stratify_group', 'original_video_id'])
    holdout_df = holdout_df.drop(columns=['stratify_bins', 'stratify_group', 'original_video_id'])

    dev_df.to_csv(TRAIN_CSV_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_CSV_PATH, index=False)

    print(f"\nOriginal dataset size: {len(df)} samples ({df['original_video_id'].nunique()} unique videos)")
    print(f"Development set size: {len(dev_df)} samples ({dev_df['video_id'].apply(lambda x: x.split('_hole_')[0]).nunique()} unique videos)")
    print(f"Hold-out set size: {len(holdout_df)} samples ({holdout_df['video_id'].apply(lambda x: x.split('_hole_')[0]).nunique()} unique videos)")
    print(f"\nSaved development data to: {TRAIN_CSV_PATH}")
    print(f"Saved hold-out data to: {HOLDOUT_CSV_PATH}")

    # --- NEW & IMPROVED Verification Step ---
    print("\n--- Verifying Split Distribution ---")

    # 1. Distribution by Material
    if 'material' in df.columns and df['material'].nunique() > 1:
        dev_material = dev_df['material'].value_counts(normalize=True)
        holdout_material = holdout_df['material'].value_counts(normalize=True)
        verification_material = pd.DataFrame({
            'dev_set_%': dev_material,
            'holdout_set_%': holdout_material
        }).fillna(0)
        print("\nDistribution of samples per Material (%):")
        print((verification_material * 100).round(1))

    # 2. Distribution by Airflow Bins (with FIXED bin edges)
    # --- THIS IS THE KEY CHANGE ---
    # First, create the bins based on the ENTIRE dataset to get fixed edges
    _, bin_edges = pd.cut(df['airflow_rate'], bins=5, retbins=True)
    
    # Now, use these exact same bin edges to categorize the subsets
    dev_bins = pd.cut(dev_df['airflow_rate'], bins=bin_edges, include_lowest=True)
    holdout_bins = pd.cut(holdout_df['airflow_rate'], bins=bin_edges, include_lowest=True)
    
    dev_counts = dev_bins.value_counts().sort_index()
    holdout_counts = holdout_bins.value_counts().sort_index()
    
    verification_bins = pd.DataFrame({
        'dev_set_count': dev_counts,
        'holdout_set_count': holdout_counts
    }).fillna(0).astype(int)
    
    print("\nDistribution of samples per Airflow Rate Bin:")
    print(verification_bins)


if __name__ == "__main__":
    main()
"""
python -m scripts.split_data --input_csv CNN_dataset/dataset_2ch_thermal_masked_gyp_hdb/metadata.csv --random_state 41
python -m scripts.split_data --input_csv output_feature_based/master_features.csv --random_state 44

"""