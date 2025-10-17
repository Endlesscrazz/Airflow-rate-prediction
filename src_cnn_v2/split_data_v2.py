# src_cnn_v2/split_data_v2.py
"""
Performs a leak-proof AND stratified 3-way split of the master feature file
into training, validation, and test sets for the V2 pipeline.

It uses StratifiedGroupKFold to ensure that the distribution of the target
variable (airflow_rate) is similar across all splits, while also keeping all
samples from a single video recording in the same set.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import argparse

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import V2 Config
from src_cnn_v2 import config_v2 as cfg

def main():
    parser = argparse.ArgumentParser(description="Split data into stratified, grouped train, validation, and test sets.")
    # We will use a 5-fold split to get an 80/20 dev/test split, then another 5-fold on dev for train/val
    args = parser.parse_args()

    print("--- Splitting Data for V2 Pipeline (Stratified & Grouped) ---")

    try:
        df_master = pd.read_csv(cfg.MASTER_METADATA_PATH)
        print(f"Loaded {len(df_master)} total samples from {cfg.MASTER_METADATA_PATH}")
    except FileNotFoundError:
        sys.exit(f"FATAL: Master metadata file not found at '{cfg.MASTER_METADATA_PATH}'.")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # --- Prepare for Stratification ---
    # Stratification works best on discrete bins. We'll bin the target variable.
    # Create 5 bins based on the quantiles of airflow_rate.
    df_master['airflow_bin'] = pd.qcut(df_master['airflow_rate'], q=5, labels=False, duplicates='drop')
    
    X = df_master
    y = df_master['airflow_bin'] # Stratify on the bins
    groups = df_master["video_id"]
    
    # --- Stage 1: Split off the Test Set using one fold of StratifiedGroupKFold ---
    print(f"\nStage 1: Separating test set (~20% of total data)...")
    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    dev_indices, test_indices = next(sgkf_test.split(X, y, groups))

    dev_df = df_master.iloc[dev_indices]
    test_df = df_master.iloc[test_indices]
    
    print(f"  - Development set size: {len(dev_df)} samples ({len(dev_df['video_id'].unique())} unique videos)")
    print(f"  - Test set size: {len(test_df)} samples ({len(test_df['video_id'].unique())} unique videos)")

    # --- Stage 2: Split the Development Set into Train and Validation ---
    print(f"\nStage 2: Splitting development set into train and validation (~80% / ~20%)...")
    
    dev_X = dev_df
    dev_y = dev_df['airflow_bin']
    dev_groups = dev_df["video_id"]
    
    # We use n_splits=5 on the dev set, which is 80% of the data.
    # This results in a 4/5 vs 1/5 split, which is 64% train and 16% validation of the original total.
    sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    train_indices, val_indices = next(sgkf_val.split(dev_X, dev_y, dev_groups))

    train_df = dev_df.iloc[train_indices]
    val_df = dev_df.iloc[val_indices]

    print(f"  - Final training set size: {len(train_df)} samples ({len(train_df['video_id'].unique())} unique videos)")
    print(f"  - Final validation set size: {len(val_df)} samples ({len(val_df['video_id'].unique())} unique videos)")

    # --- Final Sanity Check ---
    total_samples = len(train_df) + len(val_df) + len(test_df)
    print(f"\nTotal samples accounted for: {total_samples} / {len(df_master)}")
    assert total_samples == len(df_master), "Mismatch in sample counts after splitting!"
    
    # Drop the temporary bin column before saving
    train_df = train_df.drop(columns=['airflow_bin'])
    val_df = val_df.drop(columns=['airflow_bin'])
    test_df = test_df.drop(columns=['airflow_bin'])
    
    # --- Save the splits ---
    train_split_path = os.path.join(cfg.OUTPUT_DIR, "train_split.csv")
    val_split_path = os.path.join(cfg.OUTPUT_DIR, "val_split.csv")
    test_split_path = os.path.join(cfg.OUTPUT_DIR, "test_split.csv")

    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)
    test_df.to_csv(test_split_path, index=False)

    print(f"\nSaved training split to: {train_split_path}")
    print(f"Saved validation split to: {val_split_path}")
    print(f"Saved test split to: {test_split_path}")
    print("\n--- Data Splitting Complete ---")

if __name__ == "__main__":
    main()

# python src_cnn_v2/split_data_v2.py