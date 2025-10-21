# src_cnn_v2/split_data_v2.py
"""
Performs a leak-proof AND stratified 3-way split of the master metadata file
into training, validation, and test sets for the V2 pipeline.

It uses StratifiedGroupKFold to ensure that the distribution of the target
variable (airflow_rate) is similar across all splits, while also keeping all
samples from a single video recording in the same set.

This script also logs a detailed summary of the split to the experiment's log file.
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

# Import V2 Config and logging utility
from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.logging_utils_v2 import log_experiment_details

def main():
    parser = argparse.ArgumentParser(description="Split data into stratified, grouped train, validation, and test sets.")
    args = parser.parse_args()

    print("--- Splitting Data for V2 Pipeline (Stratified & Grouped) ---")

    try:
        df_master = pd.read_csv(cfg.MASTER_METADATA_PATH)
        print(f"Loaded {len(df_master)} total samples from {cfg.MASTER_METADATA_PATH}")
    except FileNotFoundError:
        sys.exit(f"FATAL: Master metadata file not found at '{cfg.MASTER_METADATA_PATH}'.")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # --- Prepare for Stratification ---
    df_master['airflow_bin'] = pd.qcut(df_master['airflow_rate'], q=5, labels=False, duplicates='drop')
    
    X = df_master
    y = df_master['airflow_bin']
    groups = df_master["video_id"]
    
    # --- Stage 1: Split off the Test Set ---
    print(f"\nStage 1: Separating test set (~20% of total data)...")
    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    dev_indices, test_indices = next(sgkf_test.split(X, y, groups))

    dev_df = df_master.iloc[dev_indices]
    test_df = df_master.iloc[test_indices]
    
    # --- Stage 2: Split the Development Set into Train and Validation ---
    print(f"\nStage 2: Splitting development set into train and validation (~80% / ~20%)...")
    
    dev_X = dev_df
    dev_y = dev_df['airflow_bin']
    dev_groups = dev_df["video_id"]
    
    sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    train_indices, val_indices = next(sgkf_val.split(dev_X, dev_y, dev_groups))

    train_df = dev_df.iloc[train_indices]
    val_df = dev_df.iloc[val_indices]
    
    # --- Final Sanity Check and Reporting ---
    print("\n--- Split Summary ---")
    print(f"  Total Samples: {len(df_master)}")
    print(f"  Unique Videos (Groups): {len(df_master['video_id'].unique())}")
    print("-" * 25)
    print(f"  Training Set:   {len(train_df)} samples ({len(train_df['video_id'].unique())} videos)")
    print(f"  Validation Set: {len(val_df)} samples ({len(val_df['video_id'].unique())} videos)")
    print(f"  Test Set:       {len(test_df)} samples ({len(test_df['video_id'].unique())} videos)")
    print("-" * 25)
    
    total_samples = len(train_df) + len(val_df) + len(test_df)
    assert total_samples == len(df_master), "Mismatch in sample counts after splitting!"

    # --- LOGGING THE SPLIT DETAILS ---
    log_filepath = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "experiment_summary.txt")
    os.makedirs(cfg.EXPERIMENT_RESULTS_DIR, exist_ok=True)

    # Convert describe() output to a dictionary for clean logging
    airflow_distribution_summary = df_master['airflow_rate'].describe().to_dict()

    split_summary = {
        "Splitting Strategy": "StratifiedGroupKFold",
        "Random Seed": cfg.RANDOM_STATE,
        "Number of Folds (for dev/test split)": 5,
        "Number of Folds (for train/val split)": 5,
        "Stratification Bins (qcut)": 5,
        "Original Dataset Stats": {
            "Total Samples": len(df_master),
            "Unique Videos (Groups)": len(groups.unique()),
            "Airflow Rate Distribution": airflow_distribution_summary
        },
        "Split Counts": {
            "Training Set": {
                "Samples": len(train_df),
                "Proportion": f"{len(train_df) / len(df_master):.2%}",
                "Unique Videos": len(train_df['video_id'].unique())
            },
            "Validation Set": {
                "Samples": len(val_df),
                "Proportion": f"{len(val_df) / len(df_master):.2%}",
                "Unique Videos": len(val_df['video_id'].unique())
            },
            "Test Set": {
                "Samples": len(test_df),
                "Proportion": f"{len(test_df) / len(df_master):.2%}",
                "Unique Videos": len(test_df['video_id'].unique())
            }
        }
    }
    
    log_experiment_details(log_filepath, "Data Splitting Details", split_summary)
    
    # Drop the temporary bin column before saving
    train_df = train_df.drop(columns=['airflow_bin'])
    val_df = val_df.drop(columns=['airflow_bin'])
    test_df = test_df.drop(columns=['airflow_bin'])
    
    # --- Save the splits --
    train_df.to_csv(cfg.TRAIN_SPLIT_PATH, index=False)
    val_df.to_csv(cfg.VAL_SPLIT_PATH, index=False)
    test_df.to_csv(cfg.TEST_SPLIT_PATH, index=False)

    print(f"\nSaved training split to: {cfg.TRAIN_SPLIT_PATH}")
    print(f"Saved validation split to: {cfg.VAL_SPLIT_PATH}")
    print(f"Saved test split to: {cfg.TEST_SPLIT_PATH}")
    print("\n--- Data Splitting Complete ---")

if __name__ == "__main__":
    main()

# python src_cnn_v2/split_data_v2.py