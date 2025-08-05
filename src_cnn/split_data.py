# src_cnn/split_data.py
"""
Splits the main metadata.csv into a development (train) and hold-out (test) set.

This version uses StratifiedGroupKFold to create a balanced split based on
airflow rate and material, ensuring the hold-out set is a representative sample
of the full dataset. It specifically handles rare classes to prevent them from
being disproportionately placed in the hold-out set.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os
import numpy as np

# --- Configuration ---
CNN_DATASET_DIR = "cnn_dataset/dataset_cnn-lstm-all-split-holes"
METADATA_PATH = os.path.join(CNN_DATASET_DIR, "metadata.csv")
TRAIN_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "train_metadata.csv")
HOLDOUT_METADATA_PATH = os.path.join(CNN_DATASET_DIR, "holdout_metadata.csv")
HOLDOUT_SIZE = 0.2
RANDOM_STATE = 44 # A fixed seed for reproducibility

def main():
    print("--- Creating STRATIFIED Train/Hold-Out Split with Verification ---")
    df = pd.read_csv(METADATA_PATH)

    # --- Stratification Strategy ---
    # We want to stratify by a combination of the most important features to ensure balance.
    # The most critical is 'airflow_rate', followed by 'material'.
    # We create a new column that combines these for stratification.
    df['stratify_group'] = df['airflow_rate'].astype(str) + '_' + df['material']

    # We need to split based on 'video_id' groups.
    groups = df['video_id']
    y = df['stratify_group']

    # StratifiedGroupKFold requires n_splits. To get a single train/test split,
    # we'll set n_splits to the approximate inverse of HOLDOUT_SIZE.
    # e.g., for 0.2 holdout size, we do a 5-fold split and take one fold as the holdout.
    n_splits = int(np.round(1 / HOLDOUT_SIZE))
    if n_splits < 2:
        raise ValueError("HOLDOUT_SIZE is too large to create a split.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # The splitter will generate n_splits sets of indices. We'll just use the first one.
    try:
        dev_idx, holdout_idx = next(sgkf.split(X=df, y=y, groups=groups))
    except ValueError as e:
        print(f"\n\033[91mCRITICAL ERROR:\033[0m Could not create a stratified split.")
        print(f"This often happens if a group ('video_id') contains multiple stratification classes, or if a class has fewer members than n_splits ({n_splits}).")
        print(f"Scikit-learn error: {e}")
        print("\nSuggestions:")
        print("1. Try changing the RANDOM_STATE.")
        print("2. Check your data for videos that might be mislabeled.")
        print("3. Consider increasing HOLDOUT_SIZE (e.g., to 0.25 for n_splits=4) if you have classes with only 3 or 4 members.")
        return


    dev_df = df.iloc[dev_idx]
    holdout_df = df.iloc[holdout_idx]

    # Drop the temporary stratification column before saving
    dev_df = dev_df.drop(columns=['stratify_group'])
    holdout_df = holdout_df.drop(columns=['stratify_group'])

    # Save the files
    dev_df.to_csv(TRAIN_METADATA_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_METADATA_PATH, index=False)

    print(f"\nOriginal dataset size: {len(df)}")
    print(f"Development (train) set size: {len(dev_df)}")
    print(f"Hold-out set size: {len(holdout_df)}")
    print(f"Saved development metadata to: {TRAIN_METADATA_PATH}")
    print(f"Saved hold-out metadata to: {HOLDOUT_METADATA_PATH}")

    # --- Verification Step ---
    print("\n--- Verifying Split Distribution ---")
    
    # Check distribution of the stratification group itself
    dev_strat_counts = df.iloc[dev_idx]['stratify_group'].value_counts().sort_index()
    holdout_strat_counts = df.iloc[holdout_idx]['stratify_group'].value_counts().sort_index()

    verification_strat_df = pd.DataFrame({
        'dev_set_count': dev_strat_counts,
        'holdout_set_count': holdout_strat_counts
    }).fillna(0).astype(int)

    print("\nDistribution of samples per stratification group (airflow_material):")
    print(verification_strat_df)

    # Also check the overall airflow rate distribution
    dev_counts = dev_df['airflow_rate'].value_counts().sort_index()
    holdout_counts = holdout_df['airflow_rate'].value_counts().sort_index()
    
    verification_df = pd.DataFrame({
        'dev_set_count': dev_counts,
        'holdout_set_count': holdout_counts
    }).fillna(0).astype(int)
    
    print("\nOverall distribution of samples per airflow rate:")
    print(verification_df)

    # Check for any airflow rates that are ONLY in the holdout set (this should not happen now)
    unique_dev = set(dev_df['airflow_rate'].unique())
    unique_holdout = set(holdout_df['airflow_rate'].unique())
    
    missing_in_dev = unique_holdout - unique_dev
    if missing_in_dev:
        print(f"\n\033[91mCRITICAL WARNING:\033[0m The following airflow rates are in the hold-out set but NOT in the development set:")
        print(sorted(list(missing_in_dev)))
    else:
        print("\n\033[92mSUCCESS:\033[0m All airflow rates in the hold-out set are represented in the development set.")


if __name__ == "__main__":
    main()

# python -m src_cnn.split_data