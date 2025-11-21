# src_cnn_v2/split_data_maksym.py
"""
Performs a data split that EXACTLY replicates the colleague's methodology,
including the specific logic for small sample sizes.

The logic is as follows:
1. Group all unique videos by their voltage.
2. For each voltage group, shuffle the list of videos.
3. Use the colleague's `split_counts(n)` function to determine how many videos
   from that group go to train, val, and test.
4. Combine the video lists from all voltage groups to get the final splits.
5. Use these video lists to create the final sample DataFrames.
"""
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from math import floor

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import V2 Config and logging utility
from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.logging_utils_v2 import log_experiment_details

# --- REPLICATION of Colleague's `split_counts` logic ---
def apportion_counts(n: int, ratios=(0.60, 0.20, 0.20)):
    """Standard 60/20/20 split for n >= 5."""
    quotas = [n * r for r in ratios]
    floors = [floor(q) for q in quotas]
    remainder = n - sum(floors)
    fracs = sorted(
        ((q - f, i) for i, (q, f) in enumerate(zip(quotas, floors))), reverse=True
    )
    for _, i in fracs[:remainder]:
        floors[i] += 1
    return floors

def split_counts(n: int):
    """
    This function is a direct copy of the colleague's logic from their
    `prepare_data.py` script to handle small numbers of videos.
    """
    if n >= 5:
        return apportion_counts(n)
    if n in (3, 4):
        return [n - 2, 1, 1]
    if n == 2:
        return [1, 0, 1]
    if n == 1:
        return [1, 0, 0]
    return [0, 0, 0]

def main():
    print("--- Splitting Data using Colleague's EXACT Voltage-Based Methodology ---")

    try:
        df_master = pd.read_csv(cfg.MASTER_METADATA_PATH)
        print(f"Loaded {len(df_master)} total samples from {cfg.MASTER_METADATA_PATH}")
    except FileNotFoundError:
        sys.exit(f"FATAL: Master metadata file not found at '{cfg.MASTER_METADATA_PATH}'.")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # --- 1. Group unique videos by voltage ---
    video_to_voltage = df_master.groupby('video_id')['voltage'].first()
    voltage_to_videos = defaultdict(list)
    for video_id, voltage in video_to_voltage.items():
        voltage_to_videos[voltage].append(video_id)

    # --- 2. Split videos within each voltage group using the replicated logic ---
    train_video_ids, val_video_ids, test_video_ids = [], [], []
    
    rng = random.Random(cfg.RANDOM_STATE)

    print("\nSplitting videos within each voltage group using exact `split_counts` logic...")
    voltage_split_details = {}
    for voltage, videos in sorted(voltage_to_videos.items()):
        rng.shuffle(videos)
        
        n = len(videos)
        n_train, n_val, n_test = split_counts(n)
        
        train_video_ids.extend(videos[:n_train])
        val_video_ids.extend(videos[n_train : n_train + n_val])
        test_video_ids.extend(videos[n_train + n_val :])
        
        split_info = f"Train={n_train}, Val={n_val}, Test={n_test}"
        print(f"  - Voltage {voltage:.2f}V ({n} videos): {split_info}")
        voltage_split_details[f"{voltage:.2f}V"] = f"({n} videos) -> {split_info}"

    # --- 3. Create the final DataFrames from the video ID lists ---
    train_df = df_master[df_master['video_id'].isin(train_video_ids)]
    val_df = df_master[df_master['video_id'].isin(val_video_ids)]
    test_df = df_master[df_master['video_id'].isin(test_video_ids)]

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
    
    # --- LOGGING THE SPLIT DETAILS (Copied and adapted from split_data_v2.py) ---
    log_filepath = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "experiment_summary.txt")
    os.makedirs(cfg.EXPERIMENT_RESULTS_DIR, exist_ok=True)

    airflow_distribution_summary = df_master['airflow_rate'].describe().to_dict()

    split_summary = {
        "Splitting Strategy": "Colleague's Voltage-Based Grouping",
        "Random Seed": cfg.RANDOM_STATE,
        "Per-Voltage Split Logic": "Replicated from colleague's `split_counts` function",
        "Per-Voltage Split Details": voltage_split_details,
        "Original Dataset Stats": {
            "Total Samples": len(df_master),
            "Unique Videos (Groups)": len(df_master['video_id'].unique()),
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

# python src_cnn_v2/split-data_maksym.py