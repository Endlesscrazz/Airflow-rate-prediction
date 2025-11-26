# src_cnn_v2/create_exact_test_split.py
"""
Creates a custom train/test split that EXACTLY matches the colleague's test set
AND ensures NO DATA LEAKAGE between Train and Validation (Grouped Split).
"""
import os
import pandas as pd
import sys
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg

# The exact list of 21 files from the colleague's output
COLLEAGUE_TEST_FILES = [
    ("T1.4V_2025-08-15-18-58-06_21_28_7_", "1"),
    ("T1.4V_2025-08-15-18-58-06_21_28_7_", "2"),
    ("temp_2025-3-13-17-14-30_21.4_27.5_6.1_", "1"),
    ("T1.6V_2025-08-14-14-56-37_21_34_13_", "1"),
    ("T1.6V_2025-08-14-14-56-37_21_34_13_", "2"),
    ("temp_2025-3-13-18-17-30_21.4_26_4.6_", "1"),
    ("temp_2025-3-13-17-50-56_21.4_32.5_11.1_", "1"),
    ("T2.0V_2025-08-15-17-32-13_21_32_11_", "1"),
    ("T2.0V_2025-08-15-17-32-13_21_32_11_", "2"),
    ("temp_2025-3-13-19-19-40_21.4_27.5_6.1_", "1"),
    ("T2.2V_2025-08-13-18-12-42_21_26_5_", "1"),
    ("T2.2V_2025-08-13-18-12-42_21_26_5_", "2"),
    ("temp_2025-3-13-19-45-15_21.4_35_13.6_", "1"),
    ("T2.4V_2025-08-13-17-33-28_21_26_5_", "1"),
    ("T2.4V_2025-08-13-17-33-28_21_26_5_", "2"),
    ("T2.6V_2025-08-13-16-41-58_21_26_5_", "1"),
    ("T2.6V_2025-08-13-16-41-58_21_26_5_", "2"),
    ("T2.8V_2025-08-13-15-24-34_21_34_13_", "1"),
    ("T2.8V_2025-08-13-15-24-34_21_34_13_", "2"),
    ("T3.0V_2025-08-13-15-03-01_21_26_5_", "1"),
    ("T3.0V_2025-08-13-15-03-01_21_26_5_", "2"),
]

def main():
    print("--- Creating LEAK-FREE Train/Val/Test Split ---")
    
    # 1. Load Master Metadata
    try:
        df_master = pd.read_csv(cfg.MASTER_METADATA_PATH)
        print(f"Loaded {len(df_master)} total samples.")
    except FileNotFoundError:
        sys.exit("Master metadata not found.")

    # 2. Identify Test Rows (Exact Match)
    test_indices = []
    for target_vid, target_hole in COLLEAGUE_TEST_FILES:
        match = df_master[
            (df_master['video_id'] == target_vid) & 
            (df_master['hole_id'].astype(str) == str(target_hole))
        ]
        if len(match) == 1:
            test_indices.append(match.index[0])
    
    test_df = df_master.loc[test_indices].copy()
    train_val_pool = df_master.drop(test_indices).copy()

    # 3. Perform Grouped Split for Train/Val
    # We must split by VIDEO_ID, not by row.
    unique_videos = train_val_pool['video_id'].unique()
    np.random.seed(42) # Fixed seed for reproducibility
    np.random.shuffle(unique_videos)
    
    # 80/20 Split on VIDEOS
    split_idx = int(len(unique_videos) * 0.8)
    train_videos = unique_videos[:split_idx]
    val_videos = unique_videos[split_idx:]
    
    train_df = train_val_pool[train_val_pool['video_id'].isin(train_videos)].copy()
    val_df = train_val_pool[train_val_pool['video_id'].isin(val_videos)].copy()

    print("\n--- Split Summary ---")
    print(f"Test Set:      {len(test_df)} samples")
    print(f"Training Set:  {len(train_df)} samples ({len(train_videos)} unique videos)")
    print(f"Validation Set:{len(val_df)} samples ({len(val_videos)} unique videos)")
    
    # Verify no video leakage
    train_vids = set(train_df['video_id'])
    val_vids = set(val_df['video_id'])
    overlap = train_vids.intersection(val_vids)
    if overlap:
        print(f"FATAL ERROR: Data Leakage Detected! Videos in both sets: {overlap}")
        sys.exit(1)
    else:
        print("Success: No video leakage between Train and Validation.")

    # 4. Save
    test_df.to_csv(cfg.TEST_SPLIT_PATH, index=False)
    val_df.to_csv(cfg.VAL_SPLIT_PATH, index=False)
    train_df.to_csv(cfg.TRAIN_SPLIT_PATH, index=False)
    print("Saved split files.")

if __name__ == "__main__":
    main()

# python -m src_cnn_v2.create_exact_test_split