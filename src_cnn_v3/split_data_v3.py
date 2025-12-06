# src_cnn_v3/split_data_v3.py
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v3 import config_v3 as cfg
from src_cnn_v3.logging_utils_v3 import log_section

def main():
    print("--- V3: Splitting Data (Stratified & Grouped) ---")

    if not os.path.exists(cfg.MASTER_METADATA_PATH):
        sys.exit(f"FATAL: Metadata not found at {cfg.MASTER_METADATA_PATH}")

    df_master = pd.read_csv(cfg.MASTER_METADATA_PATH)
    print(f"Loaded {len(df_master)} samples.")

    # 1. Create Stratification Bins
    df_master['flow_bin'] = pd.qcut(df_master['airflow_rate'], q=5, labels=False, duplicates='drop')
    
    X = df_master
    y = df_master['flow_bin']
    groups = df_master['video_id'] 

    # 2. Split Test Set (20%)
    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    dev_idx, test_idx = next(sgkf_test.split(X, y, groups))
    
    df_dev = df_master.iloc[dev_idx]
    df_test = df_master.iloc[test_idx]

    # 3. Split Dev into Train (80%) / Val (20%)
    sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    
    dev_X = df_dev
    dev_y = df_dev['flow_bin']
    dev_groups = df_dev['video_id']
    
    train_idx, val_idx = next(sgkf_val.split(dev_X, dev_y, dev_groups))
    
    df_train = df_dev.iloc[train_idx]
    df_val = df_dev.iloc[val_idx]

    # 4. Save
    for df in [df_train, df_val, df_test]:
        df.drop(columns=['flow_bin'], inplace=True, errors='ignore')

    df_train.to_csv(cfg.TRAIN_SPLIT_PATH, index=False)
    df_val.to_csv(cfg.VAL_SPLIT_PATH, index=False)
    df_test.to_csv(cfg.TEST_SPLIT_PATH, index=False)

    log_data = {
        "Random Seed": cfg.RANDOM_STATE,
        "Total Samples": len(df_master),
        "Train Set": f"{len(df_train)} samples ({len(df_train['video_id'].unique())} videos)",
        "Val Set": f"{len(df_val)} samples ({len(df_val['video_id'].unique())} videos)",
        "Test Set": f"{len(df_test)} samples ({len(df_test['video_id'].unique())} videos)",
        "Stratification Strategy": "StratifiedGroupKFold on Airflow Bins"
    }
    log_section("Data Splitting", log_data)

    print("\n--- Split Summary ---")
    print(f"  Train: {len(df_train)} samples")
    print(f"  Val:   {len(df_val)} samples")
    print(f"  Test:  {len(df_test)} samples")
    print(f"  Saved to: {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    main()