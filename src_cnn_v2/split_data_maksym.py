# src_cnn_v2/split_data_maksym.py
"""
Performs a data split that EXACTLY replicates the colleague's methodology.
Logic: Group by Voltage -> Shuffle -> Split (60/20/20).
"""
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from math import floor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from src_cnn_v2 import config_v2 as cfg

def split_counts(n: int):
    """Colleague's exact logic for small sample sizes."""
    if n >= 5:
        # 60/20/20 logic
        quotas = [n * 0.6, n * 0.2, n * 0.2]
        floors = [floor(q) for q in quotas]
        remainder = n - sum(floors)
        # Add remainder to largest fractions
        fracs = sorted(((q - f, i) for i, (q, f) in enumerate(zip(quotas, floors))), reverse=True)
        for _, i in fracs[:remainder]: floors[i] += 1
        return floors
    if n in (3, 4): return [n - 2, 1, 1]
    if n == 2: return [1, 0, 1]
    if n == 1: return [1, 0, 0]
    return [0, 0, 0]

def main():
    print("--- Splitting Data (Colleague's Voltage Logic) ---")
    df_master = pd.read_csv(cfg.MASTER_METADATA_PATH)

    # 1. Group videos by Voltage
    video_to_voltage = df_master.groupby('video_id')['voltage'].first()
    voltage_to_videos = defaultdict(list)
    for vid, volt in video_to_voltage.items():
        voltage_to_videos[volt].append(vid)

    train_vids, val_vids, test_vids = [], [], []
    rng = random.Random(cfg.RANDOM_STATE)

    # 2. Split within each voltage group
    for volt, videos in sorted(voltage_to_videos.items()):
        rng.shuffle(videos)
        n_train, n_val, n_test = split_counts(len(videos))
        
        train_vids.extend(videos[:n_train])
        val_vids.extend(videos[n_train : n_train + n_val])
        test_vids.extend(videos[n_train + n_val :])
        
        print(f"Voltage {volt}: {len(videos)} videos -> {n_train} Train, {n_val} Val, {n_test} Test")

    # 3. Save
    train_df = df_master[df_master['video_id'].isin(train_vids)]
    val_df = df_master[df_master['video_id'].isin(val_vids)]
    test_df = df_master[df_master['video_id'].isin(test_vids)]

    train_df.to_csv(cfg.TRAIN_SPLIT_PATH, index=False)
    val_df.to_csv(cfg.VAL_SPLIT_PATH, index=False)
    test_df.to_csv(cfg.TEST_SPLIT_PATH, index=False)
    print(f"Saved splits. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    main()

# python src_cnn_v2/split_data_maksym.py