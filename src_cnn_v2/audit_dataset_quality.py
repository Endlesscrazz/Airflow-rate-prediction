# src_cnn_v2/audit_dataset_quality.py
"""
Dataset Quality Audit Tool.

Scans the generated .npy files to statistically detect bad data.
It checks for:
1. Low Signal (Variance): Is the image flat/empty?
2. Off-Center (Center of Mass): Is the leak actually in the middle?
3. Saturation: Is the image purely 1.0 or 0.0?

How to Run:
  python src_cnn_v2/audit_dataset_quality.py
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy.ndimage
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg

# --- THRESHOLDS FOR "BAD" DATA ---
# Flag if standard deviation is lower than this (flat image)
MIN_STD_DEV = 0.01 
# Flag if center of mass is more than this many pixels away from center (7.5, 7.5)
MAX_CENTER_OFFSET = 4.0 

def analyze_single_sample(row, dataset_dir):
    """Calculates quality metrics for a single sample."""
    try:
        filepath = os.path.join(dataset_dir, row['image_path'])
        data = np.load(filepath) # Shape: (Time, H, W) or (H, W, T) depending on stage
        
        # Ensure standard shape (Time, H, W)
        if data.shape[1] != cfg.CROP_SIZE: 
             data = data.transpose(2, 0, 1)

        # 1. Calculate Time-Averaged Image
        avg_img = np.mean(data, axis=0) # Shape (H, W)

        # 2. Metric: Signal Strength (Standard Deviation)
        std_dev = np.std(avg_img)

        # 3. Metric: Center of Mass (Centering)
        # Subtract min to focus on the "hot" part relative to background
        weights = avg_img - np.min(avg_img)
        if np.sum(weights) == 0:
            cy, cx = 0, 0
        else:
            cy, cx = scipy.ndimage.center_of_mass(weights)
        
        # Distance from geometric center (7.5, 7.5 for 15x15)
        center = cfg.CROP_SIZE / 2.0
        offset = np.sqrt((cy - center)**2 + (cx - center)**2)

        return {
            'sample_id': row['sample_id'],
            'std_dev': std_dev,
            'center_offset': offset,
            'max_val': np.max(data),
            'min_val': np.min(data),
            'status': 'OK'
        }
    except Exception as e:
        return {'sample_id': row['sample_id'], 'status': 'ERROR', 'error_msg': str(e)}

def main():
    print(f"--- Starting Dataset Quality Audit ---")
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    print(f"Dataset Dir: {cfg.DATASET_DIR}")

    # 1. Load Metadata
    train_df = pd.read_csv(cfg.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(cfg.VAL_METADATA_PATH)
    test_df = pd.read_csv(cfg.TEST_METADATA_PATH)
    
    # Combine all splits
    full_df = pd.concat([train_df, val_df, test_df])
    
    # 2. Filter: Analyze ALL originals, but only 5% of augmentations
    originals = full_df[~full_df['sample_id'].str.contains('_aug_')]
    augmented = full_df[full_df['sample_id'].str.contains('_aug_')]
    
    # Sample 5% of augmented to save time
    augmented_sample = augmented.sample(frac=0.05, random_state=42)
    
    audit_df = pd.concat([originals, augmented_sample])
    print(f"Auditing {len(audit_df)} files ({len(originals)} originals, {len(augmented_sample)} augmented samples)...")

    # 3. Run Parallel Analysis
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(analyze_single_sample)(row, cfg.DATASET_DIR) 
        for _, row in audit_df.iterrows()
    )
    
    results_df = pd.DataFrame(results)

    # 4. Flag "Suspicious" Files
    low_signal = results_df[results_df['std_dev'] < MIN_STD_DEV]
    off_center = results_df[results_df['center_offset'] > MAX_CENTER_OFFSET]
    errors = results_df[results_df['status'] == 'ERROR']

    print("\n" + "="*30)
    print("       AUDIT RESULTS       ")
    print("="*30)
    print(f"Total Files Checked: {len(results_df)}")
    print("-" * 20)
    print(f"Files with Low Signal (StdDev < {MIN_STD_DEV}): {len(low_signal)}")
    print(f"Files Off-Center (Offset > {MAX_CENTER_OFFSET}px):   {len(off_center)}")
    print(f"Files with Read Errors:                   {len(errors)}")
    
    if len(low_signal) > 0:
        print("\n[!] WARNING: Found files with almost no thermal signal (Flat/Empty).")
        print("    Top 5 Low Signal Samples:")
        print(low_signal[['sample_id', 'std_dev']].head(5).to_string(index=False))

    if len(off_center) > 0:
        print("\n[!] WARNING: Found files where the leak is far from the center.")
        print("    Top 5 Off-Center Samples:")
        print(off_center[['sample_id', 'center_offset']].head(5).to_string(index=False))

    # 5. Save Report
    report_path = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "data_quality_audit.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\nDetailed audit report saved to: {report_path}")

if __name__ == "__main__":
    main()

# python -m src_cnn_v2.audit_dataset_quality