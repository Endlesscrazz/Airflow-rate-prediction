# src_cnn_v2/create_metadata_v2.py
"""
Creates a lightweight master metadata file for the V2 pipeline.

This script reads the ground truth CSV and, for each sample, verifies that
the required raw .mat video and .npy mask files exist. It does NOT calculate
any handcrafted features, making it much faster than generate_master_features.py.

The output is a 'master_metadata_v2.csv' file containing only the essential
columns needed for the V2 data preparation workflow.
"""
import os
import sys
import pandas as pd
from tqdm import tqdm
import glob

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import V2 Config
from src_cnn_v2 import config_v2 as cfg

def main():
    print("--- Creating Lightweight Master Metadata for V2 Pipeline ---")

    # Define the output path using the V2 config
    METADATA_SAVE_PATH = os.path.join(cfg.OUTPUT_DIR, "master_metadata_v2.csv")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    try:
        df_ground_truth = pd.read_csv(cfg.GROUND_TRUTH_CSV_PATH)
        print(f"Loaded {len(df_ground_truth)} total samples from {cfg.GROUND_TRUTH_CSV_PATH}")
    except FileNotFoundError:
        sys.exit(f"FATAL: Ground truth CSV not found at '{cfg.GROUND_TRUTH_CSV_PATH}'.")

    valid_samples = []
    
    for index, row in tqdm(df_ground_truth.iterrows(), total=len(df_ground_truth), desc="Verifying samples"):
        try:
            video_id = row['video_id']
            hole_id = str(row['hole_id'])
            
            # --- Verify existence of raw files (using the robust logic) ---
            mat_filepath, mask_dir_path, found_config_key = (None, None, None)
            for d_key, d_conf in cfg.DATASET_CONFIGS.items():
                video_search_pattern = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"], '**', f"{video_id}.mat")
                video_results = glob.glob(video_search_pattern, recursive=True)
                if video_results:
                    mat_filepath, found_config_key = video_results[0], d_key
                    break
            
            if not mat_filepath:
                print(f"\nWarning: Skipping sample '{video_id}_{hole_id}'. Reason: .mat file not found.")
                continue

            mask_subfolder = cfg.DATASET_CONFIGS[found_config_key]["dataset_subfolder"]
            mask_search_pattern = os.path.join(cfg.RAW_MASK_PARENT_DIR, mask_subfolder, '**', video_id)
            mask_dir_results = glob.glob(mask_search_pattern, recursive=True)
            if mask_dir_results:
                for path in mask_dir_results:
                    if os.path.isdir(path): mask_dir_path = path; break
            
            if not mask_dir_path:
                print(f"\nWarning: Skipping sample '{video_id}_{hole_id}'. Reason: Mask directory not found.")
                continue

            # --- ROBUST MASK FILE FINDING LOGIC ---
            # This logic handles multiple naming conventions.
            
            # Path 1: Exact match (e.g., hole_id '2' -> _mask_2.npy) - for multi-hole
            path1 = os.path.join(mask_dir_path, f"{video_id}_mask_{hole_id}.npy")
            
            # Path 2: 0-indexed for hole_id '1' (e.g., hole_id '1' -> _mask_0.npy) - for single-hole
            path2 = os.path.join(mask_dir_path, f"{video_id}_mask_0.npy")

            individual_mask_path = None
            if os.path.exists(path1):
                individual_mask_path = path1
            elif hole_id == '1' and os.path.exists(path2):
                individual_mask_path = path2
            
            if not individual_mask_path:
                # If neither path was found, print a detailed warning and skip.
                print(f"\nWarning: Skipping sample '{video_id}_{hole_id}'. Reason: Mask file not found.")
                print(f"  - Tried path 1: {path1}")
                if hole_id == '1': print(f"  - Tried path 2: {path2}")
                continue
            # --- END OF ROBUST LOGIC ---

            # If all files exist, add the sample to our list
            sample_data = row.to_dict()
            sample_data['sample_id'] = f"{video_id}_{hole_id}"
            valid_samples.append(sample_data)

        except Exception as e:
            print(f"\nError processing row {index}: {e}")
            continue

    # Create the final DataFrame
    df_meta = pd.DataFrame(valid_samples)

    if df_meta.empty:
        print("\nFATAL: No valid samples found. Check paths in config_v2.py and file locations.")
        return
        
    df_meta.to_csv(METADATA_SAVE_PATH, index=False)
    
    print(f"\n--- Master Metadata Creation Complete ---")
    print(f"Verified {len(df_meta)} valid samples.")
    print(f"Saved lightweight metadata to: {METADATA_SAVE_PATH}")

if __name__ == "__main__":
    main()

# python src_cnn_v2/create_metadata_v2.py