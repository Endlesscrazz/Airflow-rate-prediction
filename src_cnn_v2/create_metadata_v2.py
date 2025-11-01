# src_cnn_v2/create_metadata_v2.py
"""
Creates a lightweight master metadata file for the V2 pipeline.

This script reads the ground truth CSV and, for each sample, verifies that
the required raw .mat video and the corresponding _coordinates.json file exist.

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

    METADATA_SAVE_PATH = cfg.MASTER_METADATA_PATH
    os.makedirs(os.path.dirname(METADATA_SAVE_PATH), exist_ok=True)

    try:
        df_ground_truth = pd.read_csv(cfg.GROUND_TRUTH_CSV_PATH)
        print(f"Loaded {len(df_ground_truth)} total samples from {cfg.GROUND_TRUTH_CSV_PATH}")
    except FileNotFoundError:
        sys.exit(f"FATAL: Ground truth CSV not found at '{cfg.GROUND_TRUTH_CSV_PATH}'. Please run 'create_ground_truth_labels.py' first.")

    valid_samples = []
    
    for index, row in tqdm(df_ground_truth.iterrows(), total=len(df_ground_truth), desc="Verifying sample files"):
        try:
            video_id = row['video_id']
            hole_id = str(row['hole_id'])

            # --- 1. Verify existence of the raw .mat video file ---
            mat_filepath, found_config_key = (None, None)
            for d_key, d_conf in cfg.DATASET_CONFIGS.items():
                # Construct search pattern within the specific dataset subfolder
                video_search_pattern = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"], '**', f"{video_id}.mat")
                video_results = glob.glob(video_search_pattern, recursive=True)
                if video_results:
                    mat_filepath, found_config_key = video_results[0], d_key
                    break
            
            if not mat_filepath:
                tqdm.write(f"\nWarning: Skipping sample '{video_id}_{hole_id}'. Reason: .mat file not found.")
                continue

            # --- 2. Verify existence of the coordinates.json file ---
            coord_subfolder = cfg.DATASET_CONFIGS[found_config_key]["dataset_subfolder"]
            # Search for the video-specific subfolder created by find_leaking_holes.py
            coord_search_pattern = os.path.join(cfg.RAW_MASK_PARENT_DIR, coord_subfolder, '**', video_id)
            coord_dir_results = glob.glob(coord_search_pattern, recursive=True)
            coord_dir_path = next((path for path in coord_dir_results if os.path.isdir(path)), None)

            if not coord_dir_path:
                tqdm.write(f"\nWarning: Skipping sample '{video_id}_{hole_id}'. Reason: Coordinate directory not found.")
                continue

            coord_path = os.path.join(coord_dir_path, f"{video_id}_coordinates.json")
            if not os.path.exists(coord_path):
                tqdm.write(f"\nWarning: Skipping sample '{video_id}_{hole_id}'. Reason: Coordinates file not found.")
                tqdm.write(f"  - Expected file at: {coord_path}")
                continue

            # If all files exist, add the sample to our list
            sample_data = row.to_dict()
            sample_data['sample_id'] = f"{video_id}_{hole_id}"
            valid_samples.append(sample_data)

        except Exception as e:
            print(f"\nError processing row {index}: {e}")
            continue

    if not valid_samples:
        print("\nFATAL: No valid samples found. Check paths in config_v2.py and ensure 'find_leaking_holes.py' has been run successfully.")
        return
        
    df_meta = pd.DataFrame(valid_samples)
    df_meta.to_csv(METADATA_SAVE_PATH, index=False)
    
    print(f"\n--- Master Metadata Creation Complete ---")
    print(f"Verified {len(df_meta)} valid samples.")
    print(f"Saved lightweight metadata to: {METADATA_SAVE_PATH}")

if __name__ == "__main__":
    main()

# python src_cnn_v2/create_metadata_v2.py