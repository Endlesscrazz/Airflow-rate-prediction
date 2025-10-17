# scripts/generate_master_features.py
"""
Generates a master CSV file containing ALL possible handcrafted features for
every sample defined in the master ground truth CSV. This is the single source
of truth for all feature-based analysis.
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy.io
from tqdm import tqdm
import glob

# --- Add project root to path for imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Import Your Project Modules ---
from src_cnn import config as cfg
from src_cnn import feature_engineering_cnn


def main():
    print("--- Generating Master Feature CSV ---")
    print(f"This will extract all {len(cfg.ALL_FEATURES_TO_CALCULATE)} possible features for every sample.")
    os.makedirs(os.path.dirname(cfg.MASTER_FEATURES_PATH), exist_ok=True)

    try:
        ground_truth_df = pd.read_csv(cfg.GROUND_TRUTH_CSV_PATH)
        print(f"\nLoaded {len(ground_truth_df)} total samples from {cfg.GROUND_TRUTH_CSV_PATH}")
    except Exception as e:
        sys.exit(f"FATAL: Could not load master ground truth CSV '{cfg.GROUND_TRUTH_CSV_PATH}'. Error: {e}")

    all_features_list = []
    
    for index, sample_row in tqdm(ground_truth_df.iterrows(), total=len(ground_truth_df), desc="Extracting features"):
        try:
            video_id = sample_row['video_id']
            # Ensure hole_id is a string for consistent path joining
            hole_id = str(sample_row['hole_id']) 
            
            mat_filepath = None
            mask_dir_path = None
            found_config_key = None

            # --- ROBUST FILE FINDING LOGIC (HARMONIZED) ---
            # 1. Find the correct dataset config and the .mat file recursively
            for d_key, d_conf in cfg.DATASET_CONFIGS.items():
                video_search_pattern = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"], '**', f"{video_id}.mat")
                video_results = glob.glob(video_search_pattern, recursive=True)
                if video_results:
                    mat_filepath = video_results[0]
                    found_config_key = d_key
                    break 
            
            if not mat_filepath:
                raise FileNotFoundError(f"Could not find .mat file for video_id '{video_id}' in any configured directory.")
            
            # 2. Use the found config key to search for the mask directory recursively
            mask_subfolder = cfg.DATASET_CONFIGS[found_config_key]["dataset_subfolder"]
            mask_search_pattern = os.path.join(cfg.RAW_MASK_PARENT_DIR, mask_subfolder, '**', video_id)
            mask_dir_results = glob.glob(mask_search_pattern, recursive=True)
            
            if mask_dir_results:
                for path in mask_dir_results:
                    if os.path.isdir(path):
                        mask_dir_path = path
                        break

            if not mask_dir_path:
                raise FileNotFoundError(f"Could not find a mask DIRECTORY for video_id '{video_id}'")

            # 3. Construct the final mask path and check for existence
            individual_mask_path = os.path.join(mask_dir_path, f"{video_id}_mask_{hole_id}.npy")
            
            if not os.path.exists(individual_mask_path):
                raise FileNotFoundError(f"Mask file does not exist at expected path: {individual_mask_path}")

            # --- END OF FILE FINDING LOGIC ---

            individual_mask = np.load(individual_mask_path).astype(bool)
            if not np.any(individual_mask): 
                # Skip empty masks to avoid calculation errors
                continue

            frames = scipy.io.loadmat(mat_filepath).get('TempFrames').astype(np.float64)

            extracted_features = feature_engineering_cnn.calculate_hotspot_features(
                frames=frames, hotspot_mask=individual_mask, envir_para=1
            )
            
            record = sample_row.to_dict()
            record['sample_id'] = f"{video_id}_{hole_id}"
            record.update(extracted_features)
            all_features_list.append(record)

        except Exception as e:
            print(f"\nWarning: Failed to process sample for video '{sample_row.get('video_id', 'N/A')}', hole '{sample_row.get('hole_id', 'N/A')}'. Reason: {e}")
            continue

    df_master = pd.DataFrame(all_features_list)
    df_master.to_csv(cfg.MASTER_FEATURES_PATH, index=False)
    
    print(f"\n--- Master Feature Generation Complete ---")
    print(f"Successfully saved {len(df_master)} samples with {len(df_master.columns)} columns to:")
    print(cfg.MASTER_FEATURES_PATH)

if __name__ == "__main__":
    main()
# python scripts/generate_master_features.py