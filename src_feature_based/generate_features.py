# scripts/generate_features.py
"""
Generates handcrafted features for each sample defined in the master ground truth CSV.
This script implements the "one hole, one sample" philosophy by iterating through
the combined ground truth file.
"""
import os
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_feature_based import config as cfg
from src_feature_based import feature_engineering

def find_mask_for_hole(video_id, hole_id, mask_parent_dir, dataset_subfolder):
    """
    Finds the specific mask file(s) for a given hole of a given video.
    """
    mask_search_dir = os.path.join(mask_parent_dir, dataset_subfolder, video_id)
    if not os.path.isdir(mask_search_dir):
        return []

    all_masks = sorted([f for f in os.listdir(mask_search_dir) if f.endswith('.npy')])

    if "2holes" in dataset_subfolder:
        if ('centerhole' in hole_id or 'largehole' in hole_id) and len(all_masks) > 0:
            return [os.path.join(mask_search_dir, all_masks[0])]
        elif ('cornerhole' in hole_id or 'smallhole' in hole_id) and len(all_masks) > 1:
            return [os.path.join(mask_search_dir, all_masks[1])]
        else:
            return []
    else:
        return [os.path.join(mask_search_dir, f) for f in all_masks]

def main():
    print("--- Generating Master Handcrafted Feature CSV ---")

    if not os.path.exists(cfg.GROUND_TRUTH_CSV_PATH):
        print(f"FATAL ERROR: Combined ground truth not found at '{cfg.GROUND_TRUTH_CSV_PATH}'")
        print("Please run `create_combined_ground_truth.py` first.")
        sys.exit(1)
    
    gt_df = pd.read_csv(cfg.GROUND_TRUTH_CSV_PATH)
    print(f"Loaded {len(gt_df)} total samples to process from the ground truth file.")

    all_features_list = []
    
    pbar = tqdm(gt_df.iterrows(), total=len(gt_df), desc="Extracting features")
    for idx, sample_row in pbar:
        video_id = sample_row['video_id']
        hole_id = sample_row['hole_id']
        material = sample_row['material']
        
        dataset_subfolder = None
        # Find the correct subfolder from the config that matches the material
        for conf_key, conf_details in cfg.DATASET_CONFIGS.items():
            if conf_details['material'] == material:
                # Check if the video file actually exists in this specific folder
                mat_path_check = os.path.join(cfg.RAW_DATA_ROOT, conf_details['dataset_subfolder'], f"{video_id}.mat")
                if os.path.exists(mat_path_check):
                    dataset_subfolder = conf_details['dataset_subfolder']
                    break # Found it, stop searching
        
        if not dataset_subfolder:
            # New, more informative warning message
            tqdm.write(f"  - WARNING: Could not find video file for ID '{video_id}' (Material: '{material}'). "
                       f"Checked all configured '{material}' directories. Skipping sample.")
            continue
            
        mat_filepath = os.path.join(cfg.RAW_DATA_ROOT, dataset_subfolder, f"{video_id}.mat")

        mask_paths = find_mask_for_hole(video_id, hole_id, cfg.RAW_MASK_PARENT_DIR, dataset_subfolder)
        
        if not mask_paths:
            tqdm.write(f"  - WARNING: No masks found for sample: {video_id} | {hole_id}. Skipping.")
            continue
            
        try:
            extracted_features = feature_engineering.calculate_features_from_video(
                mat_filepath,
                mask_paths
            )
            
            record = sample_row.to_dict()
            record.update(extracted_features)
            all_features_list.append(record)

        except Exception as e:
            print(f"  - ERROR: Failed to process {sample_row['video_id']}: {e}")
            continue

    if not all_features_list:
        print("\n No features were generated. Check paths and file names.")
        return

    df_master = pd.DataFrame(all_features_list)
    
    for feature in cfg.ALL_POSSIBLE_FEATURES:
        if feature not in df_master.columns:
            df_master[feature] = np.nan
            
    output_path = os.path.join(cfg.OUTPUT_DIR, "master_features.csv")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    df_master.to_csv(output_path, index=False)
    
    print(f"\n--- Master Feature Generation Complete ---")
    print(f"Successfully saved {len(df_master)} samples with {len(df_master.columns)} columns to:")
    print(output_path)

if __name__ == "__main__":
    main()

# python -m src_feature_based.generate_features