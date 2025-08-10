# scripts/generate_master_features.py
"""
A one-time utility script to generate a master CSV file containing ALL possible
handcrafted features for every sample in the raw dataset.

This version correctly handles two-hole videos by treating each hole as an
independent sample, mirroring the logic in the deep learning data creation pipeline.
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy.io
from tqdm import tqdm
import fnmatch

# --- Add project root to path for imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Import Your Project Modules ---
from src_cnn import config as cfg, feature_engineering
from archive.old_cnn_scripts import data_utils

# --- Configuration for this script ---
# We will override the config to ensure ALL features are calculated
ALL_POSSIBLE_FEATURES = [
    'hotspot_area', 'hotspot_avg_temp_change_rate_initial', 'hotspot_avg_temp_change_magnitude_initial',
    'peak_pixel_temp_change_rate_initial', 'peak_pixel_temp_change_magnitude_initial', 'temp_mean_avg_initial',
    'temp_std_avg_initial', 'temp_min_overall_initial', 'temp_max_overall_initial',
    'stabilized_mean_deltaT', 'overall_mean_deltaT', 'max_abs_mean_deltaT',
    'stabilized_std_deltaT', 'overall_std_deltaT', 'mean_area_significant_change',
    'stabilized_area_significant_change', 'max_area_significant_change',
    'num_hotspots', 'hotspot_solidity', 'centroid_distance', 'time_to_peak_mean_temp',
    'temperature_skewness', 'temperature_kurtosis', 'rate_of_std_change_initial', 'peak_to_average_ratio',
    'radial_profile_0', 'radial_profile_1', 'radial_profile_2', 'radial_profile_3', 'radial_profile_4',
    'bbox_area', 'bbox_aspect_ratio'
]

# Temporarily override the config to ensure ALL features are calculated for this script
cfg.HANDCRAFTED_FEATURES_TO_EXTRACT = ALL_POSSIBLE_FEATURES
OUTPUT_CSV_PATH = os.path.join(project_root, "master_features.csv")


def main():
    print("--- Generating Master Feature CSV ---")
    print(f"This will extract all {len(ALL_POSSIBLE_FEATURES)} possible features for every sample.")

    # Step 1: Scan all raw video and mask files (logic from create_dataset.py)
    print("\nScanning for all video and mask files...")
    # ... (This file scanning logic is correct and does not need to change) ...
    video_to_masks_map = {}
    for d_key, d_conf in cfg.DATASET_CONFIGS.items():
        dataset_path_load = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"])
        mask_root_path_load = os.path.join(cfg.RAW_MASK_PARENT_DIR, d_conf["mask_subfolder"])
        if not os.path.isdir(dataset_path_load) or not os.path.isdir(mask_root_path_load): continue
        for root_load, _, files_load in os.walk(dataset_path_load):
            for mat_filename_load in fnmatch.filter(files_load, '*.mat'):
                mat_filepath_load = os.path.join(root_load, mat_filename_load)
                video_id = os.path.splitext(mat_filename_load)[0]
                if video_id not in video_to_masks_map:
                    try:
                        folder_name_load = os.path.basename(os.path.dirname(mat_filepath_load))
                        airflow_load = data_utils.parse_airflow_rate(folder_name_load)
                        delta_t_load = data_utils.parse_delta_T(mat_filename_load)
                        if delta_t_load is None: continue
                        video_to_masks_map[video_id] = {
                            "video_id": video_id, "mat_filepath": mat_filepath_load, "mask_paths": [],
                            "delta_T": float(delta_t_load), "airflow_rate": float(airflow_load),
                            "material": d_conf["material"], "source_dataset": d_conf["dataset_subfolder"]
                        }
                    except Exception: continue
                mat_basename = os.path.splitext(mat_filename_load)[0]
                relative_path_part = os.path.relpath(root_load, dataset_path_load)
                mask_search_dir = os.path.join(mask_root_path_load, relative_path_part, mat_basename)
                if os.path.isdir(mask_search_dir):
                    mask_files = fnmatch.filter(os.listdir(mask_search_dir), f"{mat_basename}_mask_*.npy")
                    mask_files.extend(fnmatch.filter(os.listdir(mask_search_dir), f"{mat_basename}_sam_mask.npy"))
                    if mask_files and video_id in video_to_masks_map:
                        for mask_filename in mask_files:
                            full_mask_path = os.path.join(mask_search_dir, mask_filename)
                            if full_mask_path not in video_to_masks_map[video_id]["mask_paths"]:
                                video_to_masks_map[video_id]["mask_paths"].append(full_mask_path)
    
    all_samples_info_list = [v_data for v_data in video_to_masks_map.values() if v_data.get("mask_paths")]
    if not all_samples_info_list:
        print("\nError: No samples found. Exiting.")
        return

    # Step 2: Extract features for each sample, correctly handling two-hole videos
    print(f"\nFound {len(all_samples_info_list)} videos to process...")
    all_features_list = []
    
    for sample_info in tqdm(all_samples_info_list, desc="Extracting features"):
        try:
            frames = scipy.io.loadmat(sample_info["mat_filepath"]).get('TempFrames').astype(np.float64)
            is_two_hole_video = "two_holes" in sample_info["source_dataset"]
            
            masks_to_process = []
            if is_two_hole_video:
                # For two-hole videos, each mask is a separate sample.
                for mask_path in sample_info['mask_paths']:
                    if mask_path.endswith('.npy'):
                        masks_to_process.append((np.load(mask_path).astype(bool), mask_path))
            else:
                # For single-hole videos, combine all found masks into one.
                combined_mask = np.zeros(frames.shape[:2], dtype=bool)
                for mask_path in sample_info['mask_paths']:
                    combined_mask = np.logical_or(combined_mask, np.load(mask_path))
                if np.any(combined_mask):
                    masks_to_process.append((combined_mask, "combined"))

            # Loop through the one (or more) masks identified for this video
            for i, (individual_mask, mask_identifier) in enumerate(masks_to_process):
                if not np.any(individual_mask):
                    continue

                # Create a unique ID for each new sample
                original_video_id = sample_info['video_id']
                new_video_id = f"{original_video_id}_hole_{i}" if is_two_hole_video else original_video_id

                # Calculate features using the individual mask
                extracted_features = feature_engineering.calculate_hotspot_features(
                    frames=frames,
                    hotspot_mask=individual_mask,
                    envir_para=-1
                )
                
                # Combine with essential metadata
                record = {
                    "video_id": new_video_id, # Use the new, unique ID
                    "airflow_rate": sample_info["airflow_rate"],
                    "delta_T": sample_info["delta_T"],
                    "material": sample_info["material"],
                    "source_dataset": sample_info["source_dataset"]
                }
                record.update(extracted_features)
                all_features_list.append(record)

        except Exception as e:
            print(f"Failed to process {sample_info['video_id']}: {e}")
            continue

    # Step 3: Create and save the DataFrame
    df_master = pd.DataFrame(all_features_list)
    df_master.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"\n--- Master Feature Generation Complete ---")
    print(f"Successfully saved {len(df_master)} samples with {len(df_master.columns)} columns to:")
    print(OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()