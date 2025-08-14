# scripts/generate_features.py
"""
Scans raw video and mask data, calculates a comprehensive set of handcrafted
features for each identified leak, links them to ground truth values from a CSV,
and saves the output to a single master feature file.
"""
import os
import sys
import pandas as pd
from tqdm import tqdm
import fnmatch
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_feature_based import config as cfg
from src_feature_based import data_utils, feature_engineering

def main():
    print("--- Generating Master Handcrafted Feature CSV ---")

    # 1. Load the ground truth airflow data from the CSV
    airflow_map = data_utils.load_airflow_from_csv(cfg.GROUND_TRUTH_CSV_PATH)
    print(f"Successfully loaded {len(airflow_map)} ground truth entries.")

    # 2. Scan all raw video and mask files
    print("\nScanning for all video and mask files...")
    video_to_masks_map = {}
    for d_key, d_conf in cfg.DATASET_CONFIGS.items():
        dataset_path_load = os.path.join(cfg.RAW_DATA_ROOT, d_conf["dataset_subfolder"])
        
        if not os.path.isdir(dataset_path_load):
            print(f"  -> WARNING: Data path not found for '{d_key}'. Skipping.")
            continue
        
        for root_load, _, files_load in os.walk(dataset_path_load):
            for mat_filename_load in fnmatch.filter(files_load, '*.mat'):
                voltage = data_utils.parse_voltage_from_filename(mat_filename_load)
                if voltage is None or voltage not in airflow_map:
                    continue
                
                mat_filepath_load = os.path.join(root_load, mat_filename_load)
                video_id = os.path.splitext(mat_filename_load)[0]
                
                if video_id not in video_to_masks_map:
                    try:
                        airflow_rate = airflow_map[voltage]
                        delta_t_load = data_utils.parse_delta_T(mat_filename_load)
                        if delta_t_load is None: continue

                        video_to_masks_map[video_id] = {
                            "video_id": video_id, "mat_filepath": mat_filepath_load, "mask_paths": [],
                            "delta_T": float(delta_t_load), "airflow_rate": float(airflow_rate),
                            "material": d_conf["material"], "source_dataset": d_conf["dataset_subfolder"]
                        }
                    except Exception:
                        continue
                
                mask_search_dir = os.path.join(cfg.RAW_MASK_PARENT_DIR, d_conf["dataset_subfolder"], video_id)

                if os.path.isdir(mask_search_dir):
                    mask_files = fnmatch.filter(os.listdir(mask_search_dir), "*.npy")
                    if mask_files and video_id in video_to_masks_map:
                        for mask_filename in mask_files:
                            full_mask_path = os.path.join(mask_search_dir, mask_filename)
                            if full_mask_path not in video_to_masks_map[video_id]["mask_paths"]:
                                video_to_masks_map[video_id]["mask_paths"].append(full_mask_path)

    all_samples_info_list = [v for v in video_to_masks_map.values() if v.get("mask_paths")]
    if not all_samples_info_list:
        print("\nError: No samples found. Exiting.")
        return

    # 3. Extract features for each sample
    print(f"\nFound {len(all_samples_info_list)} videos to process...")
    all_features_list = []
    
    for sample_info in tqdm(all_samples_info_list, desc="Extracting features"):
        try:

            extracted_features = feature_engineering.calculate_features_from_video(
                sample_info["mat_filepath"],
                sample_info["mask_paths"]
            )
            
            record = {
                "video_id": sample_info["video_id"],
                "airflow_rate": sample_info["airflow_rate"],
                "delta_T": sample_info["delta_T"],
                "material": sample_info["material"],
            }
            record.update(extracted_features)
            all_features_list.append(record)

        except Exception as e:
            print(f"Failed to process {sample_info['video_id']}: {e}")
            continue

    # 4. Create and save the DataFrame
    df_master = pd.DataFrame(all_features_list)
    # Ensure all possible feature columns exist, filling missing ones with NaN
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