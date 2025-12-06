# src_cnn_v3/create_metadata_v3.py
import os
import sys
import pandas as pd
import json
import glob
from tqdm import tqdm

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v3 import config_v3 as cfg

def validate_geometry(hole_id, features):
    """
    Quality Assurance: Returns False if the detected shape is physically impossible.
    """
    ar = features['aspect_ratio']
    area = features['area_px']
    
    # 1. Minimum Size Check (Noise filtering)
    if area < 20: 
        return False
        
    # 2. Slit Checks (Hole 1 and 10)
    # These should be long and thin. If AR is close to 1.0, it's just a blob/fragment.
    if hole_id in [1, 10]:
        if ar < 1.8: # Threshold: Length must be at least 1.8x width
            return False
            
    # 3. Circle Checks (Holes 2-9)
    # These should be roughly square/circular. If AR is huge, it's a streak artifact.
    # (Optional, usually less critical)
    # if hole_id in [2,3,4,5,6,7,8,9]:
    #     if ar > 3.0: return False
        
    return True

def main():
    print("--- V3: Creating Master Metadata (With QA) ---")
    
    if not os.path.exists(cfg.GROUND_TRUTH_CSV_PATH):
        sys.exit(f"FATAL: Ground truth CSV not found: {cfg.GROUND_TRUTH_CSV_PATH}")
        
    df_gt = pd.read_csv(cfg.GROUND_TRUTH_CSV_PATH)
    print(f"Loaded {len(df_gt)} ground truth records.")
    
    valid_samples = []
    skipped_counts = {"missing_file": 0, "sam_miss": 0, "qa_fail": 0}
    
    grouped = df_gt.groupby('video_id')
    
    for video_id, group in tqdm(grouped, desc="Merging Features"):
        source_key = group.iloc[0]['source_dataset_key']
        if source_key not in cfg.DATASET_CONFIGS: continue
            
        d_conf = cfg.DATASET_CONFIGS[source_key]
        subfolder = d_conf['dataset_subfolder']
        
        search_pattern = os.path.join(cfg.INTERMEDIATE_DATA_DIR, subfolder, "**", f"{video_id}", f"{video_id}_features.json")
        found_files = glob.glob(search_pattern, recursive=True)
        
        if not found_files:
            skipped_counts["missing_file"] += len(group)
            continue
            
        json_path = found_files[0]
        
        try:
            with open(json_path, 'r') as f:
                features_list = json.load(f)
        except Exception:
            skipped_counts["missing_file"] += len(group)
            continue
            
        feat_map = {item['hole_id']: item for item in features_list}
        
        for _, row in group.iterrows():
            hole_id = int(row['hole_id'])
            
            if hole_id in feat_map:
                feat = feat_map[hole_id]
                
                # --- APPLY QA CHECK ---
                if not validate_geometry(hole_id, feat):
                    skipped_counts["qa_fail"] += 1
                    # Optional: Print info about rejected sample
                    print(f"Rejecting {video_id} Hole {hole_id}: AR={feat['aspect_ratio']:.2f}")
                    continue
                
                merged_record = row.to_dict()
                merged_record.update({
                    'obb_center_x': feat['center_x'],
                    'obb_center_y': feat['center_y'],
                    'obb_width': feat['obb_width'],
                    'obb_height': feat['obb_height'],
                    'obb_angle': feat['obb_angle'],
                    'feat_area': feat['area_px'],
                    'feat_aspect': feat['aspect_ratio'],
                    'feat_extent': feat['extent'],
                    'mask_path': feat['mask_path']
                })
                valid_samples.append(merged_record)
            else:
                skipped_counts["sam_miss"] += 1

    if valid_samples:
        df_master = pd.DataFrame(valid_samples)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        df_master.to_csv(cfg.MASTER_METADATA_PATH, index=False)
        
        print("\n" + "="*40)
        print(f"SUCCESS: Master Metadata created.")
        print(f"  - Output: {cfg.MASTER_METADATA_PATH}")
        print(f"  - Valid Samples: {len(df_master)}")
        print("  - Skipped Samples:")
        print(f"    * Missing JSON File: {skipped_counts['missing_file']}")
        print(f"    * Not Detected by SAM: {skipped_counts['sam_miss']}")
        print(f"    * Failed QA (Bad Geometry): {skipped_counts['qa_fail']}")
        print("="*40)
        
        if 'feat_area' in df_master.columns:
            corr = df_master['feat_area'].corr(df_master['airflow_rate'])
            print(f"  [Sanity Check] Correlation (Area vs Airflow): {corr:.4f}")
    else:
        print("\nFAILURE: No valid samples found.")

if __name__ == "__main__":
    main()