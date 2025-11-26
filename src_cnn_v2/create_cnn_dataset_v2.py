# src_cnn_v2/create_cnn_dataset_v2.py
"""
Prepares a V2 dataset for the bottom-up CNN approach.
MATCHES COLLEAGUE PREPROCESSING EXACTLY:
- Reads pre-generated coordinate .json files.
- Frame Selection: CONSECUTIVE (0, 1, 2...) not Linspace.
- Padding: Replicates last frame if video is too short.
- Data Content: RAW TEMPERATURE VALUES (No normalization on disk).
- Augmentation: Adds Gaussian Noise to RAW data.
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy.io
from tqdm import tqdm
import glob
import random
import traceback
import json
import argparse

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.augmentation_utils import add_gaussian_noise

def crop_sequence(frames, center_x, center_y, crop_size):
    """Crops a fixed size patch around the center coordinates."""
    H, W, T = frames.shape
    half_crop = crop_size // 2
    
    # Calculate bounds ensuring we stay within image
    x_start = max(0, center_x - half_crop)
    x_end = min(W, x_start + crop_size)
    
    # Adjust if hitting right edge
    if x_end - x_start < crop_size:
        x_start = max(0, W - crop_size)
        x_end = W
    
    y_start = max(0, center_y - half_crop)
    y_end = min(H, y_start + crop_size)
    
    # Adjust if hitting bottom edge
    if y_end - y_start < crop_size:
        y_start = max(0, H - crop_size)
        y_end = H
        
    return frames[y_start:y_end, x_start:x_end, :]

def process_split(df_split, is_training_set, output_dir, debug=False):
    all_metadata_rows = []
    counts = {'created': 0, 'skipped': 0}
    
    iterator = df_split.iterrows()
    if not debug:
        iterator = tqdm(iterator, total=len(df_split), desc="Processing")

    for index, master_row in iterator:
        try:
            video_id = master_row['video_id']
            hole_id = str(master_row['hole_id'])
            sample_id = master_row['sample_id']
            numeric_hole_id = hole_id.split('_')[0]

            original_sample_id_v2 = f"{sample_id}_orig"
            original_filename = f"{original_sample_id_v2}.npy"
            original_filepath = os.path.join(output_dir, original_filename)

            cropped_sequence = None

            # --- 1. GENERATE ORIGINAL SEQUENCE ---
            if os.path.exists(original_filepath):
                counts['skipped'] += 1
            else:
                counts['created'] += 1
                
                # A. Find Raw Video File
                mat_filepath = None
                for d_key, d_conf in cfg.DATASET_CONFIGS.items():
                    video_search_pattern = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"], '**', f"{video_id}.mat")
                    video_results = glob.glob(video_search_pattern, recursive=True)
                    if video_results:
                        mat_filepath = video_results[0]
                        break
                
                if not mat_filepath:
                    if debug: print(f"Skipping {video_id}: .mat file not found")
                    continue
                
                # B. Find Coordinates File
                coord_path = None
                for d_key, d_conf in cfg.DATASET_CONFIGS.items():
                    coord_search_pattern = os.path.join(cfg.RAW_MASK_PARENT_DIR, d_conf["dataset_subfolder"], '**', video_id)
                    coord_dir_results = glob.glob(coord_search_pattern, recursive=True)
                    coord_dir_path = next((path for path in coord_dir_results if os.path.isdir(path)), None)
                    if coord_dir_path:
                        possible_path = os.path.join(coord_dir_path, f"{video_id}_coordinates.json")
                        if os.path.exists(possible_path):
                            coord_path = possible_path
                            break
                
                if not coord_path:
                    if debug: print(f"Skipping {video_id}: Coordinates not found")
                    continue
                
                with open(coord_path, 'r') as f:
                    all_leaks_in_video = json.load(f)
                
                target_leak_data = next((leak for leak in all_leaks_in_video if str(leak['hole_id']) == numeric_hole_id), None)
                if target_leak_data is None:
                    if debug: print(f"Skipping {video_id}: Hole {numeric_hole_id} not in json")
                    continue
                
                center_x = target_leak_data['center_x']
                center_y = target_leak_data['center_y']
                
                # C. Load Raw Frames
                # Use float32 to save space but keep precision
                frames = scipy.io.loadmat(mat_filepath).get('TempFrames').astype(np.float32)

                # --- FIX: DO NOT NORMALIZE HERE ---
                # The colleague saves RAW temperature data (e.g., 23.0).
                # Normalization happens in the DataLoader.
                
                # 1. Just Transpose to (Height, Width, Time) if not already
                # Mat files are usually (H, W, T)
                
                # D. Consecutive Frame Selection
                num_frames_needed = cfg.NUM_FRAMES_PER_SAMPLE
                
                if frames.shape[2] >= num_frames_needed:
                    # Take first N frames consecutively
                    selected_frames = frames[:, :, :num_frames_needed]
                else:
                    # Pad by repeating the last frame
                    padding_needed = num_frames_needed - frames.shape[2]
                    last_frame = frames[:, :, -1:] 
                    padding_block = np.repeat(last_frame, padding_needed, axis=2)
                    selected_frames = np.concatenate([frames, padding_block], axis=2)

                # E. Crop RAW Data
                cropped_frames = crop_sequence(selected_frames, center_x, center_y, cfg.V2_DATASET_PARAMS["CROP_SIZE"])
                
                # Transpose to (Time, Height, Width) for PyTorch convention
                # Colleague saves as (H, W, T), but our loader expects (T, H, W) or handles it.
                # To be consistent with *our* loader (dataset_utils_v2.py), we save as (T, H, W).
                # Note: Colleague saves (H, W, T), but his loader transposes it.
                cropped_sequence = cropped_frames.transpose(2, 0, 1)

                np.save(original_filepath, cropped_sequence)

            # Base metadata record
            base_record = {
                'original_sample_id': sample_id, 
                'video_id': video_id, 
                'hole_id': hole_id,
                'airflow_rate': master_row['airflow_rate'], 
                'delta_T': master_row['delta_T']
            }

            # --- 2. AUGMENTATION (TRAINING ONLY) ---
            if is_training_set:
                # Load the raw sequence if we didn't just create it
                if cropped_sequence is None:
                    cropped_sequence = np.load(original_filepath)

                aug_params = cfg.V2_DATASET_PARAMS.get("AUGMENTATION_PARAMS", {})
                
                for i in range(cfg.V2_DATASET_PARAMS["NUM_AUGMENTATIONS"]):
                    aug_filename = f"{sample_id}_aug_{i+1}.npy"
                    aug_filepath = os.path.join(output_dir, aug_filename)
                    
                    if not os.path.exists(aug_filepath):
                        # Add Gaussian Noise to RAW data
                        augmented_sequence = add_gaussian_noise(
                            cropped_sequence.copy(), 
                            noise_level=aug_params.get("NOISE_LEVEL", 0.105)
                        )
                        np.save(aug_filepath, augmented_sequence)
                    
                    all_metadata_rows.append({
                        'sample_id': f"{sample_id}_aug_{i+1}",
                        'image_path': aug_filename,
                        **base_record
                    })
            else:
                # Validation/Test: Only original
                all_metadata_rows.append({
                    'sample_id': original_sample_id_v2,
                    'image_path': original_filename,
                    **base_record
                })

        except Exception as e:
            if debug:
                print(f"Error processing {master_row.get('sample_id')}: {e}")
                traceback.print_exc()
                sys.exit(1)
            continue
            
    print(f"  - Processed. Created: {counts['created']}, Skipped: {counts['skipped']}")
    return all_metadata_rows, counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    random.seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)
    
    # Ensure output dir exists
    os.makedirs(cfg.DATASET_DIR, exist_ok=True)

    print("--- Creating RAW V2 Dataset (Matches Colleague: Raw Crop) ---")
    print(f"Target Directory: {cfg.DATASET_DIR}")
    
    try:
        train_df = pd.read_csv(cfg.TRAIN_SPLIT_PATH)
        val_df = pd.read_csv(cfg.VAL_SPLIT_PATH)
        test_df = pd.read_csv(cfg.TEST_SPLIT_PATH)
    except FileNotFoundError:
        sys.exit(f"FATAL: Split files not found. Please run split_data_maksym.py first.")

    train_meta, _ = process_split(train_df, True, cfg.DATASET_DIR, args.debug)
    val_meta, _ = process_split(val_df, False, cfg.DATASET_DIR, args.debug)
    test_meta, _ = process_split(test_df, False, cfg.DATASET_DIR, args.debug)
    
    pd.DataFrame(train_meta).to_csv(cfg.TRAIN_METADATA_PATH, index=False)
    pd.DataFrame(val_meta).to_csv(cfg.VAL_METADATA_PATH, index=False)
    pd.DataFrame(test_meta).to_csv(cfg.TEST_METADATA_PATH, index=False)
    
    print(f"\nDone. Metadata saved to {cfg.TRAIN_METADATA_PATH}")

if __name__ == "__main__":
    main()
    
# python src_cnn_v2/create_cnn_dataset_v2.py --debug
