# src_cnn_v2/create_dataset_v2.py
"""
Prepares a V2 dataset for the bottom-up CNN approach.

This script reads pre-defined train/validation splits, then processes video
sequences by:
1. Finding the hotspot centroid using the provided mask.
2. Extracting a small, fixed-size crop around the centroid for each frame.
3. For TRAINING samples ONLY, creating multiple augmented (noisy) versions.
4. Saving the cropped sequences as .npy files and creating a new metadata CSV.
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy.io
import cv2
from tqdm import tqdm
import glob
import random
import traceback

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import V2 Config
from src_cnn_v2 import config_v2 as cfg

def get_mask_centroid(mask):
    """Calculates the center of mass of a binary mask."""
    try:
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0:
            true_points = np.argwhere(mask)
            return tuple(true_points[0][[1, 0]]) if len(true_points) > 0 else (None, None)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    except Exception:
        return None, None

def crop_sequence(frames, center_x, center_y, crop_size):
    """Crops each frame in a sequence around a given center point."""
    H, W, T = frames.shape
    half_crop = crop_size // 2
    x_start = max(0, center_x - half_crop)
    x_end = min(W, x_start + crop_size)
    if x_end - x_start < crop_size: x_start = x_end - crop_size
    y_start = max(0, center_y - half_crop)
    y_end = min(H, y_start + crop_size)
    if y_end - y_start < crop_size: y_start = y_end - crop_size
    return frames[y_start:y_end, x_start:x_end, :]

def add_gaussian_noise(sequence, noise_level):
    """Adds Gaussian noise to a sequence of frames."""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def process_split(df_split, is_training_set, output_dir, debug=False):
    """Processes a dataframe split (train or val) and returns its metadata."""
    all_metadata_rows = []
    failed_samples = []

    desc = "Processing Training Samples" if is_training_set else "Processing Validation Samples"
    
    # Disable tqdm in debug mode for cleaner output
    iterator = df_split.iterrows()
    if not debug:
        iterator = tqdm(iterator, total=len(df_split), desc=desc)

    for index, master_row in iterator:
        try:
            video_id = master_row['video_id']
            hole_id = str(master_row['hole_id'])
            sample_id = master_row['sample_id']

            if debug:
                print(f"\n--- DEBUG: Processing sample_id: {sample_id} ---")

            # --- Find Raw Files ---
            mat_filepath, mask_dir_path, found_config_key = (None, None, None)
            for d_key, d_conf in cfg.DATASET_CONFIGS.items():
                video_search_pattern = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"], '**', f"{video_id}.mat")
                if debug: print(f"  - Searching for .mat: {video_search_pattern}")
                video_results = glob.glob(video_search_pattern, recursive=True)
                if video_results:
                    mat_filepath, found_config_key = video_results[0], d_key
                    if debug: print(f"  -> FOUND .mat: {mat_filepath}")
                    break
            
            if not mat_filepath: raise FileNotFoundError(f".mat file not found for video_id '{video_id}'")
            
            mask_subfolder = cfg.DATASET_CONFIGS[found_config_key]["dataset_subfolder"]
            mask_search_pattern = os.path.join(cfg.RAW_MASK_PARENT_DIR, mask_subfolder, '**', video_id)
            if debug: print(f"  - Searching for mask dir: {mask_search_pattern}")
            mask_dir_results = glob.glob(mask_search_pattern, recursive=True)
            if mask_dir_results:
                for path in mask_dir_results:
                    if os.path.isdir(path):
                        mask_dir_path = path
                        if debug: print(f"  -> FOUND mask dir: {mask_dir_path}")
                        break
            
            if not mask_dir_path: raise FileNotFoundError(f"Mask directory not found for video_id '{video_id}'")

            # --- Robust Mask File Finding ---
            path1 = os.path.join(mask_dir_path, f"{video_id}_mask_{hole_id}.npy")
            path2 = os.path.join(mask_dir_path, f"{video_id}_mask_0.npy")
            
            individual_mask_path = None
            if os.path.exists(path1):
                individual_mask_path = path1
            elif hole_id == '1' and os.path.exists(path2):
                individual_mask_path = path2

            if debug:
                print(f"  - Checking for mask path 1: {path1}")
                if hole_id == '1': print(f"  - Checking for mask path 2: {path2}")
                if individual_mask_path: print(f"  -> FOUND mask file: {individual_mask_path}")

            if not individual_mask_path: raise FileNotFoundError("Mask file not found.")

            # --- Load Data & Crop ---
            individual_mask = np.load(individual_mask_path)
            frames = scipy.io.loadmat(mat_filepath).get('TempFrames').astype(np.float32)
            
            center_x, center_y = get_mask_centroid(individual_mask)
            if center_x is None: raise ValueError("Mask is empty.")
                
            end_frame = min(frames.shape[2], int(cfg.FOCUS_DURATION_SECONDS * cfg.TRUE_FPS))
            if end_frame < cfg.NUM_FRAMES_PER_SAMPLE: raise ValueError(f"Video too short ({frames.shape[2]} frames)")
                
            frame_indices = np.linspace(0, end_frame - 1, cfg.NUM_FRAMES_PER_SAMPLE, dtype=int)
            selected_frames = frames[:, :, frame_indices]
            
            cropped_frames = crop_sequence(selected_frames, center_x, center_y, cfg.V2_DATASET_PARAMS["CROP_SIZE"])
            if cropped_frames.shape[:2] != (cfg.V2_DATASET_PARAMS["CROP_SIZE"], cfg.V2_DATASET_PARAMS["CROP_SIZE"]):
                raise ValueError(f"Cropped shape is incorrect: {cropped_frames.shape[:2]}")
            
            cropped_sequence = cropped_frames.transpose(2, 0, 1)

            # --- Save Original (Clean) Sample ---
            original_sample_id = f"{sample_id}_orig"
            original_filename = f"{original_sample_id}.npy"
            np.save(os.path.join(output_dir, original_filename), cropped_sequence)
            
            all_metadata_rows.append({'sample_id': original_sample_id, 'image_path': original_filename, 'airflow_rate': master_row['airflow_rate'], 'delta_T': master_row['delta_T']})

            # --- AUGMENTATION ---
            if is_training_set:
                for i in range(cfg.V2_DATASET_PARAMS["NUM_AUGMENTATIONS"]):
                    # ... (augmentation logic as before) ...
                    aug_sample_id = f"{sample_id}_aug_{i+1}"
                    aug_filename = f"{aug_sample_id}.npy"
                    noisy_sequence = add_gaussian_noise(cropped_sequence, cfg.V2_DATASET_PARAMS["NOISE_LEVEL"])
                    np.save(os.path.join(output_dir, aug_filename), noisy_sequence)
                    all_metadata_rows.append({'sample_id': aug_sample_id, 'image_path': aug_filename, 'airflow_rate': master_row['airflow_rate'], 'delta_T': master_row['delta_T']})

        except Exception as e:
            if debug:
                print(f"\n\n--- SCRIPT STOPPED DUE TO ERROR ---")
                print(f"Failed on sample_id: '{master_row.get('sample_id', 'N/A')}'")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Details: {e}")
                traceback.print_exc() # Print full traceback
                sys.exit(1) # Stop execution
            
            failed_samples.append({'sample_id': master_row.get('sample_id', 'N/A'), 'error': str(e)})
            continue
            
    return all_metadata_rows, failed_samples

def main():
    # --- DEBUG FLAG ---
    DEBUG = False
    # Set to False to run on the full dataset without stopping on errors.

    random.seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)
    
    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.V2_DATASET_PARAMS["OUTPUT_SUBDIR"])
    os.makedirs(output_dir, exist_ok=True)
    
    train_split_path = os.path.join(cfg.OUTPUT_DIR, "train_split.csv")
    val_split_path = os.path.join(cfg.OUTPUT_DIR, "val_split.csv")
    test_split_path = os.path.join(cfg.OUTPUT_DIR, "test_split.csv")

    print("--- Starting V2 Dataset Creation (Cropped & Augmented) ---")
    print(f"Output directory: {output_dir}")

    try:
        train_df = pd.read_csv(train_split_path)
        val_df = pd.read_csv(val_split_path)
        test_df = pd.read_csv(test_split_path)
        print(f"Loaded {len(train_df)} train, {len(val_df)} validation, and {len(test_df)} test samples from pre-defined splits.")
    except FileNotFoundError:
        sys.exit(f"FATAL: Split files not found. Please run 'split_data_v2.py' first.")

    # --- Process all three splits ---
    train_metadata, train_fails = process_split(train_df, is_training_set=True, output_dir=output_dir, debug=DEBUG)
    val_metadata, val_fails = process_split(val_df, is_training_set=False, output_dir=output_dir, debug=DEBUG)
    test_metadata, test_fails = process_split(test_df, is_training_set=False, output_dir=output_dir, debug=DEBUG)

    failed_samples = train_fails + val_fails + test_fails

    if not (train_metadata or val_metadata or test_metadata):
        print("\nFATAL: No metadata was generated. Check for errors.")
        return

    # --- Save metadata for each split ---
    df_meta_train = pd.DataFrame(train_metadata)
    df_meta_val = pd.DataFrame(val_metadata)
    df_meta_test = pd.DataFrame(test_metadata)
    
    df_meta_train.to_csv(os.path.join(output_dir, "train_metadata_v2.csv"), index=False)
    df_meta_val.to_csv(os.path.join(output_dir, "val_metadata_v2.csv"), index=False)
    df_meta_test.to_csv(os.path.join(output_dir, "test_metadata_v2.csv"), index=False)
    
    print(f"\nSuccessfully created V2 dataset.")
    print(f"  Training samples created: {len(df_meta_train)} (including augmentations)")
    print(f"  Validation samples created: {len(df_meta_val)}")
    print(f"  Test samples created: {len(df_meta_test)}")
    print(f"  Metadata saved to '{output_dir}'")
    
    if failed_samples:
        print(f"\nWarning: {len(failed_samples)} original samples failed during processing.")
        for i, failed in enumerate(failed_samples[:5]):
            print(f"  - Sample ID: '{failed['sample_id']}', Reason: {failed['error']}")

if __name__ == "__main__":
    main()

# python src_cnn_v2/create_cnn_dataset_v2.py