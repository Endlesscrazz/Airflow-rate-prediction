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


from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.augmentation_utils import add_gaussian_noise, augment_geometric
from src_cnn_v2.logging_utils_v2 import log_experiment_details


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
    if x_end - x_start < crop_size:
        x_start = x_end - crop_size
    y_start = max(0, center_y - half_crop)
    y_end = min(H, y_start + crop_size)
    if y_end - y_start < crop_size:
        y_start = y_end - crop_size
    return frames[y_start:y_end, x_start:x_end, :]


def add_gaussian_noise(sequence, noise_level):
    """Adds Gaussian noise to a sequence of frames."""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise


def process_split(df_split, is_training_set, output_dir, debug=False):
    all_metadata_rows = []
    failed_samples = []
    desc = "Processing " + ("Training" if is_training_set else "Test/Validation") + " Samples"
    
    iterator = df_split.iterrows()
    if not debug:
        iterator = tqdm(iterator, total=len(df_split), desc=desc)

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

            # --- INTELLIGENT SKIPPING LOGIC ---
            if os.path.exists(original_filepath):
                if debug: print(f"  - Base crop exists for {sample_id}, skipping creation.")
                # Load the existing crop if we need it for augmentation
                if is_training_set:
                    cropped_sequence = np.load(original_filepath)
            else:
                # --- This is the slow part, only run if the file is missing ---
                mat_filepath, mask_dir_path, found_config_key = (None, None, None)
                for d_key, d_conf in cfg.DATASET_CONFIGS.items():
                    video_search_pattern = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"], '**', f"{video_id}.mat")
                    video_results = glob.glob(video_search_pattern, recursive=True)
                    if video_results:
                        mat_filepath, found_config_key = video_results[0], d_key
                        break
                if not mat_filepath: raise FileNotFoundError(f".mat file not found for video_id '{video_id}'")
                
                mask_subfolder = cfg.DATASET_CONFIGS[found_config_key]["dataset_subfolder"]
                mask_search_pattern = os.path.join(cfg.RAW_MASK_PARENT_DIR, mask_subfolder, '**', video_id)
                mask_dir_results = glob.glob(mask_search_pattern, recursive=True)
                mask_dir_path = next((path for path in mask_dir_results if os.path.isdir(path)), None)
                if not mask_dir_path: raise FileNotFoundError(f"Mask directory not found for video_id '{video_id}'")
                
                path1 = os.path.join(mask_dir_path, f"{video_id}_mask_{numeric_hole_id}.npy")
                path2 = os.path.join(mask_dir_path, f"{video_id}_mask_0.npy")
                individual_mask_path = path1 if os.path.exists(path1) else (path2 if numeric_hole_id == '1' and os.path.exists(path2) else None)
                if not individual_mask_path: raise FileNotFoundError("Mask file not found.")

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
                np.save(original_filepath, cropped_sequence)

            # --- Metadata is always added ---
            all_metadata_rows.append({'sample_id': original_sample_id_v2, 'original_sample_id': sample_id, 'video_id': video_id, 'hole_id': hole_id, 'image_path': original_filename, 'airflow_rate': master_row['airflow_rate'], 'delta_T': master_row['delta_T']})

            # --- Augmentation Loop (with skipping) ---
            if is_training_set:
                aug_params = cfg.V2_DATASET_PARAMS.get("AUGMENTATION_PARAMS", {})
                for i in range(cfg.V2_DATASET_PARAMS["NUM_AUGMENTATIONS"]):
                    aug_sample_id = f"{sample_id}_aug_{i+1}"
                    aug_filename = f"{aug_sample_id}.npy"
                    aug_filepath = os.path.join(output_dir, aug_filename)
                    
                    if not os.path.exists(aug_filepath):
                        if cropped_sequence is None: # Should not happen if logic is correct
                           cropped_sequence = np.load(original_filepath)
                        
                        augmented_sequence = cropped_sequence.copy()
                        if i % 2 == 0:
                            augmented_sequence = add_gaussian_noise(augmented_sequence, noise_level=aug_params.get("NOISE_LEVEL", 0.05))
                        elif cfg.V2_DATASET_PARAMS.get("ENABLE_GEOMETRIC_AUGMENTATION", False):
                            augmented_sequence = augment_geometric(augmented_sequence, rotation_degrees=aug_params.get("ROTATION_DEGREES", 10), translation_frac=aug_params.get("TRANSLATION_FRAC", 0.1))
                        
                        np.save(aug_filepath, augmented_sequence)
                    
                    all_metadata_rows.append({'sample_id': aug_sample_id, 'original_sample_id': sample_id, 'video_id': video_id, 'hole_id': hole_id, 'image_path': aug_filename, 'airflow_rate': master_row['airflow_rate'], 'delta_T': master_row['delta_T']})

            # --- AUGMENTATION ---
            if is_training_set:
                aug_params = cfg.V2_DATASET_PARAMS.get("AUGMENTATION_PARAMS", {})
                
                for i in range(cfg.V2_DATASET_PARAMS["NUM_AUGMENTATIONS"]):
                    aug_sample_id = f"{sample_id}_aug_{i+1}"
                    aug_filename = f"{aug_sample_id}.npy"
                    
                    augmented_sequence = cropped_sequence.copy()
                    
                    # ---AUGMENTATION LOGIC ---
                    # Apply noise augmentation to roughly half of the copies
                    if i % 2 == 0:
                        augmented_sequence = add_gaussian_noise(
                            augmented_sequence, 
                            noise_level=aug_params.get("NOISE_LEVEL", 0.05)
                        )
                    
                    # Apply geometric augmentation to the other half, if enabled
                    if i % 2 != 0 and cfg.V2_DATASET_PARAMS.get("ENABLE_GEOMETRIC_AUGMENTATION", False):
                        augmented_sequence = augment_geometric(
                            augmented_sequence,
                            rotation_degrees=aug_params.get("ROTATION_DEGREES", 10),
                            translation_frac=aug_params.get("TRANSLATION_FRAC", 0.1)
                        )

                    np.save(os.path.join(output_dir, aug_filename), augmented_sequence)
                    
                    # Add metadata for the augmented sample
                    all_metadata_rows.append({
                        'sample_id': aug_sample_id, 'original_sample_id': sample_id,
                        'video_id': video_id, 'hole_id': hole_id, 'image_path': aug_filename,
                        'airflow_rate': master_row['airflow_rate'], 'delta_T': master_row['delta_T']
                    })

        except Exception as e:
            if debug:
                print(f"\n\n--- SCRIPT STOPPED DUE TO ERROR ---")
                print(
                    f"Failed on sample_id: '{master_row.get('sample_id', 'N/A')}'")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Details: {e}")
                traceback.print_exc()
                sys.exit(1)
            failed_samples.append(
                {'sample_id': master_row.get('sample_id', 'N/A'), 'error': str(e)})
            continue

    return all_metadata_rows, failed_samples


def main():
    # DEBUG FLAG 
    DEBUG = False

    random.seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)

    output_dir = os.path.join(
        cfg.OUTPUT_DIR, cfg.V2_DATASET_PARAMS["OUTPUT_SUBDIR"])
    os.makedirs(output_dir, exist_ok=True)

    train_split_path = cfg.TRAIN_SPLIT_PATH
    val_split_path = cfg.VAL_SPLIT_PATH
    test_split_path = cfg.TEST_SPLIT_PATH

    print("--- Starting V2 Dataset Creation (Cropped & Augmented) ---")
    print(f"Output directory: {output_dir}")

    try:
        train_df = pd.read_csv(train_split_path)
        val_df = pd.read_csv(val_split_path)
        test_df = pd.read_csv(test_split_path)
        print(
            f"Loaded {len(train_df)} train, {len(val_df)} validation, and {len(test_df)} test samples from pre-defined splits.")
    except FileNotFoundError:
        sys.exit(
            f"FATAL: Split files not found. Please run 'split_data_v2.py' first.")

    # --- Process all three splits ---
    train_metadata, train_fails = process_split(
        train_df, is_training_set=True, output_dir=output_dir, debug=DEBUG)
    val_metadata, val_fails = process_split(
        val_df, is_training_set=False, output_dir=output_dir, debug=DEBUG)
    test_metadata, test_fails = process_split(
        test_df, is_training_set=False, output_dir=output_dir, debug=DEBUG)

    failed_samples = train_fails + val_fails + test_fails

    if not (train_metadata or val_metadata or test_metadata):
        print("\nFATAL: No metadata was generated. Check for errors.")
        return

    # --- Save metadata for each split ---
    df_meta_train = pd.DataFrame(train_metadata)
    df_meta_val = pd.DataFrame(val_metadata)
    df_meta_test = pd.DataFrame(test_metadata)

    df_meta_train.to_csv(cfg.TRAIN_METADATA_PATH, index=False)
    df_meta_val.to_csv(cfg.VAL_METADATA_PATH, index=False)
    df_meta_test.to_csv(cfg.TEST_METADATA_PATH, index=False)

    print(f"\nSuccessfully created V2 dataset.")
    print(
        f"  Training samples created: {len(df_meta_train)} (including augmentations)")
    print(f"  Validation samples created: {len(df_meta_val)}")
    print(f"  Test samples created: {len(df_meta_test)}")
    print(f"  Metadata saved to '{output_dir}'")

    # --- NEW: LOG EXPERIMENT PARAMETERS ---
    log_filepath = os.path.join(
        cfg.EXPERIMENT_RESULTS_DIR, "experiment_summary.txt")
    os.makedirs(cfg.EXPERIMENT_RESULTS_DIR, exist_ok=True)

    data_creation_params = {
        "Experiment Name": cfg.EXPERIMENT_NAME,
        "Source Ground Truth CSV": os.path.basename(cfg.GROUND_TRUTH_CSV_PATH),
        "V2 Dataset Parameters": cfg.V2_DATASET_PARAMS,
        "Frames Per Sample": cfg.NUM_FRAMES_PER_SAMPLE,
        "Focus Duration (seconds)": cfg.FOCUS_DURATION_SECONDS,
        "Final Train Samples (with augmentations)": len(df_meta_train),
        "Final Validation Samples": len(df_meta_val),
        "Final Test Samples": len(df_meta_test),
        "Original Train Samples": len(train_df),
        "Original Validation Samples": len(val_df),
        "Original Test Samples": len(test_df),
    }

    log_experiment_details(
        log_filepath, "Data Creation Parameters", data_creation_params)

    if failed_samples:
        print(
            f"\nWarning: {len(failed_samples)} original samples failed during processing.")
        for i, failed in enumerate(failed_samples[:5]):
            print(
                f"  - Sample ID: '{failed['sample_id']}', Reason: {failed['error']}")


if __name__ == "__main__":
    main()

# python src_cnn_v2/create_cnn_dataset_v2.py
