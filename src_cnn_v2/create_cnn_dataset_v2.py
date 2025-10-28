# src_cnn_v2/create_cnn_dataset_v2.py
"""
Prepares a V2 dataset for the bottom-up CNN approach. (VERSION 5 - FINAL)

- Reads pre-generated coordinate .json files to find leak epicenters.
- Creates fixed-size .npy crops.
- Implements a "Clean Replay" strategy for robust training data balance.
- Creates configurable augmentations (geometric + noise) for the training set.
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
import json
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.augmentation_utils import add_gaussian_noise, augment_geometric
from src_cnn_v2.logging_utils_v2 import log_experiment_details

def crop_sequence(frames, center_x, center_y, crop_size):
    H, W, T = frames.shape
    half_crop = crop_size // 2
    x_start = max(0, center_x - half_crop)
    x_end = min(W, x_start + crop_size)
    if x_end - x_start < crop_size: x_start = x_end - crop_size
    y_start = max(0, center_y - half_crop)
    y_end = min(H, y_start + crop_size)
    if y_end - y_start < crop_size: y_start = y_end - crop_size
    return frames[y_start:y_end, x_start:x_end, :]

def process_split(df_split, split_name, output_dir, debug=False):
    all_metadata_rows = []
    failed_samples = []
    desc = f"Processing {split_name.capitalize()} Samples"
    
    iterator = df_split.iterrows()
    if not debug:
        iterator = tqdm(iterator, total=len(df_split), desc=desc)

    for index, master_row in iterator:
        try:
            video_id = master_row['video_id']
            hole_id = str(master_row['hole_id'])
            sample_id = master_row['sample_id']
            numeric_hole_id = hole_id.split('_')[0]

            # Define path for the original, clean cropped file
            original_sample_id_v2 = f"{sample_id}_orig"
            original_filename = f"{original_sample_id_v2}.npy"
            original_filepath = os.path.join(output_dir, original_filename)

            cropped_sequence = None

            # This block loads or creates the clean, original crop. It runs for all splits.
            if not os.path.exists(original_filepath):
                mat_filepath, found_config_key = (None, None)
                for d_key, d_conf in cfg.DATASET_CONFIGS.items():
                    video_search_pattern = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"], '**', f"{video_id}.mat")
                    video_results = glob.glob(video_search_pattern, recursive=True)
                    if video_results:
                        mat_filepath, found_config_key = video_results[0], d_key
                        break
                if not mat_filepath: raise FileNotFoundError(f".mat file not found for video_id '{video_id}'")
                
                coord_subfolder = cfg.DATASET_CONFIGS[found_config_key]["dataset_subfolder"]
                coord_search_pattern = os.path.join(cfg.RAW_MASK_PARENT_DIR, coord_subfolder, '**', video_id)
                coord_dir_results = glob.glob(coord_search_pattern, recursive=True)
                coord_dir_path = next((path for path in coord_dir_results if os.path.isdir(path)), None)
                if not coord_dir_path: raise FileNotFoundError(f"Coordinate directory not found for video_id '{video_id}'")
                
                coord_path = os.path.join(coord_dir_path, f"{video_id}_coordinates.json")
                if not os.path.exists(coord_path):
                    raise FileNotFoundError(f"Coordinates file not found for video '{video_id}' at: {coord_path}")

                with open(coord_path, 'r') as f:
                    all_leaks_in_video = json.load(f)
                
                target_leak_data = next((leak for leak in all_leaks_in_video if str(leak['hole_id']) == numeric_hole_id), None)
                if target_leak_data is None:
                    raise ValueError(f"Hole ID '{numeric_hole_id}' not found in coordinates file: {coord_path}")

                center_x = target_leak_data['center_x']
                center_y = target_leak_data['center_y']
                
                frames = scipy.io.loadmat(mat_filepath).get('TempFrames').astype(np.float32)
                
                end_frame = min(frames.shape[2], int(cfg.FOCUS_DURATION_SECONDS * cfg.TRUE_FPS))
                if end_frame < cfg.NUM_FRAMES_PER_SAMPLE: raise ValueError(f"Video too short ({frames.shape[2]} frames)")
                
                frame_indices = np.linspace(0, end_frame - 1, cfg.NUM_FRAMES_PER_SAMPLE, dtype=int)
                selected_frames = frames[:, :, frame_indices]
                cropped_frames = crop_sequence(selected_frames, center_x, center_y, cfg.V2_DATASET_PARAMS["CROP_SIZE"])
                if cropped_frames.shape[:2] != (cfg.V2_DATASET_PARAMS["CROP_SIZE"], cfg.V2_DATASET_PARAMS["CROP_SIZE"]):
                    raise ValueError(f"Cropped shape is incorrect: {cropped_frames.shape[:2]}")
                
                cropped_sequence = cropped_frames.transpose(2, 0, 1)

                # --- ADD INSTANCE-WISE NORMALIZATION ---
                seq_max = np.max(cropped_sequence)
                if seq_max > 1e-6:
                    cropped_sequence = cropped_sequence / seq_max
                # --- END ---

                np.save(original_filepath, cropped_sequence)

            # This is the base record for all generated metadata rows
            base_record = {
                'original_sample_id': sample_id, 'video_id': video_id, 'hole_id': hole_id,
                'airflow_rate': master_row['airflow_rate'], 'delta_T': master_row['delta_T']
            }

            # --- LOGIC FOR TRAINING SET ---
            if split_name == "train":
                # Add some "replays" of the clean original sample
                #replay_factor = max(1, cfg.V2_DATASET_PARAMS["NUM_AUGMENTATIONS"] // 9)
                replay_factor = 1
                for i in range(replay_factor):
                    replay_sample_id = f"{sample_id}_orig_replay_{i}"
                    all_metadata_rows.append({
                        'sample_id': replay_sample_id,
                        'image_path': original_filename,
                        **base_record
                    })
                
                if cropped_sequence is None:
                    cropped_sequence = np.load(original_filepath)

                # Create N augmented (noisy) versions
                aug_params = cfg.V2_DATASET_PARAMS.get("AUGMENTATION_PARAMS", {})
                for i in range(cfg.V2_DATASET_PARAMS["NUM_AUGMENTATIONS"]):
                    aug_sample_id = f"{sample_id}_aug_{i+1}"
                    aug_filename = f"{aug_sample_id}.npy"
                    aug_filepath = os.path.join(output_dir, aug_filename)
                    
                    if not os.path.exists(aug_filepath):
                        augmented_sequence = cropped_sequence.copy()
                        # NOTE: Geometric augmentation is disabled in your config, but this handles it
                        if cfg.V2_DATASET_PARAMS.get("ENABLE_GEOMETRIC_AUGMENTATION", False):
                            augmented_sequence = augment_geometric(
                                augmented_sequence,
                                rotation_degrees=aug_params.get("ROTATION_DEGREES", 10),
                                translation_frac=aug_params.get("TRANSLATION_FRAC", 0.1)
                            )
                        
                        augmented_sequence = add_gaussian_noise(
                            augmented_sequence, 
                            noise_level=aug_params.get("NOISE_LEVEL", 0.05)
                        )
                        np.save(aug_filepath, augmented_sequence)
                    
                    all_metadata_rows.append({
                        'sample_id': aug_sample_id,
                        'image_path': aug_filename,
                        **base_record
                    })

            # --- NEW LOGIC FOR VALIDATION SET ---
            elif split_name == "validation":
                if cropped_sequence is None:
                    cropped_sequence = np.load(original_filepath)
                
                # Create a SINGLE noisy version for validation
                val_aug_sample_id = f"{sample_id}_val_aug"
                val_aug_filename = f"{val_aug_sample_id}.npy"
                val_aug_filepath = os.path.join(output_dir, val_aug_filename)

                if not os.path.exists(val_aug_filepath):
                    aug_params = cfg.V2_DATASET_PARAMS.get("AUGMENTATION_PARAMS", {})
                    noise_level = aug_params.get("NOISE_LEVEL", 0.05)
                    noisy_sequence = add_gaussian_noise(cropped_sequence.copy(), noise_level)
                    np.save(val_aug_filepath, noisy_sequence)
                
                # Add only the single noisy sample to the metadata
                all_metadata_rows.append({
                    'sample_id': val_aug_sample_id,
                    'image_path': val_aug_filename,
                    **base_record
                })

            # --- LOGIC FOR TEST SET (AND ANY OTHER CASE) ---
            else: # This will handle split_name == "test"
                # For the test set, only use the original, clean file
                all_metadata_rows.append({
                    'sample_id': original_sample_id_v2,
                    'image_path': original_filename,
                    **base_record
                })
        except Exception as e:
            if debug:
                print(f"\n\n--- SCRIPT STOPPED DUE TO ERROR ---")
                print(f"Failed on sample_id: '{master_row.get('sample_id', 'N/A')}'")
                traceback.print_exc()
                sys.exit(1)
            failed_samples.append({'sample_id': master_row.get('sample_id', 'N/A'), 'error': str(e)})
            continue
    return all_metadata_rows, failed_samples

def main():
    parser = argparse.ArgumentParser(description="V2 Dataset Creation Script.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode to stop on first error.")
    args = parser.parse_args()
    random.seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)
    output_dir = cfg.DATASET_DIR
    os.makedirs(output_dir, exist_ok=True)
    print("--- Starting V2 Dataset Creation (Cropped & Augmented) ---")
    print(f"Output directory: {output_dir}")
    try:
        train_df = pd.read_csv(cfg.TRAIN_SPLIT_PATH)
        val_df = pd.read_csv(cfg.VAL_SPLIT_PATH)
        test_df = pd.read_csv(cfg.TEST_SPLIT_PATH)
    except FileNotFoundError:
        sys.exit(f"FATAL: Split files not found. Please run 'split_data_v2.py' first.")
    train_metadata, train_fails = process_split(train_df, split_name="train", output_dir=output_dir, debug=args.debug)
    val_metadata, val_fails = process_split(val_df, split_name="validation", output_dir=output_dir, debug=args.debug)
    test_metadata, test_fails = process_split(test_df, split_name="test", output_dir=output_dir, debug=args.debug)
    if not (train_metadata or val_metadata or test_metadata):
        print("\nFATAL: No metadata was generated. Check for errors.")
        return
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

    # if failed_samples:
    #     print(
    #         f"\nWarning: {len(failed_samples)} original samples failed during processing.")
    #     for i, failed in enumerate(failed_samples[:5]):
    #         print(
    #             f"  - Sample ID: '{failed['sample_id']}', Reason: {failed['error']}")


if __name__ == "__main__":
    main()

# python src_cnn_v2/create_cnn_dataset_v2.py
