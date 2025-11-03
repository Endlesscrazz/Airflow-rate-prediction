# src_cnn_v2/create_cnn_dataset_v2.py
"""
Prepares a V2 dataset for the bottom-up CNN approach. (VERSION 6 - FINAL)
- Reads pre-generated coordinate .json files to find leak epicenters.
- Creates fixed-size .npy crops with RAW thermal values.
- Implements a "Clean Replay" strategy for robust training data balance.
- Creates configurable augmentations (geometric + noise) for the training set.
- Correctly counts and logs created vs. skipped files.
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

def process_split(df_split, is_training_set, output_dir, debug=False):
    all_metadata_rows = []
    failed_samples = []
    counts = {'created': 0, 'skipped': 0}
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

            if os.path.exists(original_filepath):
                counts['skipped'] += 1
            else:
                counts['created'] += 1
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

                center_x, center_y = target_leak_data['center_x'], target_leak_data['center_y']
                
                frames = scipy.io.loadmat(mat_filepath).get('TempFrames').astype(np.float32)
                
                end_frame = min(frames.shape[2], int(cfg.FOCUS_DURATION_SECONDS * cfg.TRUE_FPS))
                if end_frame < cfg.NUM_FRAMES_PER_SAMPLE: raise ValueError(f"Video too short ({frames.shape[2]} frames)")
                
                frame_indices = np.linspace(0, end_frame - 1, cfg.NUM_FRAMES_PER_SAMPLE, dtype=int)
                selected_frames = frames[:, :, frame_indices]
                cropped_frames = crop_sequence(selected_frames, center_x, center_y, cfg.V2_DATASET_PARAMS["CROP_SIZE"])
                if cropped_frames.shape[:2] != (cfg.V2_DATASET_PARAMS["CROP_SIZE"], cfg.V2_DATASET_PARAMS["CROP_SIZE"]):
                    raise ValueError(f"Cropped shape is incorrect: {cropped_frames.shape[:2]}")
                
                cropped_sequence = cropped_frames.transpose(2, 0, 1)
                np.save(original_filepath, cropped_sequence)

            base_record = {
                'original_sample_id': sample_id, 'video_id': video_id, 'hole_id': hole_id,
                'airflow_rate': master_row['airflow_rate'], 'delta_T': master_row['delta_T']
            }

            if is_training_set:
                replay_factor = max(1, cfg.V2_DATASET_PARAMS["NUM_AUGMENTATIONS"] // 9)
                for i in range(replay_factor):
                    all_metadata_rows.append({
                        'sample_id': f"{sample_id}_orig_replay_{i}",
                        'image_path': original_filename,
                        **base_record
                    })
                
                if cropped_sequence is None:
                    cropped_sequence = np.load(original_filepath)

                aug_params = cfg.V2_DATASET_PARAMS.get("AUGMENTATION_PARAMS", {})
                for i in range(cfg.V2_DATASET_PARAMS["NUM_AUGMENTATIONS"]):
                    aug_filename = f"{sample_id}_aug_{i+1}.npy"
                    aug_filepath = os.path.join(output_dir, aug_filename)
                    
                    if os.path.exists(aug_filepath):
                        counts['skipped'] += 1
                    else:
                        counts['created'] += 1
                        augmented_sequence = cropped_sequence.copy()
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
                        'sample_id': f"{sample_id}_aug_{i+1}",
                        'image_path': aug_filename,
                        **base_record
                    })
            else:
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
            
    print(f"  - Split processing complete. Created: {counts['created']} new .npy files, Skipped: {counts['skipped']} existing files.")
    return all_metadata_rows, failed_samples, counts

def main():
    parser = argparse.ArgumentParser(description="V2 Dataset Creation Script.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode to stop on first error and get verbose logs.")
    args = parser.parse_args()

    random.seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)
    output_dir = cfg.DATASET_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("--- Starting V2 Dataset Creation (Cropped & Augmented) ---")
    print(f"  - Using data split from random seed: {cfg.RANDOM_STATE}")
    print(f"  - Output directory for .npy files: {output_dir}")

    try:
        train_df = pd.read_csv(cfg.TRAIN_SPLIT_PATH)
        val_df = pd.read_csv(cfg.VAL_SPLIT_PATH)
        test_df = pd.read_csv(cfg.TEST_SPLIT_PATH)
    except FileNotFoundError:
        sys.exit(f"FATAL: Split files for seed {cfg.RANDOM_STATE} not found. Please run 'split_data_v2.py' first.")

    train_metadata, train_fails, train_counts = process_split(train_df, is_training_set=True, output_dir=output_dir, debug=args.debug)
    val_metadata, val_fails, val_counts = process_split(val_df, is_training_set=False, output_dir=output_dir, debug=args.debug)
    test_metadata, test_fails, test_counts = process_split(test_df, is_training_set=False, output_dir=output_dir, debug=args.debug)
    failed_samples = train_fails + val_fails + test_fails

    if not (train_metadata or val_metadata or test_metadata):
        print("\nFATAL: No metadata was generated. Check for errors."); return

    df_meta_train = pd.DataFrame(train_metadata)
    df_meta_val = pd.DataFrame(val_metadata)
    df_meta_test = pd.DataFrame(test_metadata)

    df_meta_train.to_csv(cfg.TRAIN_METADATA_PATH, index=False)
    df_meta_val.to_csv(cfg.VAL_METADATA_PATH, index=False)
    df_meta_test.to_csv(cfg.TEST_METADATA_PATH, index=False)

    print(f"\nSuccessfully created V2 dataset.")
    print(f"  - Training samples created: {len(df_meta_train)}")
    print(f"  - Validation samples created: {len(df_meta_val)}")
    print(f"  - Test samples created: {len(df_meta_test)}")
    print(f"  - Metadata saved to versioned files (e.g., {os.path.basename(cfg.TRAIN_METADATA_PATH)})")

    log_filepath = os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "experiment_summary.txt")
    os.makedirs(cfg.EXPERIMENT_RESULTS_DIR, exist_ok=True)
    data_creation_params = {
        "Experiment Name": cfg.EXPERIMENT_NAME,
        "Source Ground Truth CSV": os.path.basename(cfg.GROUND_TRUTH_CSV_PATH),
        "V2 Dataset Parameters": {k: v for k, v in cfg.V2_DATASET_PARAMS.items() if k != "OUTPUT_SUBDIR"},
        "Frames Per Sample": cfg.NUM_FRAMES_PER_SAMPLE,
        "Focus Duration (seconds)": cfg.FOCUS_DURATION_SECONDS,
        "Final Train Samples (with augmentations)": len(df_meta_train),
        "Final Validation Samples": len(df_meta_val),
        "Final Test Samples": len(df_meta_test),
        "Original Train Samples": len(train_df),
        "Original Validation Samples": len(val_df),
        "Original Test Samples": len(test_df),
        "File Generation Summary": {
            "Training Set": train_counts,
            "Validation Set": val_counts,
            "Test Set": test_counts,
            "Total Created": train_counts['created'] + val_counts['created'] + test_counts['created'],
            "Total Skipped": train_counts['skipped'] + val_counts['skipped'] + test_counts['skipped']
        }
    }
    log_experiment_details(log_filepath, "Data Creation Parameters", data_creation_params)

    if failed_samples:
        print(f"\nWarning: {len(failed_samples)} original samples failed during processing.")
        for i, failed in enumerate(failed_samples[:5]):
            print(f"  - Sample ID: '{failed['sample_id']}', Reason: {failed['error']}")

if __name__ == "__main__":
    main()

# python src_cnn_v2/create_cnn_dataset_v2.py --debug
