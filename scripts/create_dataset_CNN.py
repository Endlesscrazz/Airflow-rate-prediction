# scripts/create_dataset.py
"""
Prepares a dataset for CNN-based models by using a master ground truth CSV.
This single, configurable script can generate one of four dataset types.
All configurations are imported from src_cnn.config.

Run using:
python -m scripts.create_dataset --type [thermal|thermal_masked|flow|hybrid]
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy.io
import cv2
from tqdm import tqdm
import datetime
import argparse
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn import feature_engineering
from src_cnn import config as cfg


def save_dataset_parameters(output_dir, dataset_type_str, num_samples, final_df, context_cols, dynamic_cols, seq_len, num_ch):
    """Saves the key parameters of the generated dataset to a text file."""
    params = {
        "Dataset Directory": os.path.relpath(output_dir, project_root),
        "Dataset Type": dataset_type_str,
        "Generation Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "---": "---",
        "Total Samples": num_samples,
        "Sequence Length": seq_len,
        "Image Channels": num_ch,
        "Image Target Size": f"{cfg.IMAGE_TARGET_SIZE[0]}x{cfg.IMAGE_TARGET_SIZE[1]}",
        "---": "---",
        "Context Features": sorted(context_cols),
        "Dynamic Features": sorted(dynamic_cols),
        "---": "---",
        "Handcrafted Features Used": sorted(list(set(cfg.HANDCRAFTED_FEATURES_TO_EXTRACT))),
    }
    save_path = os.path.join(output_dir, "dataset_parameters.txt")
    with open(save_path, 'w') as f:
        f.write("--- DATASET PARAMETERS ---\n\n")
        for key, value in params.items():
            if isinstance(value, list):
                f.write(f"{key:<35}: \n")
                for item in value:
                    f.write(f"{'':<37}- {item}\n")
            else:
                f.write(f"{key:<35}: {value}\n")
    print(f"\nSuccessfully saved dataset parameters to: {save_path}")


def find_data_files(video_id):
    """Finds the .mat video file and all associated .npy mask files for a given video_id."""
    mat_filepath = None
    mask_dir = None
    
    # Search for the .mat file and mask directory across all configured raw data locations
    for d_conf in cfg.DATASET_CONFIGS.values():
        # Check for video file
        potential_mat_path = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"], f"{video_id}.mat")
        if os.path.exists(potential_mat_path):
            mat_filepath = potential_mat_path
        
        # Check for mask directory
        potential_mask_dir = os.path.join(cfg.RAW_MASK_PARENT_DIR, d_conf["dataset_subfolder"], video_id)
        if os.path.isdir(potential_mask_dir):
            mask_dir = potential_mask_dir
        
        # If we found both, we can stop searching
        if mat_filepath and mask_dir:
            break

    if not mat_filepath or not mask_dir:
        return None, []

    # Find and sort all mask files in the found directory
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, f"{video_id}_mask_*.npy")))
    return mat_filepath, mask_paths


def create_dataset(dataset_type):
    """Main function to generate the specified dataset."""
    FOCUS_DURATION_FRAMES = int(cfg.FOCUS_DURATION_SECONDS * cfg.TRUE_FPS)

    type_to_folder = {
        'thermal': "dataset_1ch_thermal_hard_crop",
        'thermal_masked': "dataset_2ch_thermal_masked",
        'flow': "dataset_2ch_flow_hard_crop",
        'hybrid': "dataset_3ch_hybrid_hard_crop"
    }
    type_to_str = {
        'thermal': "1-Channel Thermal (Hard Crop)",
        'thermal_masked': "2-Channel Thermal + Mask",
        'flow': "2-Channel Optical Flow",
        'hybrid': "3-Channel Hybrid"
    }
    output_dir = os.path.join(cfg.PROCESSED_DATASET_DIR, type_to_folder.get(dataset_type))
    DATASET_TYPE_STR = type_to_str.get(dataset_type)
    METADATA_SAVE_PATH = os.path.join(output_dir, "metadata.csv")
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Dataset Creation: {DATASET_TYPE_STR} ---")
    
    try:
        ground_truth_df = pd.read_csv(cfg.GROUND_TRUTH_CSV_PATH)
        print(f"Loaded {len(ground_truth_df)} total samples from {cfg.GROUND_TRUTH_CSV_PATH}")
    except Exception as e:
        print(f"FATAL: Could not load master ground truth CSV '{cfg.GROUND_TRUTH_CSV_PATH}'. Error: {e}")
        return

    print("\nProcessing samples based on ground truth CSV...")
    all_metadata, failed_samples = [], []
    final_sequence_length, final_num_channels = 0, 0

    for index, sample_row in tqdm(ground_truth_df.iterrows(), total=len(ground_truth_df), desc="Processing Samples"):
        try:
            video_id = sample_row['video_id']
            hole_id = sample_row['hole_id']
            
            mat_filepath, mask_paths = find_data_files(video_id)

            if not mat_filepath:
                raise FileNotFoundError(f"Could not find .mat file for video_id '{video_id}'")
            if not mask_paths:
                raise FileNotFoundError(f"Could not find any masks for video_id '{video_id}'")

            hole_index_str = str(hole_id).split('_')[0]
            if not hole_index_str.isdigit():
                raise ValueError(f"Could not parse numeric index from hole_id '{hole_id}'")
            hole_index = int(hole_index_str) - 1

            if not (0 <= hole_index < len(mask_paths)):
                raise IndexError(f"Mask index {hole_index} is out of bounds for video '{video_id}' (found {len(mask_paths)} masks)")
            
            individual_mask_path = mask_paths[hole_index]
            individual_mask = np.load(individual_mask_path).astype(bool)

            mat_data = scipy.io.loadmat(mat_filepath)
            frames = mat_data.get('TempFrames').astype(np.float64)
            H, W, T = frames.shape
            end_frame = min(T, FOCUS_DURATION_FRAMES)

            if end_frame < cfg.NUM_FRAMES_PER_SAMPLE:
                raise ValueError(f"Video too short ({T} frames)")

            frame_indices = np.linspace(0, end_frame - 1, cfg.NUM_FRAMES_PER_SAMPLE, dtype=int)
            selected_frames = frames[:, :, frame_indices]

            if not np.any(individual_mask):
                raise ValueError("Mask is empty")

            processed_frames_list = []
            sample_id = f"{video_id}_{hole_id}"
            
            if dataset_type == 'thermal_masked':
                for frame_idx in range(cfg.NUM_FRAMES_PER_SAMPLE):
                    full_frame = selected_frames[:, :, frame_idx]
                    frame_resized = cv2.resize(full_frame, cfg.IMAGE_TARGET_SIZE, interpolation=cv2.INTER_AREA)
                    mask_resized = cv2.resize(individual_mask.astype(np.float32), cfg.IMAGE_TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                    stacked_frame = np.stack([frame_resized, mask_resized], axis=-1)
                    processed_frames_list.append(stacked_frame)
            else:
                x, y, w, h = cv2.boundingRect(individual_mask.astype(np.uint8))
                pad_w = int(w * cfg.ROI_PADDING_PERCENT / 2); pad_h = int(h * cfg.ROI_PADDING_PERCENT / 2)
                x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                x2, y2 = min(W, x + w + pad_w), min(H, y + h + pad_h)
                cropped_rois = selected_frames[y1:y2, x1:x2, :]
                if cropped_rois.size == 0: continue
                if dataset_type == 'thermal':
                    for i in range(cfg.NUM_FRAMES_PER_SAMPLE):
                        processed_frames_list.append(cv2.resize(cropped_rois[:, :, i], cfg.IMAGE_TARGET_SIZE, cv2.INTER_AREA))
                elif dataset_type in ['flow', 'hybrid']:
                    for i in range(cfg.NUM_FRAMES_PER_SAMPLE - 1):
                        prev = cv2.normalize(cv2.resize(cropped_rois[:, :, i], cfg.IMAGE_TARGET_SIZE), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        next = cv2.normalize(cv2.resize(cropped_rois[:, :, i+1], cfg.IMAGE_TARGET_SIZE), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        if dataset_type == 'flow': processed_frames_list.append(flow)
                        else:
                            thermal_ch = (cropped_rois[:, :, i] - cropped_rois[:, :, i].min()) / (cropped_rois[:, :, i].max() - cropped_rois[:, :, i].min() + 1e-6)
                            processed_frames_list.append(np.dstack([cv2.resize(thermal_ch, cfg.IMAGE_TARGET_SIZE), flow]))
            
            if not processed_frames_list: continue

            final_sequence_array = np.stack(processed_frames_list, axis=0)
            final_sequence_length = final_sequence_array.shape[0]
            final_num_channels = final_sequence_array.shape[-1] if final_sequence_array.ndim == 4 else 1

            image_filename = f"{sample_id}.npy"
            image_save_path_full = os.path.join(output_dir, image_filename)
            np.save(image_save_path_full, final_sequence_array)

            handcrafted_features = feature_engineering.calculate_hotspot_features(frames=frames, hotspot_mask=individual_mask)

            current_metadata = {
                'video_id': video_id,
                'sample_id': sample_id,
                'image_path': image_filename,
                'airflow_rate': sample_row['airflow_rate'],
                'delta_T': sample_row['delta_T'],
                'material': sample_row['material']
            }
            if handcrafted_features: current_metadata.update(handcrafted_features)
            all_metadata.append(current_metadata)

        except Exception as e:
            # Add to failed list, but also re-raise the error for the first failure to stop execution for debugging
            failed_samples.append({'video_id': sample_row.get('video_id', 'Unknown'), 'hole_id': sample_row.get('hole_id', 'Unknown'), 'error': str(e)})
            print(f"\n\n--- SCRIPT STOPPED DUE TO ERROR ---")
            print(f"Failed on sample index {index}, video_id: '{sample_row.get('video_id')}', hole_id: '{sample_row.get('hole_id')}'")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            print(f"Please check your file paths, config.py, and ground truth CSV.")
            print(f"-------------------------------------\n")
            raise e # Stop the script on the first error

    # --- Feature Engineering and Saving ---
    if not all_metadata:
        print("Error: No metadata was generated. All samples failed.")
        if failed_samples:
            print(f"\nSummary of the first 5 failures:")
            for i, failed in enumerate(failed_samples[:5]):
                 print(f"  - Sample {i+1}: ID='{failed['video_id']}', Hole='{failed['hole_id']}', Reason: {failed['error']}")
        return

    df_metadata_raw = pd.DataFrame(all_metadata)
    final_df = df_metadata_raw[['sample_id', 'video_id', 'image_path', 'airflow_rate', 'material']].copy()
    context_feature_cols, dynamic_feature_cols = [], []
    final_df['delta_T_log'] = np.log1p(df_metadata_raw['delta_T'].astype(float).clip(lower=0))
    context_feature_cols.append('delta_T_log')

    # (The rest of the feature engineering logic is the same)
    if 'hotspot_area' in cfg.HANDCRAFTED_FEATURES_TO_EXTRACT and 'hotspot_area' in df_metadata_raw.columns:
        if cfg.LOG_TRANSFORM_AREA:
            final_df['hotspot_area_log'] = np.log1p(df_metadata_raw['hotspot_area'].astype(float).clip(lower=0))
            dynamic_feature_cols.append('hotspot_area_log')
        else:
            final_df['hotspot_area'] = df_metadata_raw['hotspot_area']
            dynamic_feature_cols.append('hotspot_area')
    if 'hotspot_avg_temp_change_rate_initial' in cfg.HANDCRAFTED_FEATURES_TO_EXTRACT and 'hotspot_avg_temp_change_rate_initial' in df_metadata_raw.columns:
        if cfg.NORMALIZE_AVG_RATE_INITIAL:
            final_df['hotspot_avg_temp_change_rate_initial_norm'] = df_metadata_raw.apply(
                lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_rate_initial']) else np.nan, axis=1)
            dynamic_feature_cols.append('hotspot_avg_temp_change_rate_initial_norm')
        else:
            final_df['hotspot_avg_temp_change_rate_initial'] = df_metadata_raw['hotspot_avg_temp_change_rate_initial']
            dynamic_feature_cols.append('hotspot_avg_temp_change_rate_initial')
    special_features = ['hotspot_area', 'hotspot_avg_temp_change_rate_initial']
    for f_name in cfg.HANDCRAFTED_FEATURES_TO_EXTRACT:
        if f_name not in special_features and f_name in df_metadata_raw.columns:
            final_df[f_name] = df_metadata_raw[f_name]
            dynamic_feature_cols.append(f_name)
    material_dummies = pd.get_dummies(df_metadata_raw['material'], prefix='material', dtype=int)
    final_df = pd.concat([final_df, material_dummies], axis=1)
    context_feature_cols.extend(material_dummies.columns.tolist())
    all_feature_cols = context_feature_cols + dynamic_feature_cols
    final_df[all_feature_cols] = final_df[all_feature_cols].fillna(final_df[all_feature_cols].median())
    final_df.to_csv(METADATA_SAVE_PATH, index=False)
    
    print(f"\nSuccessfully saved metadata for {len(final_df)} samples to: {METADATA_SAVE_PATH}")
    save_dataset_parameters(
        output_dir=output_dir, dataset_type_str=DATASET_TYPE_STR,
        num_samples=len(final_df), final_df=final_df,
        context_cols=context_feature_cols, dynamic_cols=dynamic_feature_cols,
        seq_len=final_sequence_length, num_ch=final_num_channels
    )

    if failed_samples:
        print(f"\nWarning: {len(failed_samples)} samples failed in total.")

    print(f"\n--- Dataset Creation Complete: {DATASET_TYPE_STR} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset for CNN-based models.")
    parser.add_argument("--type", type=str, required=True, choices=['thermal', 'flow', 'hybrid', 'thermal_masked'],
                        help="The type of dataset to generate.")
    args = parser.parse_args()
    create_dataset(args.type)

"""
python -m scripts.create_dataset_CNN --type thermal_masked
"""
