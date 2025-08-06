# src_cnn/create_dataset.py
"""
Prepares a dataset for CNN-based models.

This single, configurable script can generate one of three dataset types:
1.  'thermal': 1-channel raw thermal video sequences.
2.  'flow': 2-channel optical flow sequences.
3.  'hybrid': 3-channel sequences (Thermal + Optical Flow).

Run using:
python -m src_cnn.create_dataset --type [thermal|flow|hybrid]
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy.io
import cv2
import fnmatch
from tqdm import tqdm
import datetime
import argparse

# --- Import project modules ---
try:
    from src_feature_based import config, data_utils, feature_engineering
except ImportError:
    print("Error: Could not import from src_feature_based. Make sure your PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Universal Configuration ---
# ROI and Image Processing Parameters
ROI_PADDING_PERCENT = 0.20
TRUE_FPS = 5
FOCUS_DURATION_SECONDS = 10
FOCUS_DURATION_FRAMES = int(FOCUS_DURATION_SECONDS * TRUE_FPS)
NUM_FRAMES_PER_SAMPLE = 25
IMAGE_TARGET_SIZE = (128, 128)

# Handcrafted Feature Engineering Configuration
SELECTED_RAW_FEATURES_TO_EXTRACT = [
    'hotspot_area',
    'hotspot_avg_temp_change_rate_initial',
    'overall_std_deltaT',
    'temp_max_overall_initial',
    'temp_std_avg_initial',
]
LOG_TRANSFORM_AREA = True
NORMALIZE_AVG_RATE_INITIAL = True

# Data source configurations
DATASET_CONFIGS = {
    "gypsum_single_hole": {"material": "gypsum", "dataset_subfolder": "dataset_gypsum", "mask_subfolder": "dataset_gypsum"},
    "gypsum_single_hole2": {"material": "gypsum", "dataset_subfolder": "dataset_gypsum2", "mask_subfolder": "dataset_gypsum2"},
    "brick_cladding_single_hole": {"material": "brick_cladding", "dataset_subfolder": "dataset_brickcladding", "mask_subfolder": "dataset_brickcladding"},
    "brick_cladding_two_holes": {"material": "brick_cladding", "dataset_subfolder": "dataset_two_holes_brickcladding", "mask_subfolder": "dataset_two_holes_brickcladding"}
}

# --- Helper Function for Parameter Saving (from before) ---
def save_dataset_parameters(output_dir, dataset_type_str, num_samples, final_df, context_cols, dynamic_cols, seq_len, num_ch):
    """Saves the key parameters of the generated dataset to a text file."""
    params = {
        "Dataset Directory": output_dir,
        "Dataset Type": dataset_type_str,
        "Generation Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "---": "---",
        "Total Samples": num_samples,
        "Sequence Length": seq_len,
        "Image Channels": num_ch,
        "Image Target Size": f"{IMAGE_TARGET_SIZE[0]}x{IMAGE_TARGET_SIZE[1]}",
        "---": "---",
        "Context Features": sorted(context_cols),
        "Dynamic Features": sorted(dynamic_cols),
        "---": "---",
        "Handcrafted Features Used": sorted(list(set(SELECTED_RAW_FEATURES_TO_EXTRACT))),
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

# --- Main Function ---
def create_dataset(dataset_type):
    """Main function to generate the specified dataset."""
    
    # 1. Configure paths and parameters based on dataset_type
    if dataset_type == 'thermal':
        CNN_DATASET_DIR = "CNN_dataset/dataset_1ch_thermal"
        DATASET_TYPE_STR = "1-Channel Thermal"
    elif dataset_type == 'flow':
        CNN_DATASET_DIR = "CNN_dataset/dataset_2ch_flow"
        DATASET_TYPE_STR = "2-Channel Optical Flow"
    elif dataset_type == 'hybrid':
        CNN_DATASET_DIR = "CNN_dataset/dataset_3ch_hybrid"
        DATASET_TYPE_STR = "3-Channel Hybrid (Thermal + Flow)"
    else:
        raise ValueError("Invalid dataset type specified.")

    METADATA_SAVE_PATH = os.path.join(CNN_DATASET_DIR, "metadata.csv")
    
    print(f"--- Starting Dataset Creation: {DATASET_TYPE_STR} ---")

    os.makedirs(CNN_DATASET_DIR, exist_ok=True)
    print(f"Output will be saved in: {CNN_DATASET_DIR}")

    # Step 1: Discover all video and mask files (This logic is identical for all types)
    print("\nStep 1: Scanning for all video and mask files...")
    video_to_masks_map = {}
    datasets_to_load_keys = list(DATASET_CONFIGS.keys())
    for d_key in datasets_to_load_keys:
        d_config = DATASET_CONFIGS[d_key]
        dataset_path_load = os.path.join(config.DATASET_PARENT_DIR, d_config["dataset_subfolder"])
        mask_root_path_load = os.path.join(config.BASE_MASK_INPUT_DIR, d_config["mask_subfolder"])
        if not os.path.isdir(dataset_path_load) or not os.path.isdir(mask_root_path_load):
            continue
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
                            "material": d_config["material"], "source_dataset": d_key
                        }
                    except Exception: continue
                mat_basename = os.path.splitext(mat_filename_load)[0]
                relative_path_part = os.path.relpath(root_load, dataset_path_load)
                mask_search_dir = os.path.join(mask_root_path_load, relative_path_part, mat_basename)
                if os.path.isdir(mask_search_dir):
                    mask_files_found = fnmatch.filter(os.listdir(mask_search_dir), f"{mat_basename}_mask_*.npy")
                    mask_files_found.extend(fnmatch.filter(os.listdir(mask_search_dir), f"{mat_basename}_sam_mask.npy"))
                    if mask_files_found and video_id in video_to_masks_map:
                        for mask_filename in mask_files_found:
                            full_mask_path = os.path.join(mask_search_dir, mask_filename)
                            if full_mask_path not in video_to_masks_map[video_id]["mask_paths"]:
                                video_to_masks_map[video_id]["mask_paths"].append(full_mask_path)
    
    all_samples_info_list = [v_data for v_data in video_to_masks_map.values() if v_data.get("mask_paths")]
    if not all_samples_info_list:
        print("\nError: No samples found. Exiting.")
        return

    # Step 2: Process each video and save ROI sequence
    print("\nStep 2: Processing videos to extract ROI sequences...")
    all_metadata = []
    failed_samples = []
    final_sequence_length, final_num_channels = 0, 0

    for sample_info in tqdm(all_samples_info_list, desc="Extracting Sequences"):
        try:
            mat_filepath = sample_info['mat_filepath']
            mat_data = scipy.io.loadmat(mat_filepath)
            frames = mat_data.get(getattr(config, 'MAT_FRAMES_KEY', 'TempFrames')).astype(np.float64)
            H, W, T = frames.shape
            end_frame = min(T, FOCUS_DURATION_FRAMES)
            if end_frame < NUM_FRAMES_PER_SAMPLE:
                raise ValueError(f"Video too short ({T} frames)")
            frame_indices = np.linspace(0, end_frame - 1, NUM_FRAMES_PER_SAMPLE, dtype=int)
            selected_frames = frames[:, :, frame_indices]
            is_two_hole_video = sample_info['source_dataset'] == 'brick_cladding_two_holes'
            masks_to_process = []

            if is_two_hole_video:
                for mask_path in sample_info['mask_paths']:
                    if mask_path.endswith('.npy'): masks_to_process.append(np.load(mask_path).astype(bool))
            else:
                combined_mask = np.zeros((H, W), dtype=bool)
                for mask_path in sample_info['mask_paths']: combined_mask = np.logical_or(combined_mask, np.load(mask_path))
                if np.sum(combined_mask) > 0: masks_to_process.append(combined_mask)
            if not masks_to_process: raise ValueError("No valid masks found")

            for i, individual_mask in enumerate(masks_to_process):
                if np.sum(individual_mask) == 0: continue
                original_video_id = sample_info['video_id']
                new_video_id = f"{original_video_id}_hole_{i}" if is_two_hole_video else original_video_id
                x, y, w, h = cv2.boundingRect(individual_mask.astype(np.uint8))
                pad_w, pad_h = int(w*ROI_PADDING_PERCENT/2), int(h*ROI_PADDING_PERCENT/2)
                x1, y1 = max(0, x-pad_w), max(0, y-pad_h)
                x2, y2 = min(W, x+w+pad_w), min(H, y+h+pad_h)
                cropped_rois = selected_frames[y1:y2, x1:x2, :]

                # --- change based on dataset type---
                processed_frames_list = []
                if dataset_type == 'thermal':
                    for frame_idx in range(NUM_FRAMES_PER_SAMPLE):
                        frame_patch = cropped_rois[:, :, frame_idx]
                        if frame_patch.size == 0: continue
                        resized_frame = cv2.resize(frame_patch, IMAGE_TARGET_SIZE, interpolation=cv2.INTER_AREA)
                        processed_frames_list.append(resized_frame)
                
                elif dataset_type == 'flow' or dataset_type == 'hybrid':
                    for frame_idx in range(NUM_FRAMES_PER_SAMPLE - 1):
                        prev_roi_resized = cv2.resize(cropped_rois[:, :, frame_idx], IMAGE_TARGET_SIZE, cv2.INTER_AREA)
                        next_roi_resized = cv2.resize(cropped_rois[:, :, frame_idx+1], IMAGE_TARGET_SIZE, cv2.INTER_AREA)
                        
                        prev_norm = cv2.normalize(prev_roi_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        next_norm = cv2.normalize(next_roi_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        
                        flow = cv2.calcOpticalFlowFarneback(prev_norm, next_norm, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                        if dataset_type == 'flow':
                            processed_frames_list.append(flow) # flow is already (H, W, 2)
                        elif dataset_type == 'hybrid':
                            min_val, max_val = prev_roi_resized.min(), prev_roi_resized.max()
                            thermal_channel = (prev_roi_resized - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(prev_roi_resized)
                            hybrid_frame = np.dstack([thermal_channel, flow])
                            processed_frames_list.append(hybrid_frame)

                if not processed_frames_list: raise ValueError("No frames were processed.")


                final_sequence_array = np.stack(processed_frames_list, axis=0)
                final_sequence_length, final_num_channels = final_sequence_array.shape[0], final_sequence_array.shape[-1] if final_sequence_array.ndim == 4 else 1

                handcrafted_features = feature_engineering.calculate_hotspot_features(frames, individual_mask, TRUE_FPS, int(5 * TRUE_FPS), 0)
                
                mat_dir = os.path.dirname(mat_filepath)
                relative_structure_path = os.path.relpath(mat_dir, config.DATASET_PARENT_DIR)
                output_dir_for_sample = os.path.join(CNN_DATASET_DIR, relative_structure_path)
                os.makedirs(output_dir_for_sample, exist_ok=True)
                
                image_filename = f"{new_video_id}.npy"
                image_save_path_full = os.path.join(output_dir_for_sample, image_filename)
                np.save(image_save_path_full, final_sequence_array)

                image_path_for_metadata = os.path.join(relative_structure_path, image_filename).replace(os.sep, '/')
                current_metadata = {
                    'video_id': new_video_id, 'image_path': image_path_for_metadata,
                    'airflow_rate': sample_info['airflow_rate'], 'delta_T': sample_info['delta_T'],
                    'material': sample_info['material']
                }
                if handcrafted_features: current_metadata.update(handcrafted_features)
                all_metadata.append(current_metadata)

        except Exception as e:
            failed_samples.append({'video_id': sample_info.get('video_id', 'Unknown'), 'error': str(e)})

    # Step 3: Create and save the final metadata file (This logic is identical for all types)
    print("\nStep 3: Generating final metadata.csv file...")
    if not all_metadata:
        print("Error: No metadata was generated.")
        return
    df_metadata_raw = pd.DataFrame(all_metadata)
    final_df = df_metadata_raw[['video_id', 'image_path', 'airflow_rate', 'material']].copy()
    context_feature_cols, dynamic_feature_cols = [], []
    final_df['delta_T_log'] = np.log1p(df_metadata_raw['delta_T'].astype(float).clip(lower=0))
    context_feature_cols.append('delta_T_log')
    if 'hotspot_area' in SELECTED_RAW_FEATURES_TO_EXTRACT and 'hotspot_area' in df_metadata_raw.columns:
        if LOG_TRANSFORM_AREA:
            final_df['hotspot_area_log'] = np.log1p(df_metadata_raw['hotspot_area'].astype(float).clip(lower=0))
            dynamic_feature_cols.append('hotspot_area_log')
        else:
            final_df['hotspot_area'] = df_metadata_raw['hotspot_area']
            dynamic_feature_cols.append('hotspot_area')
    if 'hotspot_avg_temp_change_rate_initial' in SELECTED_RAW_FEATURES_TO_EXTRACT and 'hotspot_avg_temp_change_rate_initial' in df_metadata_raw.columns:
        if NORMALIZE_AVG_RATE_INITIAL:
            final_df['hotspot_avg_temp_change_rate_initial_norm'] = df_metadata_raw.apply(lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_rate_initial']) else np.nan, axis=1)
            dynamic_feature_cols.append('hotspot_avg_temp_change_rate_initial_norm')
        else:
            final_df['hotspot_avg_temp_change_rate_initial'] = df_metadata_raw['hotspot_avg_temp_change_rate_initial']
            dynamic_feature_cols.append('hotspot_avg_temp_change_rate_initial')
    special_features = ['hotspot_area', 'hotspot_avg_temp_change_rate_initial']
    for f_name in SELECTED_RAW_FEATURES_TO_EXTRACT:
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
    
    # Step 4: Save dataset parameters for easy reference
    save_dataset_parameters(
        output_dir=CNN_DATASET_DIR, dataset_type_str=DATASET_TYPE_STR,
        num_samples=len(final_df), final_df=final_df,
        context_cols=context_feature_cols, dynamic_cols=dynamic_feature_cols,
        seq_len=final_sequence_length, num_ch=final_num_channels
    )

    if failed_samples:
        print(f"\nWarning: {len(failed_samples)} samples failed during processing.")
        for failed in failed_samples: print(f"  - ID: {failed['video_id']}, Reason: {failed['error']}")
    
    print(f"\n--- Dataset Creation Complete: {DATASET_TYPE_STR} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset for CNN-based models.")
    parser.add_argument("--type", type=str, required=True, choices=['thermal', 'flow', 'hybrid'],
                        help="The type of dataset to generate.")
    args = parser.parse_args()
    create_dataset(args.type)

"""
python -m src_cnn.create_dataset --type flow
"""