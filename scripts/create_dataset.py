# scripts/create_dataset.py
"""
Prepares a dataset for CNN-based models.
This single, configurable script can generate one of three dataset types.
All configurations are imported from src_cnn.config.

Run using:
python -m scripts.create_dataset --type [thermal|flow|hybrid]
"""
from src_cnn import feature_engineering
from src_cnn import data_utils
from src_cnn import config as cfg
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


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


def create_dataset(dataset_type):
    """Main function to generate the specified dataset."""
    FOCUS_DURATION_FRAMES = int(cfg.FOCUS_DURATION_SECONDS * cfg.TRUE_FPS)

    if dataset_type == 'thermal':
        output_dir = os.path.join(cfg.PROCESSED_DATASET_DIR, "dataset_1ch_thermal_hard_crop")
        DATASET_TYPE_STR = "1-Channel Thermal (Hard Crop)"
    elif dataset_type == 'thermal_masked':
        output_dir = os.path.join(cfg.PROCESSED_DATASET_DIR, "dataset_2ch_thermal_masked")
        DATASET_TYPE_STR = "2-Channel Thermal + Mask"
    elif dataset_type == 'flow':
        output_dir = os.path.join(cfg.PROCESSED_DATASET_DIR, "dataset_2ch_flow_hard_crop")
        DATASET_TYPE_STR = "2-Channel Optical Flow"
    elif dataset_type == 'hybrid':
        output_dir = os.path.join(cfg.PROCESSED_DATASET_DIR, "dataset_3ch_hybrid_hard_crop")
        DATASET_TYPE_STR = "3-Channel Hybrid"
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    METADATA_SAVE_PATH = os.path.join(output_dir, "metadata.csv")
    
    print(f"--- Starting Dataset Creation: {DATASET_TYPE_STR} ---")
    
    # 1. Load Ground Truth Airflow Data
    try:
        airflow_map = data_utils.load_airflow_from_csv(cfg.GROUND_TRUTH_CSV_PATH)
        print(f"Loaded {len(airflow_map)} ground truth entries from {cfg.GROUND_TRUTH_CSV_PATH}")
    except Exception as e:
        print(f"Could not load or parse ground truth CSV. Aborting. Error: {e}")
        return

    # 2. Scan for video and mask files
    print("\nScanning for all video and mask files...")
    video_to_masks_map = {}
    for d_key, d_conf in cfg.DATASET_CONFIGS.items():
        dataset_path_load = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"])
        if not os.path.isdir(dataset_path_load):
            print(f"  -> WARNING: Data path not found for '{d_key}'. Skipping.")
            continue
        
        for root_load, _, files_load in os.walk(dataset_path_load):
            for mat_filename_load in fnmatch.filter(files_load, '*.mat'):
                # Link video to ground truth via voltage parsed from filename
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
                    except Exception as e:
                        print(f"Error processing metadata for {mat_filename_load}: {e}")
                        continue
                
                # New mask finding logic
                mask_search_dir = os.path.join(cfg.RAW_MASK_PARENT_DIR, d_conf["dataset_subfolder"], video_id)
                if os.path.isdir(mask_search_dir):
                    mask_files = fnmatch.filter(os.listdir(mask_search_dir), "*.npy")
                    if mask_files:
                        video_to_masks_map[video_id]["mask_paths"].extend([os.path.join(mask_search_dir, mf) for mf in mask_files])
    
    all_samples_info_list = [v for v in video_to_masks_map.values() if v.get("mask_paths")]
    if not all_samples_info_list:
        print("\nError: No samples found. Exiting.")
        return

    print("\nStep 2: Processing videos to extract ROI sequences...")
    all_metadata, failed_samples = [], []
    final_sequence_length, final_num_channels = 0, 0

    for sample_info in tqdm(all_samples_info_list, desc="Extracting Sequences"):
        try:
            mat_filepath = sample_info['mat_filepath']
            mat_data = scipy.io.loadmat(mat_filepath)
            # Use a default key 'TempFrames' if MAT_FRAMES_KEY is not in the old config
            frames = mat_data.get('TempFrames').astype(np.float64)
            H, W, T = frames.shape
            end_frame = min(T, FOCUS_DURATION_FRAMES)

            if end_frame < cfg.NUM_FRAMES_PER_SAMPLE:
                raise ValueError(f"Video too short ({T} frames)")

            frame_indices = np.linspace(
                0, end_frame - 1, cfg.NUM_FRAMES_PER_SAMPLE, dtype=int)
            selected_frames = frames[:, :, frame_indices]
            is_two_hole_video = sample_info['source_dataset'] == 'brick_cladding_two_holes'
            masks_to_process = []

            if is_two_hole_video:
                for mask_path in sample_info['mask_paths']:
                    if mask_path.endswith('.npy'):
                        masks_to_process.append(
                            np.load(mask_path).astype(bool))
            else:
                combined_mask = np.zeros((H, W), dtype=bool)
                for mask_path in sample_info['mask_paths']:
                    combined_mask = np.logical_or(
                        combined_mask, np.load(mask_path))
                if np.sum(combined_mask) > 0:
                    masks_to_process.append(combined_mask)
            if not masks_to_process:
                raise ValueError("No valid masks found")

            for i, individual_mask in enumerate(masks_to_process):
                if not np.any(individual_mask): continue

                original_video_id = sample_info['video_id']
                new_video_id = f"{original_video_id}_hole_{i}" if is_two_hole_video else original_video_id
                
                processed_frames_list = []
                
                
                if dataset_type == 'thermal_masked':
                    # NEW LOGIC: Use full frame and add mask as a channel
                    for frame_idx in range(cfg.NUM_FRAMES_PER_SAMPLE):
                        full_frame = selected_frames[:, :, frame_idx]
                        frame_resized = cv2.resize(full_frame, cfg.IMAGE_TARGET_SIZE, interpolation=cv2.INTER_AREA)
                        mask_resized = cv2.resize(individual_mask.astype(np.float32), cfg.IMAGE_TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                        stacked_frame = np.stack([frame_resized, mask_resized], axis=-1)
                        processed_frames_list.append(stacked_frame)
                else:
                    # OLD LOGIC: Hard crop to the bounding box
                    x, y, w, h = cv2.boundingRect(individual_mask.astype(np.uint8))
                    pad_w = int(w * cfg.ROI_PADDING_PERCENT / 2)
                    pad_h = int(h * cfg.ROI_PADDING_PERCENT / 2)
                    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                    x2, y2 = min(W, x + w + pad_w), min(H, y + h + pad_h)
                    cropped_rois = selected_frames[y1:y2, x1:x2, :]

                    if cropped_rois.size == 0:
                        print(f"Warning: Cropped ROI is empty for {new_video_id}. Skipping.")
                        continue

                    if dataset_type == 'thermal':
                        for frame_idx in range(cfg.NUM_FRAMES_PER_SAMPLE):
                            resized = cv2.resize(cropped_rois[:, :, frame_idx], cfg.IMAGE_TARGET_SIZE, interpolation=cv2.INTER_AREA)
                            processed_frames_list.append(resized)

                    elif dataset_type in ['flow', 'hybrid']:
                        for frame_idx in range(cfg.NUM_FRAMES_PER_SAMPLE - 1):
                            prev_roi = cv2.resize(cropped_rois[:, :, frame_idx], cfg.IMAGE_TARGET_SIZE, cv2.INTER_AREA)
                            next_roi = cv2.resize(cropped_rois[:, :, frame_idx + 1], cfg.IMAGE_TARGET_SIZE, cv2.INTER_AREA)
                            
                            prev_norm = cv2.normalize(prev_roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            next_norm = cv2.normalize(next_roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            
                            flow = cv2.calcOpticalFlowFarneback(prev_norm, next_norm, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                            if dataset_type == 'flow':
                                processed_frames_list.append(flow)
                            elif dataset_type == 'hybrid':
                                min_val, max_val = prev_roi.min(), prev_roi.max()
                                thermal_ch = (prev_roi - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(prev_roi)
                                hybrid_frame = np.dstack([thermal_ch, flow])
                                processed_frames_list.append(hybrid_frame)

                if not processed_frames_list:
                    raise ValueError("No frames were processed.")

                final_sequence_array = np.stack(processed_frames_list, axis=0)
                final_sequence_length = final_sequence_array.shape[0]
                final_num_channels = final_sequence_array.shape[-1] if final_sequence_array.ndim == 4 else 1

                handcrafted_features = feature_engineering.calculate_hotspot_features(
                    frames=frames,
                    hotspot_mask=individual_mask,
                    envir_para=1
                )

                mat_dir = os.path.dirname(mat_filepath)
                relative_structure_path = os.path.relpath(
                    mat_dir, cfg.RAW_DATASET_PARENT_DIR)
                output_dir_for_sample = os.path.join(
                    output_dir, relative_structure_path)
                os.makedirs(output_dir_for_sample, exist_ok=True)

                image_filename = f"{new_video_id}.npy"
                image_save_path_full = os.path.join(
                    output_dir_for_sample, image_filename)
                np.save(image_save_path_full, final_sequence_array)

                image_path_for_metadata = os.path.join(
                    relative_structure_path, image_filename).replace(os.sep, '/')
                current_metadata = {
                    'video_id': new_video_id, 'image_path': image_path_for_metadata,
                    'airflow_rate': sample_info['airflow_rate'], 'delta_T': sample_info['delta_T'],
                    'material': sample_info['material']
                }
                if handcrafted_features:
                    current_metadata.update(handcrafted_features)
                all_metadata.append(current_metadata)

        except Exception as e:
            failed_samples.append({'video_id': sample_info.get(
                'video_id', 'Unknown'), 'error': str(e)})

    print("\nStep 3: Generating final metadata.csv file...")
    if not all_metadata:
        print("Error: No metadata was generated.")
        return

    df_metadata_raw = pd.DataFrame(all_metadata)
    final_df = df_metadata_raw[[
        'video_id', 'image_path', 'airflow_rate', 'material']].copy()
    context_feature_cols, dynamic_feature_cols = [], []
    final_df['delta_T_log'] = np.log1p(
        df_metadata_raw['delta_T'].astype(float).clip(lower=0))
    context_feature_cols.append('delta_T_log')

    if 'hotspot_area' in cfg.HANDCRAFTED_FEATURES_TO_EXTRACT and 'hotspot_area' in df_metadata_raw.columns:
        if cfg.LOG_TRANSFORM_AREA:
            final_df['hotspot_area_log'] = np.log1p(
                df_metadata_raw['hotspot_area'].astype(float).clip(lower=0))
            dynamic_feature_cols.append('hotspot_area_log')
        else:
            final_df['hotspot_area'] = df_metadata_raw['hotspot_area']
            dynamic_feature_cols.append('hotspot_area')

    if 'hotspot_avg_temp_change_rate_initial' in cfg.HANDCRAFTED_FEATURES_TO_EXTRACT and 'hotspot_avg_temp_change_rate_initial' in df_metadata_raw.columns:
        if cfg.NORMALIZE_AVG_RATE_INITIAL:
            final_df['hotspot_avg_temp_change_rate_initial_norm'] = df_metadata_raw.apply(
                lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_rate_initial']) else np.nan, axis=1)
            dynamic_feature_cols.append(
                'hotspot_avg_temp_change_rate_initial_norm')
        else:
            final_df['hotspot_avg_temp_change_rate_initial'] = df_metadata_raw['hotspot_avg_temp_change_rate_initial']
            dynamic_feature_cols.append('hotspot_avg_temp_change_rate_initial')

    special_features = ['hotspot_area', 'hotspot_avg_temp_change_rate_initial']
    for f_name in cfg.HANDCRAFTED_FEATURES_TO_EXTRACT:
        if f_name not in special_features and f_name in df_metadata_raw.columns:
            final_df[f_name] = df_metadata_raw[f_name]
            dynamic_feature_cols.append(f_name)

    material_dummies = pd.get_dummies(
        df_metadata_raw['material'], prefix='material', dtype=int)
    final_df = pd.concat([final_df, material_dummies], axis=1)
    context_feature_cols.extend(material_dummies.columns.tolist())

    all_feature_cols = context_feature_cols + dynamic_feature_cols
    final_df[all_feature_cols] = final_df[all_feature_cols].fillna(
        final_df[all_feature_cols].median())
    final_df.to_csv(METADATA_SAVE_PATH, index=False)
    print(
        f"\nSuccessfully saved metadata for {len(final_df)} samples to: {METADATA_SAVE_PATH}")

    save_dataset_parameters(
        output_dir=output_dir, dataset_type_str=DATASET_TYPE_STR,
        num_samples=len(final_df), final_df=final_df,
        context_cols=context_feature_cols, dynamic_cols=dynamic_feature_cols,
        seq_len=final_sequence_length, num_ch=final_num_channels
    )

    if failed_samples:
        print(
            f"\nWarning: {len(failed_samples)} samples failed during processing.")
        for failed in failed_samples:
            print(f"  - ID: {failed['video_id']}, Reason: {failed['error']}")

    print(f"\n--- Dataset Creation Complete: {DATASET_TYPE_STR} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dataset for CNN-based models.")
    parser.add_argument("--type", type=str, required=True, choices=['thermal', 'flow', 'hybrid','thermal_masked'],
                        help="The type of dataset to generate.")
    args = parser.parse_args()
    create_dataset(args.type)

"""
python -m scripts.create_dataset --type thermal_masked
"""
