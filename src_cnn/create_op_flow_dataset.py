# create_cnn_dataset.py
"""
Prepares the dataset for the CNN+LSTM model.
(Version with Sequence Extraction)

This script performs a one-time process to:
1. Scan all raw video and mask data.
2. For each video:
    a. Load the raw thermal video frames (.mat).
    b. Load and combine all associated segmentation masks (.npy).
    c. Select a fixed-length sequence of frames from the beginning of the video.
    d. Find a single bounding box that covers all hotspots in the combined mask.
    e. Crop a padded Region of Interest (ROI) from EACH frame in the sequence.
    f. Resize each cropped frame to a fixed size (e.g., 128x128).
    g. Stack the processed frames and save as a single .npy file.
3. Generate a single `metadata.csv` file that links each saved sequence to its label.
"""
import os
import datetime
import sys
import pandas as pd
import numpy as np
import scipy.io
import cv2
import fnmatch
from tqdm import tqdm

# --- Import project modules ---
from src_feature_based import config
from src_feature_based import data_utils
from src_feature_based import feature_engineering

# --- Configuration ---
CNN_DATASET_DIR = "cnn_dataset/dataset_cnn_lstm_flow"
METADATA_SAVE_PATH = os.path.join(CNN_DATASET_DIR, "metadata.csv")
DATASET_TYPE = "2-Channel optical flow"

# ROI and Image Processing Parameters
ROI_PADDING_PERCENT = 0.20
TRUE_FPS = 5
FOCUS_DURATION_SECONDS = 10
FOCUS_DURATION_FRAMES = int(FOCUS_DURATION_SECONDS * TRUE_FPS)
NUM_FRAMES_PER_SAMPLE = 25
IMAGE_TARGET_SIZE = (128, 128)

# --- NEW: Feature Engineering Configuration (mirrored from main_nested.py) ---
SELECTED_RAW_FEATURES_TO_EXTRACT = [
    'hotspot_area',
    'hotspot_avg_temp_change_rate_initial',
    'overall_std_deltaT',
    'temp_max_overall_initial',
    'temp_std_avg_initial',
]
LOG_TRANSFORM_AREA = True
NORMALIZE_AVG_RATE_INITIAL = True
# -----------------------------------------------------------------------------

# Data source configurations
DATASET_CONFIGS = {
    "gypsum_single_hole": {"material": "gypsum", "dataset_subfolder": "dataset_gypsum", "mask_subfolder": "dataset_gypsum"},
    "gypsum_single_hole2": {"material": "gypsum", "dataset_subfolder": "dataset_gypsum2", "mask_subfolder": "dataset_gypsum2"},
    "brick_cladding_single_hole": {"material": "brick_cladding", "dataset_subfolder": "dataset_brickcladding", "mask_subfolder": "dataset_brickcladding"},
    "brick_cladding_two_holes": {"material": "brick_cladding", "dataset_subfolder": "dataset_two_holes_brickcladding", "mask_subfolder": "dataset_two_holes_brickcladding"}
}
def save_dataset_parameters(output_dir, dataset_type, num_samples, final_df, context_cols, dynamic_cols):
    """Saves the key parameters of the generated dataset to a text file."""
    
    # Infer image shape from a sample .npy file
    try:
        sample_path = os.path.join(output_dir, final_df['image_path'].iloc[0])
        sample_array = np.load(sample_path)
        # Shape is (Seq, H, W, C) or (Seq, H, W) -> we want C
        if sample_array.ndim == 4:
            num_channels = sample_array.shape[-1]
            image_shape_str = f"{sample_array.shape[1]}x{sample_array.shape[2]}"
            sequence_length = sample_array.shape[0]
        elif sample_array.ndim == 3: # For single-channel thermal data
            num_channels = 1
            image_shape_str = f"{sample_array.shape[1]}x{sample_array.shape[2]}"
            sequence_length = sample_array.shape[0]
    except Exception as e:
        num_channels = "N/A (Error reading sample)"
        image_shape_str = "N/A"
        sequence_length = "N/A"
        print(f"Warning: Could not read sample file to determine shape. Error: {e}")


    params = {
        "Dataset Directory": output_dir,
        "Dataset Type": dataset_type,
        "Generation Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "---": "---",
        "Total Samples": num_samples,
        "Sequence Length": sequence_length,
        "Image Channels": num_channels,
        "Image Target Size": image_shape_str,
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

def create_cnn_dataset():
    """Main function to generate the CNN-ready dataset."""
    print("--- Starting CNN+LSTM Dataset Creation (Sequence Data) ---")

    # Step 0: Setup Output Directories
    os.makedirs(CNN_DATASET_DIR, exist_ok=True)
    print(f"Output will be saved in: {CNN_DATASET_DIR}")

    # Step 1: Discover all video and mask files (with enhanced logging)
    print("\nStep 1: Scanning for all video and mask files...")
    video_to_masks_map = {}
    datasets_to_load_keys = list(DATASET_CONFIGS.keys())
    dataset_sample_counts = {key: 0 for key in datasets_to_load_keys}
    for d_key in datasets_to_load_keys:
        d_config = DATASET_CONFIGS[d_key]
        print(f"\nScanning dataset config: '{d_key}'")
        dataset_path_load = os.path.join(config.DATASET_PARENT_DIR, d_config["dataset_subfolder"])
        mask_root_path_load = os.path.join(config.BASE_MASK_INPUT_DIR, d_config["mask_subfolder"])
        print(f"  - Video Path: {dataset_path_load}")
        print(f"  - Mask Path:  {mask_root_path_load}")
        if not os.path.isdir(dataset_path_load) or not os.path.isdir(mask_root_path_load):
            print(f"  -> WARNING: Path not found. Skipping.")
            continue
        
        videos_found_in_config = 0
        masks_found_for_videos_in_config = 0
        for root_load, _, files_load in os.walk(dataset_path_load):
            for mat_filename_load in fnmatch.filter(files_load, '*.mat'):
                mat_filepath_load = os.path.join(root_load, mat_filename_load)
                video_id = os.path.splitext(mat_filename_load)[0]
                is_new_video = video_id not in video_to_masks_map
                if is_new_video:
                    try:
                        folder_name_load = os.path.basename(os.path.dirname(mat_filepath_load))
                        airflow_load = data_utils.parse_airflow_rate(folder_name_load)
                        delta_t_load = data_utils.parse_delta_T(mat_filename_load)
                        if delta_t_load is None: continue
                        videos_found_in_config += 1
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
                        masks_found_for_videos_in_config += 1
                        for mask_filename in mask_files_found:
                            full_mask_path = os.path.join(mask_search_dir, mask_filename)
                            if full_mask_path not in video_to_masks_map[video_id]["mask_paths"]:
                                video_to_masks_map[video_id]["mask_paths"].append(full_mask_path)
        
        print(f"  -> Found {videos_found_in_config} '.mat' files with valid metadata.")
        print(f"  -> Found masks for {masks_found_for_videos_in_config} of those videos.")

    # Final filtering and summary (unchanged)
    all_samples_info_list = []
    for video_id, v_data in video_to_masks_map.items():
        if v_data.get("mask_paths"):
            all_samples_info_list.append(v_data)
            source_key = v_data["source_dataset"]
            dataset_sample_counts[source_key] += 1
    if not all_samples_info_list:
        print("\nError: No samples found. Exiting.")
        return

    # --- Step 2: Process each video and save ROI sequence ---
    print("\nStep 2: Processing videos to extract ROI sequences...")
    all_metadata = []
    failed_samples = []

    for sample_info in tqdm(all_samples_info_list, desc="Extracting Sequences"):
        try:
            video_id = sample_info['video_id']
            mat_filepath = sample_info['mat_filepath']

            # Load frames
            mat_key = getattr(config, 'MAT_FRAMES_KEY', 'TempFrames')
            mat_data = scipy.io.loadmat(mat_filepath)
            frames = mat_data.get(mat_key, None).astype(np.float64)
            if frames is None: raise ValueError("Frames not found")
            H, W, T = frames.shape

            # Select a sequence of frames from the beginning
            end_frame = min(T, FOCUS_DURATION_FRAMES) # This will now correctly be min(T, 50)
            if end_frame < NUM_FRAMES_PER_SAMPLE:
                raise ValueError(f"Video too short ({T} frames) to extract {NUM_FRAMES_PER_SAMPLE} frames from the first {FOCUS_DURATION_SECONDS} seconds.")

            # Select 25 evenly spaced frames from this CORRECT 10-second window
            frame_indices = np.linspace(0, end_frame - 1, NUM_FRAMES_PER_SAMPLE, dtype=int)
            selected_frames = frames[:, :, frame_indices]

            is_two_hole_video = sample_info['source_dataset'] == 'brick_cladding_two_holes'
            
            masks_to_process = []
            if is_two_hole_video:
                # For two-hole videos, each mask file is a separate sample.
                for mask_path in sample_info['mask_paths']:
                    # Filter out non-mask files if any exist
                    if mask_path.endswith('.npy'):
                        masks_to_process.append(np.load(mask_path).astype(bool))
            else:
                # For single-hole videos, combine all found masks into one.
                combined_mask = np.zeros((H, W), dtype=bool)
                for mask_path in sample_info['mask_paths']:
                    mask = np.load(mask_path)
                    combined_mask = np.logical_or(combined_mask, mask)
                if np.sum(combined_mask) > 0:
                    masks_to_process.append(combined_mask)
            
            if not masks_to_process:
                raise ValueError("No valid masks found to process for this video.")

            # Loop through the one (or more) masks identified for this video
            for i, individual_mask in enumerate(masks_to_process):
                if np.sum(individual_mask) == 0:
                    continue # Skip empty masks

                # Create a unique ID for each new sample
                original_video_id = sample_info['video_id']
                new_video_id = f"{original_video_id}_hole_{i}" if is_two_hole_video else original_video_id

                # Find bounding box for the *individual* mask
                x, y, w, h = cv2.boundingRect(individual_mask.astype(np.uint8))
                
                pad_w = int(w * ROI_PADDING_PERCENT / 2)
                pad_h = int(h * ROI_PADDING_PERCENT / 2)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(W, x + w + pad_w)
                y2 = min(H, y + h + pad_h)

                cropped_rois = selected_frames[y1:y2, x1:x2, :]

                processed_flow_sequence = []
                # Loop N-1 times because flow is calculated between pairs of frames
                for frame_idx in range(NUM_FRAMES_PER_SAMPLE - 1):
                    # Get two consecutive ROIs
                    prev_roi = cropped_rois[:, :, frame_idx]
                    next_roi = cropped_rois[:, :, frame_idx+1]
                    
                    # Resize to the target size BEFORE flow calculation for consistency
                    prev_roi_resized = cv2.resize(prev_roi, IMAGE_TARGET_SIZE, interpolation=cv2.INTER_AREA)
                    next_roi_resized = cv2.resize(next_roi, IMAGE_TARGET_SIZE, interpolation=cv2.INTER_AREA)

                    # Normalize to 0-255 uint8 range for the OpenCV flow function
                    prev_roi_norm = cv2.normalize(prev_roi_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    next_roi_norm = cv2.normalize(next_roi_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    # Calculate dense optical flow using Farneback method
                    # The result 'flow' is a 2-channel image (Height, Width, 2)
                    # where flow[y, x] = [flow_in_x_direction, flow_in_y_direction]
                    flow = cv2.calcOpticalFlowFarneback(
                        prev=prev_roi_norm, 
                        next=next_roi_norm, 
                        flow=None, 
                        pyr_scale=0.5, 
                        levels=3, 
                        winsize=15, 
                        iterations=3, 
                        poly_n=5, 
                        poly_sigma=1.2, 
                        flags=0
                    )
                    processed_flow_sequence.append(flow)

                # We now have a list of (H, W, 2) arrays. Stack them.
                # The resulting shape will be (Sequence_Length-1, Height, Width, Channels)
                # e.g., (24, 128, 128, 2)
                final_flow_array = np.stack(processed_flow_sequence, axis=0)
                
                handcrafted_features = feature_engineering.calculate_hotspot_features(
                    frames=frames,
                    hotspot_mask=individual_mask, # Use the individual mask
                    fps=TRUE_FPS,
                    focus_duration_frames=int(5 * TRUE_FPS),
                    envir_para=0
                )

                #final_sequence_array = np.stack(processed_sequence, axis=0)
                
                mat_dir = os.path.dirname(mat_filepath)
                relative_structure_path = os.path.relpath(mat_dir, config.DATASET_PARENT_DIR)
                output_dir_for_sample = os.path.join(CNN_DATASET_DIR, relative_structure_path)
                os.makedirs(output_dir_for_sample, exist_ok=True)
                
                image_filename = f"{new_video_id}.npy"
                image_save_path_full = os.path.join(output_dir_for_sample, image_filename)
                np.save(image_save_path_full, final_flow_array)

                image_path_for_metadata = os.path.join(relative_structure_path, image_filename).replace(os.sep, '/')
                current_metadata = {
                    'video_id': new_video_id,
                    'image_path': image_path_for_metadata,
                    'airflow_rate': sample_info['airflow_rate'],
                    'delta_T': sample_info['delta_T'],
                    'material': sample_info['material']
                }
                if handcrafted_features:
                    current_metadata.update(handcrafted_features)
                
                all_metadata.append(current_metadata)

        except Exception as e:
            failed_samples.append({'video_id': sample_info.get('video_id', 'Unknown'), 'error': str(e)})

    # --- Step 3: Create and save the final metadata file ---
    print("\nStep 3: Generating final metadata.csv file...")
    if not all_metadata:
        print("Error: No metadata was generated. Could not create metadata.csv.")
        return
        
    df_metadata_raw = pd.DataFrame(all_metadata)
    print("  - Pre-processing all tabular features...")
    
    # Start with the essential columns that don't need processing
    final_df = df_metadata_raw[['video_id', 'image_path', 'airflow_rate', 'material']].copy()
    
    # Create lists to hold the names of our processed feature columns
    context_feature_cols = []
    dynamic_feature_cols = []

    # 1. Process delta_T
    final_df['delta_T_log'] = np.log1p(df_metadata_raw['delta_T'].astype(float).clip(lower=0))
    context_feature_cols.append('delta_T_log')

    # 2. Process selected raw features based on global config
    if 'hotspot_area' in SELECTED_RAW_FEATURES_TO_EXTRACT and 'hotspot_area' in df_metadata_raw.columns:
        if LOG_TRANSFORM_AREA:
            final_df['hotspot_area_log'] = np.log1p(df_metadata_raw['hotspot_area'].astype(float).clip(lower=0))
            dynamic_feature_cols.append('hotspot_area_log')
        else:
            final_df['hotspot_area'] = df_metadata_raw['hotspot_area']
            dynamic_feature_cols.append('hotspot_area')

    if 'hotspot_avg_temp_change_rate_initial' in SELECTED_RAW_FEATURES_TO_EXTRACT and 'hotspot_avg_temp_change_rate_initial' in df_metadata_raw.columns:
        if NORMALIZE_AVG_RATE_INITIAL:
            final_df['hotspot_avg_temp_change_rate_initial_norm'] = df_metadata_raw.apply(
                lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r['delta_T'] != 0 and np.isfinite(r['hotspot_avg_temp_change_rate_initial']) else np.nan,
                axis=1
            )
            dynamic_feature_cols.append('hotspot_avg_temp_change_rate_initial_norm')
        else:
            final_df['hotspot_avg_temp_change_rate_initial'] = df_metadata_raw['hotspot_avg_temp_change_rate_initial']
            dynamic_feature_cols.append('hotspot_avg_temp_change_rate_initial')

    # 3. Add all OTHER selected raw features that don't have special transforms
    special_features = ['hotspot_area', 'hotspot_avg_temp_change_rate_initial']
    for f_name in SELECTED_RAW_FEATURES_TO_EXTRACT:
        if f_name not in special_features and f_name in df_metadata_raw.columns:
            final_df[f_name] = df_metadata_raw[f_name]
            dynamic_feature_cols.append(f_name)
    
    # 4. One-hot encode material
    material_dummies = pd.get_dummies(df_metadata_raw['material'], prefix='material', dtype=int)
    final_df = pd.concat([final_df, material_dummies], axis=1)
    context_feature_cols.extend(material_dummies.columns.tolist())

    # 5. Fill any remaining NaNs in the new feature columns
    all_feature_cols = context_feature_cols + dynamic_feature_cols
    # Use median for imputation as it's more robust to outliers
    final_df[all_feature_cols] = final_df[all_feature_cols].fillna(final_df[all_feature_cols].median())

    # 6. Save the final, clean metadata
    final_df.to_csv(METADATA_SAVE_PATH, index=False)
    
    print(f"\nSuccessfully saved metadata for {len(final_df)} samples to: {METADATA_SAVE_PATH}")
    print(f"Context feature columns created: {sorted(context_feature_cols)}")
    print(f"Dynamic feature columns created: {sorted(dynamic_feature_cols)}")

    if failed_samples:
        print(f"\nWarning: {len(failed_samples)} samples failed during processing.")
        for failed in failed_samples:
            print(f"  - ID: {failed['video_id']}, Reason: {failed['error']}")

    save_dataset_parameters(
        output_dir=CNN_DATASET_DIR,
        dataset_type=DATASET_TYPE,
        num_samples=len(final_df),
        final_df=final_df,
        context_cols=context_feature_cols,
        dynamic_cols=dynamic_feature_cols
    )
    print("\n--- CNN Dataset Creation Complete ---")

if __name__ == "__main__":
    create_cnn_dataset()

# python -m src_cnn.create_op_flow_dataset