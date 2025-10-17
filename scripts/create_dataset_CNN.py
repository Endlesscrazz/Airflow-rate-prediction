# scripts/create_dataset_CNN.py
"""
Prepares a dataset for CNN-based models. This script processes video sequences
and merges them with pre-calculated, selected handcrafted features from a
master feature file.
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

from src_cnn import config as cfg

def save_dataset_parameters(output_dir, dataset_type_str, num_samples, context_cols, dynamic_cols, seq_len, num_ch):
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
        "Context Features Used": sorted(context_cols),
        "Dynamic Features Used": sorted(dynamic_cols),
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

def prepare_tabular_features(df_in):
    """Applies all required transformations to the tabular feature data."""
    df = df_in.copy()
    
    # Apply log transforms
    if 'delta_T' in df.columns:
        df['delta_T_log'] = np.log1p(df['delta_T'])
    if cfg.LOG_TRANSFORM_AREA and 'hotspot_area' in df.columns:
        df['hotspot_area_log'] = np.log1p(df['hotspot_area'])
    
    # Apply normalization by delta_T
    if cfg.NORMALIZE_AVG_RATE_INITIAL and 'hotspot_avg_temp_change_rate_initial' in df.columns:
        df['hotspot_avg_temp_change_rate_initial_norm'] = df.apply(
            lambda r: r['hotspot_avg_temp_change_rate_initial'] / r['delta_T'] if r.get('delta_T', 0) != 0 else 0, axis=1)
    
    if cfg.NORMALIZE_CUMULATIVE_FEATURES:
        features_to_normalize = ['cumulative_raw_delta_sum', 'cumulative_abs_delta_sum', 'auc_mean_temp_delta', 'mean_pixel_volatility']
        for feature in features_to_normalize:
            if feature in df.columns:
                df[f"{feature}_norm"] = df.apply(
                    lambda r: r[feature] / r['delta_T'] if r.get('delta_T', 0) != 0 and pd.notna(r.get(feature)) else np.nan, axis=1)

    # Create dummy variables for material
    if 'material' in df.columns:
        df = pd.concat([df, pd.get_dummies(df['material'], prefix='material', dtype=int)], axis=1)
        
    return df

def create_dataset(dataset_type):
    """Main function to generate the specified dataset."""
    FOCUS_DURATION_FRAMES = int(cfg.FOCUS_DURATION_SECONDS * cfg.TRUE_FPS)

    type_to_folder = {'thermal': "dataset_1ch_thermal", 'thermal_masked': "dataset_2ch_thermal_masked", 'flow': "dataset_2ch_flow", 'hybrid': "dataset_3ch_hybrid"}
    type_to_str = {'thermal': "1-Channel Thermal", 'thermal_masked': "2-Channel Thermal + Mask", 'flow': "2-Channel Optical Flow", 'hybrid': "3-Channel Hybrid"}
    
    output_subdir_name = type_to_folder.get(dataset_type)
    output_dir = os.path.join(cfg.OUTPUT_DIR, output_subdir_name)
    DATASET_TYPE_STR = type_to_str.get(dataset_type)
    METADATA_SAVE_PATH = os.path.join(output_dir, "metadata.csv")
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Dataset Creation: {DATASET_TYPE_STR} ---")
    print(f"Output directory: {output_dir}")
    
    final_sequence_length = cfg.NUM_FRAMES_PER_SAMPLE
    if dataset_type in ['flow', 'hybrid']: final_sequence_length -= 1
    final_num_channels = cfg.IMAGE_CHANNELS_BY_TYPE.get(dataset_type, 0)
    
    try:
        df_features = pd.read_csv(cfg.MASTER_FEATURES_PATH)
        print(f"Loaded {len(df_features)} samples from master feature file: {cfg.MASTER_FEATURES_PATH}")
    except Exception as e:
        sys.exit(f"FATAL: Could not load master feature file. Please run generate_master_features.py first. Error: {e}")

    all_metadata_rows = []
    failed_samples = []
    
    for index, feature_row in tqdm(df_features.iterrows(), total=len(df_features), desc="Processing Samples"):
        try:
            sample_id = feature_row['sample_id']
            video_id = feature_row['video_id']
            hole_id = str(feature_row['hole_id'])
            
            mat_filepath = None
            mask_dir_path = None
            found_config_key = None

            # --- ROBUST FILE FINDING LOGIC (HARMONIZED) ---
            for d_key, d_conf in cfg.DATASET_CONFIGS.items():
                video_search_pattern = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"], '**', f"{video_id}.mat")
                video_results = glob.glob(video_search_pattern, recursive=True)
                if video_results:
                    mat_filepath = video_results[0]
                    found_config_key = d_key
                    break 
            
            if not mat_filepath:
                raise FileNotFoundError(f"Raw data files not found for video_id '{video_id}'")

            mask_subfolder = cfg.DATASET_CONFIGS[found_config_key]["dataset_subfolder"]
            mask_search_pattern = os.path.join(cfg.RAW_MASK_PARENT_DIR, mask_subfolder, '**', video_id)
            mask_dir_results = glob.glob(mask_search_pattern, recursive=True)
            
            if mask_dir_results:
                for path in mask_dir_results:
                    if os.path.isdir(path):
                        mask_dir_path = path
                        break

            if not mask_dir_path:
                raise FileNotFoundError(f"Could not find a mask DIRECTORY for video_id '{video_id}'")

            individual_mask_path = os.path.join(mask_dir_path, f"{video_id}_mask_{hole_id}.npy")
            if not os.path.exists(individual_mask_path):
                raise FileNotFoundError(f"Mask file does not exist at expected path: {individual_mask_path}")
            
            # --- END OF FILE FINDING LOGIC ---

            individual_mask = np.load(individual_mask_path).astype(bool)
            if not np.any(individual_mask):
                continue

            mat_data = scipy.io.loadmat(mat_filepath)
            frames = mat_data.get('TempFrames').astype(np.float64)
            H, W, T = frames.shape
            end_frame = min(T, FOCUS_DURATION_FRAMES)
            if end_frame < cfg.NUM_FRAMES_PER_SAMPLE:
                raise ValueError(f"Video too short ({T} frames)")

            frame_indices = np.linspace(0, end_frame - 1, cfg.NUM_FRAMES_PER_SAMPLE, dtype=int)
            selected_frames = frames[:, :, frame_indices]

            processed_frames_list = []
            if dataset_type == 'thermal_masked':
                for i in range(cfg.NUM_FRAMES_PER_SAMPLE):
                    frame_resized = cv2.resize(selected_frames[:, :, i], cfg.IMAGE_TARGET_SIZE, cv2.INTER_AREA)
                    mask_resized = cv2.resize(individual_mask.astype(np.float32), cfg.IMAGE_TARGET_SIZE, cv2.INTER_NEAREST)
                    processed_frames_list.append(np.stack([frame_resized, mask_resized], axis=-1))
            else:
                # Add logic for other dataset types here if needed
                pass

            if not processed_frames_list: continue

            final_sequence_array = np.stack(processed_frames_list, axis=0)
            image_filename = f"{sample_id}.npy"
            np.save(os.path.join(output_dir, image_filename), final_sequence_array)

            current_metadata = feature_row.to_dict()
            current_metadata['image_path'] = image_filename
            all_metadata_rows.append(current_metadata)

        except Exception as e:
            failed_samples.append({'sample_id': feature_row.get('sample_id', 'N/A'), 'error': str(e)})
            continue

    if not all_metadata_rows:
        print("\nFATAL: No metadata was generated. Check for errors in the processing loop.")
        return

    df_full_raw = pd.DataFrame(all_metadata_rows)
    
    # Apply all tabular feature transformations
    df_full_transformed = prepare_tabular_features(df_full_raw)
    
    # Define the final columns needed for the metadata file
    base_cols = ['sample_id', 'video_id', 'image_path', 'airflow_rate']
    context_cols = cfg.CONTEXT_FEATURES + [f"material_{m}" for m in cfg.ALL_POSSIBLE_MATERIALS]
    dynamic_cols = cfg.DYNAMIC_FEATURES
    
    final_context_cols = [c for c in context_cols if c in df_full_transformed.columns]
    final_dynamic_cols = [c for c in dynamic_cols if c in df_full_transformed.columns]
    final_cols_to_keep = base_cols + final_context_cols + final_dynamic_cols
    
    final_df = df_full_transformed[final_cols_to_keep].copy()

    # Impute any remaining NaNs (e.g., from normalization division by zero)
    feature_cols = final_context_cols + final_dynamic_cols
    final_df[feature_cols] = final_df[feature_cols].fillna(final_df[feature_cols].median())
    
    final_df.to_csv(METADATA_SAVE_PATH, index=False)
    
    print(f"\nSuccessfully saved metadata for {len(final_df)} samples to: {METADATA_SAVE_PATH}")
    
    save_dataset_parameters(
        output_dir=output_dir, dataset_type_str=DATASET_TYPE_STR, num_samples=len(final_df),
        context_cols=final_context_cols, dynamic_cols=final_dynamic_cols,
        seq_len=final_sequence_length, num_ch=final_num_channels
    )

    if failed_samples:
        print(f"\nWarning: {len(failed_samples)} samples failed during processing.")
        for i, failed in enumerate(failed_samples[:5]):
            print(f"  - Sample {i+1}: ID='{failed['sample_id']}', Reason: {failed['error']}")

    print(f"\n--- Dataset Creation Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset for CNN-based models.")
    parser.add_argument("--type", type=str, required=True, choices=['thermal', 'flow', 'hybrid', 'thermal_masked'],
                        help="The type of dataset to generate.")
    args = parser.parse_args()
    create_dataset(args.type)
"""
python -m scripts.create_dataset_CNN --type thermal_masked
"""
