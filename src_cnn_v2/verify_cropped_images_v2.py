# src_cnn_v2/verify_crops_v2.py
"""
A utility script to visually verify the output of create_dataset_v2.py.
This version works without a 'source_dataset_key' in the metadata.
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import glob
import argparse

# --- Move sys.path insert to the top ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg
# --- FIX: Correct the import statement if filename was changed ---
from src_cnn_v2.create_cnn_dataset_v2 import get_mask_centroid

def main():
    parser = argparse.ArgumentParser(description="Visually verify the cropped dataset.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples to visualize.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Which split's metadata to use for verification.")
    parser.add_argument("--raw_video_dir", type=str, required=True, 
                        help="Path to the parent directory containing raw .mat video datasets (e.g., /Volumes/One_Touch/...)")
    parser.add_argument("--raw_mask_dir", type=str, required=True,
                        help="Path to the parent directory containing the SAM mask datasets (e.g., output_SAM/datasets).")
    args = parser.parse_args()

    print(f"--- Verifying V2 Cropped Dataset for Experiment: {cfg.EXPERIMENT_NAME} ---")
    print(f"  - Using data split from random seed: {cfg.RANDOM_STATE}")
    print(f"Will visualize {args.num_samples} random samples from the '{args.split}' split.")

    DATASET_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.V2_DATASET_PARAMS["OUTPUT_SUBDIR"])
    
    # Use the correct versioned path from the config
    if args.split == 'train':
        METADATA_PATH = cfg.TRAIN_METADATA_PATH
    elif args.split == 'val':
        METADATA_PATH = cfg.VAL_METADATA_PATH
    else: # test
        METADATA_PATH = cfg.TEST_METADATA_PATH
    
    VERIFICATION_DIR = os.path.join(cfg.OUTPUT_DIR, "crop_verification")
    os.makedirs(VERIFICATION_DIR, exist_ok=True)
    print(f"Saving output plots to: {VERIFICATION_DIR}")

    try:
        df_meta = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        sys.exit(f"FATAL: Metadata file not found at '{METADATA_PATH}'. Please run create_dataset_v2.py with RANDOM_STATE={cfg.RANDOM_STATE} first.")

    # We only need these columns for this script to function
    required_cols = ['sample_id', 'video_id', 'hole_id', 'image_path']
    if not all(col in df_meta.columns for col in required_cols):
        sys.exit(f"FATAL: Metadata file is missing required columns. Expected at least: {required_cols}")

    df_meta_orig = df_meta[df_meta['sample_id'].str.contains('_orig')].copy()
    num_to_sample = min(args.num_samples, len(df_meta_orig))
    if num_to_sample == 0:
        print("No original samples found to verify."); return
        
    random_samples = df_meta_orig.sample(n=num_to_sample, random_state=cfg.RANDOM_STATE)

    for index, row in random_samples.iterrows():
        try:
            print(f"\nProcessing sample: {row['sample_id']}...")
            
            video_id = row['video_id']
            hole_id = str(row['hole_id'])
            numeric_hole_id = hole_id.split('_')[0]

            # --- FIX: Re-introduce the loop to find the source dataset ---
            mat_filepath, found_config_key = (None, None)
            for d_key, d_conf in cfg.DATASET_CONFIGS.items():
                video_search_pattern = os.path.join(args.raw_video_dir, d_conf["dataset_subfolder"], '**', f"{video_id}.mat")
                video_results = glob.glob(video_search_pattern, recursive=True)
                if video_results:
                    mat_filepath, found_config_key = video_results[0], d_key
                    break
            
            if not mat_filepath: raise FileNotFoundError(f".mat file not found for video_id '{video_id}'")
            
            mask_subfolder = cfg.DATASET_CONFIGS[found_config_key]["dataset_subfolder"]
            mask_search_pattern = os.path.join(args.raw_mask_dir, mask_subfolder, '**', video_id)
            mask_dir_results = glob.glob(mask_search_pattern, recursive=True)
            mask_dir_path = next((path for path in mask_dir_results if os.path.isdir(path)), None)
            if not mask_dir_path: raise FileNotFoundError(f"Mask directory not found for video_id '{video_id}'")
            # --- END FIX ---

            path1 = os.path.join(mask_dir_path, f"{video_id}_mask_{numeric_hole_id}.npy")
            path2 = os.path.join(mask_dir_path, f"{video_id}_mask_0.npy")
            individual_mask_path = path1 if os.path.exists(path1) else (path2 if numeric_hole_id == '1' and os.path.exists(path2) else None)
            if not individual_mask_path: raise FileNotFoundError(f"Mask file not found for hole {numeric_hole_id}.")

            full_frames = scipy.io.loadmat(mat_filepath).get('TempFrames').astype(np.float32)
            mask = np.load(individual_mask_path)
            
            cropped_sequence_path = os.path.join(DATASET_DIR, row['image_path'])
            cropped_sequence = np.load(cropped_sequence_path)
            
            # ... (The rest of the script for plotting is unchanged and correct) ...
            mid_frame_idx = full_frames.shape[2] // 2
            full_frame_mid = full_frames[:, :, mid_frame_idx]
            cropped_mid_idx = cropped_sequence.shape[0] // 2
            cropped_frame_mid = cropped_sequence[cropped_mid_idx, :, :]
            center_x, center_y = get_mask_centroid(mask)
            if center_x is None: raise ValueError("Cannot get centroid from mask.")
            crop_size = cfg.V2_DATASET_PARAMS["CROP_SIZE"]
            half_crop = crop_size // 2
            x_start = max(0, center_x - half_crop)
            y_start = max(0, center_y - half_crop)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(full_frame_mid, cmap='inferno')
            ax1.set_title(f'Original Frame (Centroid: {center_x}, {center_y})')
            rect = patches.Rectangle((x_start, y_start), crop_size, crop_size, linewidth=1.5, edgecolor='lime', facecolor='none')
            ax1.add_patch(rect)
            ax1.axis('off')
            ax2.imshow(cropped_frame_mid, cmap='inferno')
            ax2.set_title(f'Cropped Frame ({crop_size}x{crop_size})')
            ax2.axis('off')
            fig.suptitle(f"Verification for: {row['sample_id']}", fontsize=16)
            plt.tight_layout()
            save_path = os.path.join(VERIFICATION_DIR, f"{row['sample_id']}.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"  -> Saved verification plot to {save_path}")

        except Exception as e:
            print(f"  -> FAILED to process sample {row.get('sample_id', 'N/A')}. Reason: {e}")
            continue
            
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    main()

"""
python src_cnn_v2/verify_cropped_images_v2.py \
  --num_samples 20 \
  --split val \
  --raw_video_dir "/Volumes/One_Touch/Airflow-rate-prediction/datasets" \
  --raw_mask_dir "output_SAM/datasets"
"""