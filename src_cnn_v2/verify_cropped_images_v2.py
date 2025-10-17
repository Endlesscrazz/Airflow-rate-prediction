# src_cnn_v2/verify_crops_v2.py
"""
A utility script to visually verify the output of create_dataset_v2.py.

It randomly selects a few samples from a metadata file and generates
side-by-side plots showing the original full frame with the crop location
and the actual cropped frame.
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

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import V2 Config and dataset creation helpers
from src_cnn_v2 import config_v2 as cfg
from src_cnn_v2.create_cnn_dataset_v2 import get_mask_centroid # Re-use the centroid function

def main():
    parser = argparse.ArgumentParser(description="Visually verify the cropped dataset.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples to visualize.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Which split's metadata to use for verification (train will show original, not augmented).")
    args = parser.parse_args()

    print(f"--- Verifying V2 Cropped Dataset ---")
    print(f"Will visualize {args.num_samples} random samples from the '{args.split}' split.")

    # --- Setup Paths ---
    CNN_DATASET_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.V2_DATASET_PARAMS["OUTPUT_SUBDIR"])
    METADATA_PATH = os.path.join(CNN_DATASET_DIR, f"{args.split}_metadata_v2.csv")
    
    # Create a directory for the verification plots
    VERIFICATION_DIR = os.path.join(cfg.OUTPUT_DIR, "crop_verification")
    os.makedirs(VERIFICATION_DIR, exist_ok=True)
    print(f"Saving output plots to: {VERIFICATION_DIR}")

    try:
        df_meta = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        sys.exit(f"FATAL: Metadata file not found at '{METADATA_PATH}'. Please run create_dataset_v2.py first.")

    # Filter for original samples only, in case 'train' split is chosen
    df_meta_orig = df_meta[df_meta['sample_id'].str.contains('_orig')].copy()
    
    # Select random samples
    num_to_sample = min(args.num_samples, len(df_meta_orig))
    random_samples = df_meta_orig.sample(n=num_to_sample, random_state=cfg.RANDOM_STATE)

    for index, row in random_samples.iterrows():
        try:
            print(f"\nProcessing sample: {row['sample_id']}...")
            
            # --- Extract info from original sample ID ---
            # e.g., "T1.4V..._1_orig" -> "T1.4V..._1"
            original_sample_id = row['sample_id'].replace('_orig', '')
            parts = original_sample_id.rsplit('_', 1)
            video_id = parts[0]
            hole_id = parts[1]

            # --- 1. Find and load original raw files ---
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
            if mask_dir_results:
                for path in mask_dir_results:
                    if os.path.isdir(path): mask_dir_path = path; break
            
            if not mask_dir_path: raise FileNotFoundError(f"Mask directory not found for video_id '{video_id}'")

            path1 = os.path.join(mask_dir_path, f"{video_id}_mask_{hole_id}.npy")
            path2 = os.path.join(mask_dir_path, f"{video_id}_mask_0.npy")
            individual_mask_path = path1 if os.path.exists(path1) else (path2 if hole_id == '1' and os.path.exists(path2) else None)
            if not individual_mask_path: raise FileNotFoundError("Mask file not found.")

            full_frames = scipy.io.loadmat(mat_filepath).get('TempFrames').astype(np.float32)
            mask = np.load(individual_mask_path)
            
            # --- 2. Load the cropped sequence ---
            cropped_sequence_path = os.path.join(CNN_DATASET_DIR, row['image_path'])
            cropped_sequence = np.load(cropped_sequence_path) # Shape (T, H, W)
            
            # --- 3. Get the middle frame for visualization ---
            mid_frame_idx = full_frames.shape[2] // 2
            full_frame_mid = full_frames[:, :, mid_frame_idx]
            
            cropped_mid_idx = cropped_sequence.shape[0] // 2
            cropped_frame_mid = cropped_sequence[cropped_mid_idx, :, :]
            
            # --- 4. Calculate centroid and bounding box ---
            center_x, center_y = get_mask_centroid(mask)
            if center_x is None: raise ValueError("Cannot get centroid from mask.")
                
            crop_size = cfg.V2_DATASET_PARAMS["CROP_SIZE"]
            half_crop = crop_size // 2
            
            x_start = max(0, center_x - half_crop)
            y_start = max(0, center_y - half_crop)
            
            # --- 5. Create the plot ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Left plot: Full frame with bounding box
            ax1.imshow(full_frame_mid, cmap='inferno')
            ax1.set_title(f'Original Frame (Centroid: {center_x}, {center_y})')
            rect = patches.Rectangle((x_start, y_start), crop_size, crop_size, linewidth=1.5, edgecolor='lime', facecolor='none')
            ax1.add_patch(rect)
            ax1.axis('off')

            # Right plot: Cropped frame
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

# python src_cnn_v2/verify_cropped_images_v2.py --num_samples 20