# src_cnn_v2/compare_datasets.py
"""
Dataset Comparison and Visualization Tool (Version 4 - Detailed Debugging)

Purpose:
  - Compares .npy files from two pipelines with different naming conventions.
  - Includes detailed error logging to pinpoint corrupted or malformed files.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import glob

def find_colleague_file_strict(my_filename, colleague_root):
    """
    Finds the exact corresponding file in the colleague's dataset using the core video ID.
    """
    parts = my_filename.replace('.npy', '').split('__')
    video_id = parts[0]
    details = parts[1].split('_')
    hole_id = details[0]
    is_augmented = 'aug' in details
    aug_num = details[-1] if is_augmented else None

    spot_name = f"spot{hole_id}"
    suffix = f"noise{aug_num}.npy" if is_augmented else "original.npy"
    
    search_pattern = f"**/*_{video_id}__{spot_name}_{suffix}"
    search_path = os.path.join(colleague_root, search_pattern)
    
    results = glob.glob(search_path, recursive=True)
    
    if results:
        return results[0]
    return None

def main():
    parser = argparse.ArgumentParser(description="Compare two processed datasets.")
    parser.add_argument("--my_dataset_dir", required=True, help="Path to your final processed dataset folder.")
    parser.add_argument("--colleague_dataset_dir", required=True, help="Path to your colleague's 'Final_data' folder.")
    parser.add_argument("--output_dir", required=True, help="Directory to save comparison plots.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to compare.")
    parser.add_argument("--compare_aug", action='store_true', help="Set this flag to compare augmented files instead of originals.")
    args = parser.parse_args()

    print("--- Running Dataset Comparison Tool (Detailed Debugging) ---")
    os.makedirs(args.output_dir, exist_ok=True)

    my_files = [f for f in os.listdir(args.my_dataset_dir) if f.endswith('.npy')]
    
    if args.compare_aug:
        target_files = [f for f in my_files if 'aug' in f]
        print(f"  - Mode: Comparing AUGMENTED files.")
    else:
        target_files = [f for f in my_files if 'orig' in f]
        print(f"  - Mode: Comparing ORIGINAL files.")

    if not target_files:
        sys.exit("FATAL: No target files ('orig' or 'aug') found in your dataset directory.")

    np.random.shuffle(target_files)
    files_to_compare = target_files[:args.num_samples]

    for my_filename in files_to_compare:
        print(f"\n--- Comparing file: {my_filename} ---")
        
        my_filepath = os.path.join(args.my_dataset_dir, my_filename)
        colleague_filepath = find_colleague_file_strict(my_filename, args.colleague_dataset_dir)

        if not colleague_filepath:
            print(f"  - WARNING: Could not find a strict match for this file. Skipping.")
            continue
            
        print(f"  - Your file:      {my_filename}")
        print(f"  - Colleague file: {os.path.basename(colleague_filepath)}")

        # --- START OF DETAILED DEBUGGING BLOCK ---
        my_data, colleague_data = None, None

        # Step 1: Check if files exist and get their sizes
        my_file_exists = os.path.exists(my_filepath)
        colleague_file_exists = os.path.exists(colleague_filepath)
        
        my_file_size = os.path.getsize(my_filepath) if my_file_exists else -1
        colleague_file_size = os.path.getsize(colleague_filepath) if colleague_file_exists else -1

        print(f"  - Your file size:      {my_file_size} bytes")
        print(f"  - Colleague file size: {colleague_file_size} bytes")
        
        # Step 2: Try to load YOUR file
        try:
            if my_file_size <= 0:
                raise ValueError("File is empty or does not exist.")
            # Your data is saved as (T, H, W)
            loaded_my_data = np.load(my_filepath)
            print(f"  - Loaded your file. Original shape: {loaded_my_data.shape}, Total elements: {loaded_my_data.size}")
            # Transpose to (H, W, T) for comparison
            my_data = loaded_my_data.transpose(1, 2, 0)
        except Exception as e:
            print(f"  - FATAL ERROR loading YOUR file ({my_filename}): {e}. Skipping this pair.")
            continue # Skip to the next file pair

        # Step 3: Try to load COLLEAGUE'S file
        try:
            if colleague_file_size <= 0:
                raise ValueError("File is empty or does not exist.")
            # Colleague's data is saved as (H, W, T)
            colleague_data = np.load(colleague_filepath)
            print(f"  - Loaded colleague's file. Shape: {colleague_data.shape}, Total elements: {colleague_data.size}")
        except Exception as e:
            print(f"  - FATAL ERROR loading COLLEAGUE'S file ({os.path.basename(colleague_filepath)}): {e}. Skipping this pair.")
            continue # Skip to the next file pair
        # --- END OF DETAILED DEBUGGING BLOCK ---

        # If we reach here, both files loaded successfully. Now we can visualize.
        fig, axs = plt.subplots(1, 3, figsize=(20, 6), dpi=100)
        fig.suptitle(f"Comparison for {my_filename}", fontsize=16)

        if my_data.shape != colleague_data.shape:
            print(f"  - WARNING: Shape mismatch after loading! Your shape: {my_data.shape}, Colleague's: {colleague_data.shape}")
            plt.close(fig)
            continue
        
        my_avg, colleague_avg = np.mean(my_data, axis=2), np.mean(colleague_data, axis=2)
        diff, mse = my_avg - colleague_avg, mean_squared_error(my_avg, colleague_avg)
        
        vmin = min(my_avg.min(), colleague_avg.min())
        vmax = max(my_avg.max(), colleague_avg.max())

        im1 = axs[0].imshow(my_avg, cmap='inferno', vmin=vmin, vmax=vmax); axs[0].set_title(f"Your Data"); fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
        im2 = axs[1].imshow(colleague_avg, cmap='inferno', vmin=vmin, vmax=vmax); axs[1].set_title(f"Colleague's Data"); fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
        im3 = axs[2].imshow(diff, cmap='coolwarm'); axs[2].set_title(f"Difference Image\nMSE: {mse:.6f}"); fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)

        for ax in axs: ax.axis('off')
        
        save_path = os.path.join(args.output_dir, f"comparison_{os.path.splitext(my_filename)[0]}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path)
        plt.close(fig)
        print(f"  - Saved comparison plot to: {save_path}")

if __name__ == "__main__":
    main()


"""

python src_cnn_v2/compare_datasets.py \
  --my_dataset_dir "/Users/shreyas/Downloads/hardyboard_all_dataset_v2/dataset_cs15_nf150_aug100" \
  --colleague_dataset_dir "/Users/shreyas/Downloads/Final_data" \
  --output_dir "dataset_comparison_output_orig" \
  --num_samples 5
"""