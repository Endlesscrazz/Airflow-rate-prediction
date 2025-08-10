# scripts/vis_dataset.py
"""
A standalone utility to visualize the generated dataset sequences for sanity checking.
This script can handle thermal, optical flow, hybrid, and thermal+mask datasets.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import argparse
import cv2
import traceback

# --- Import project modules ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn import config as cfg

def visualize_dataset(dataset_dir, max_samples, output_dir_name):
    """Main function to load sequences and save them as animated GIFs."""
    METADATA_PATH = os.path.join(dataset_dir, "metadata.csv")
    PARAMS_PATH = os.path.join(dataset_dir, "dataset_parameters.txt")
    VISUALIZATION_OUTPUT_DIR = os.path.join("visualizations", output_dir_name)

    print("--- Dataset Visualization Tool ---")

    if not os.path.isfile(METADATA_PATH):
        print(f"FATAL ERROR: Metadata file not found at '{METADATA_PATH}'", file=sys.stderr)
        sys.exit(1)
    
    # --- Determine dataset type and channels ---
    num_channels = None
    dataset_type_str = 'unknown'
    if os.path.isfile(PARAMS_PATH):
        with open(PARAMS_PATH, 'r') as f:
            for line in f:
                if "Image Channels" in line:
                    num_channels = int(line.split(":")[1].strip())
                if "Dataset Type" in line:
                    dataset_type_str = line.split(":")[1].strip()
    
    if num_channels is None:
        print(f"WARNING: Could not read info from {PARAMS_PATH}. Inferring from sample file.", file=sys.stderr)
        df_temp = pd.read_csv(METADATA_PATH)
        sample_path = os.path.join(dataset_dir, df_temp['image_path'].iloc[0])
        sample_array = np.load(sample_path)
        num_channels = sample_array.shape[-1] if sample_array.ndim == 4 else 1

    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    print(f"Dataset to visualize: '{dataset_dir}' (Type: {dataset_type_str}, Channels: {num_channels})")
    print(f"Visualization GIFs will be saved to: '{VISUALIZATION_OUTPUT_DIR}'")

    df_metadata = pd.read_csv(METADATA_PATH)
    samples_to_process = df_metadata
    if max_samples is not None and max_samples < len(df_metadata):
        samples_to_process = df_metadata.sample(n=max_samples, random_state=cfg.RANDOM_STATE)
        print(f"Processing a random subset of {max_samples} samples.")
    
    for _, sample_row in tqdm(samples_to_process.iterrows(), total=len(samples_to_process), desc="Generating GIFs"):
        video_id = sample_row['video_id']
        
        try:
            sequence_path = os.path.join(dataset_dir, sample_row['image_path'])
            sequence_array = np.load(sequence_path)
            
            frames_for_gif = []
            
            for i in range(sequence_array.shape[0]):
                
                # ... (Plotting logic for different channels is correct and unchanged) ...
                if num_channels == 1:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.imshow(sequence_array[i, :, :], cmap='hot'); ax.set_title("Thermal Channel")
                elif num_channels == 2:
                    if 'Mask' in dataset_type_str:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        axes[0].imshow(sequence_array[i, :, :, 0], cmap='hot'); axes[0].set_title("Ch 0: Thermal")
                        axes[1].imshow(sequence_array[i, :, :, 1], cmap='gray', vmin=0, vmax=1); axes[1].set_title("Ch 1: Mask")
                    else:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        flow = sequence_array[i, :, :, :]
                        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
                        hsv[..., 0] = angle * 180 / np.pi / 2
                        hsv[..., 1] = 255
                        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                        ax.imshow(rgb_flow); ax.set_title("Optical Flow (Colorized)")
                elif num_channels == 3:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    hybrid_frame = sequence_array[i, :, :, :]
                    axes[0].imshow(hybrid_frame[:, :, 0], cmap='hot'); axes[0].set_title("Ch 0: Thermal")
                    axes[1].imshow(hybrid_frame[:, :, 1], cmap='RdBu'); axes[1].set_title("Ch 1: Flow-X")
                    axes[2].imshow(hybrid_frame[:, :, 2], cmap='RdBu'); axes[2].set_title("Ch 2: Flow-Y")
                else:
                    raise ValueError(f"Unsupported number of channels: {num_channels}")

                main_title = (f"ID: {video_id} | Frame: {i+1}/{sequence_array.shape[0]}\n"
                              f"Airflow: {sample_row['airflow_rate']:.2f}, Material: {sample_row['material']}")
                fig.suptitle(main_title, fontsize=12)
                
                for axis in fig.get_axes(): axis.axis('off')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # --- START OF BUG FIX ---
                # Use the modern, recommended way to get the image buffer
                fig.canvas.draw()
                # Get the RGBA buffer from the canvas
                rgba_buffer = fig.canvas.buffer_rgba()
                # Convert it to a NumPy array
                image_array = np.asarray(rgba_buffer)
                # We only need the RGB channels for the GIF, so slice off the alpha channel
                frames_for_gif.append(image_array[:, :, :3])
                # --- END OF BUG FIX ---
                
                plt.close(fig)

            save_filename = f"{video_id}.gif"
            save_path = os.path.join(VISUALIZATION_OUTPUT_DIR, save_filename)
            imageio.mimsave(save_path, frames_for_gif, fps=5)

        except Exception as e:
            print(f"\n[ERROR] Could not process sample {video_id}: {e}")
            traceback.print_exc()
            continue
            
    print(f"\n--- Visualization Complete. Saved GIFs to '{VISUALIZATION_OUTPUT_DIR}'. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated CNN datasets.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the dataset directory to visualize.")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Maximum number of random samples to visualize. Set to 0 for all.")
    args = parser.parse_args()
    
    max_s = args.max_samples if args.max_samples > 0 else None
    output_dir_name = os.path.basename(args.dataset_dir.strip('/\\'))
    
    visualize_dataset(args.dataset_dir, max_s, output_dir_name)

# python -m scripts.vis_cnn_dataset --dataset_dir "CNN_dataset/dataset_2ch_thermal_masked_f10s"