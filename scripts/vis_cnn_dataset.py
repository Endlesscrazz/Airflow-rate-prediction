# scripts/vis_dataset.py
"""
A standalone utility to visualize the generated dataset sequences for sanity checking.
This script can handle thermal, optical flow, and hybrid datasets.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import argparse
import cv2

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
    
    # Determine number of channels from parameters file
    try:
        with open(PARAMS_PATH, 'r') as f:
            lines = f.readlines()
            channel_line = [line for line in lines if "Image Channels" in line][0]
            num_channels = int(channel_line.split(":")[1].strip())
    except (FileNotFoundError, IndexError):
        print(f"WARNING: Could not read dataset_parameters.txt. This is not a critical error.", file=sys.stderr)
        # Fallback to inferring from a sample
        df_temp = pd.read_csv(METADATA_PATH)
        sample_path = os.path.join(dataset_dir, df_temp['image_path'].iloc[0])
        sample_array = np.load(sample_path)
        num_channels = sample_array.shape[-1] if sample_array.ndim == 4 else 1

    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    print(f"Dataset to visualize: '{dataset_dir}' ({num_channels} channels)")
    print(f"Visualization GIFs will be saved to: '{VISUALIZATION_OUTPUT_DIR}'")

    df_metadata = pd.read_csv(METADATA_PATH)
    samples_to_process = df_metadata
    if max_samples is not None and max_samples < len(df_metadata):
        samples_to_process = df_metadata.sample(n=max_samples, random_state=cfg.RANDOM_STATE)
        print(f"Processing a random subset of {max_samples} samples.")
    else:
        samples_to_process = df_metadata
        print(f"Processing all {len(df_metadata)} samples found in metadata.")

    # 4. Loop and Generate GIFs
    for _, sample_row in tqdm(samples_to_process.iterrows(), total=len(samples_to_process), desc="Generating GIFs"):
        video_id = sample_row['video_id']
        
        try:
            sequence_path = os.path.join(dataset_dir, sample_row['image_path'])
            sequence_array = np.load(sequence_path)
            
            frames_for_gif = []
            
            # This loop generates one image (frame of the GIF) for each timestep in the sequence
            for i in range(sequence_array.shape[0]):
                
                # --- THIS IS THE CORE LOGIC THAT HANDLES DIFFERENT CHANNEL COUNTS ---
                if num_channels == 1:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    frame = sequence_array[i, :, :]
                    ax.imshow(frame, cmap='hot')
                    ax.set_title("Thermal Channel")
                
                elif num_channels == 2:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    flow = sequence_array[i, :, :, :] # Shape (H, W, 2)
                    # Use OpenCV's built-in HSV method for visualizing flow
                    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
                    hsv[..., 0] = angle * 180 / np.pi / 2
                    hsv[..., 1] = 255
                    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    ax.imshow(rgb_flow)
                    ax.set_title("Optical Flow (Colorized)")

                elif num_channels == 3:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    hybrid_frame = sequence_array[i, :, :, :] # Shape (H, W, 3)
                    
                    # Channel 0: Thermal
                    axes[0].imshow(hybrid_frame[:, :, 0], cmap='hot')
                    axes[0].set_title("Channel 0: Thermal")
                    
                    # Channel 1: Flow-X
                    axes[1].imshow(hybrid_frame[:, :, 1], cmap='RdBu')
                    axes[1].set_title("Channel 1: Flow-X")

                    # Channel 2: Flow-Y
                    axes[2].imshow(hybrid_frame[:, :, 2], cmap='RdBu')
                    axes[2].set_title("Channel 2: Flow-Y")
                
                else:
                    raise ValueError(f"Unsupported number of channels: {num_channels}")

                # --- Common plotting code for all types ---
                main_title = (
                    f"ID: {video_id} | Frame: {i+1}/{sequence_array.shape[0]}\n"
                    f"Airflow: {sample_row['airflow_rate']:.2f}, Material: {sample_row['material']}"
                )
                fig.suptitle(main_title, fontsize=12)
                
                # Turn off axes for all subplots
                for axis in fig.get_axes():
                    axis.axis('off')
                
                fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
                
                # Convert plot to an image buffer for the GIF
                fig.canvas.draw()
                image_rgba = np.asarray(fig.canvas.buffer_rgba())
                image_rgb = image_rgba[:, :, :3]
                frames_for_gif.append(image_rgb)
                plt.close(fig)

            # Save the final animated GIF
            save_filename = f"{video_id}.gif"
            save_path = os.path.join(VISUALIZATION_OUTPUT_DIR, save_filename)
            imageio.mimsave(save_path, frames_for_gif, fps=5)

        except Exception as e:
            print(f"\n[ERROR] Could not process sample {video_id}: {e}")
            import traceback
            traceback.print_exc() # Print full error for debugging
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

"""
python -m src_cnn.vis_cnn_dataset --dataset_dir "CNN_dataset/dataset_2ch_flow"

python -m src_cnn.vis_cnn_dataset --dataset_dir "CNN_dataset/dataset_1ch_thermal"

python -m src_cnn.vis_dataset --dataset_dir "cnn_dataset/dataset_3ch_hybrid"

python -m src_cnn.vis_dataset --dataset_dir "cnn_dataset/dataset_2ch_flow" --max_samples 20
"""