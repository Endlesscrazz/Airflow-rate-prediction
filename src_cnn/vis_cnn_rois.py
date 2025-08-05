# src_cnn/vis_cnn_rois.py
"""
A standalone utility to visualize the generated ROI sequences for sanity checking
the CNN+LSTM dataset.
(Version with corrected image buffer handling)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
from tqdm import tqdm
import imageio # For saving GIFs

# --- Configuration ---
CNN_DATASET_DIR = "cnn_dataset/dataset_cnn_lstm_f10s"
METADATA_PATH = os.path.join(CNN_DATASET_DIR, "metadata.csv")
VISUALIZATION_OUTPUT_DIR = "verified_masks/dataset_cnn_lstm_f10s"
MAX_SAMPLES_TO_PROCESS = 8

def visualize_roi_sequences():
    """Main function to load ROI sequences and save them as animated GIFs."""
    print("--- ROI Sequence Visualization Tool ---")

    if not os.path.isfile(METADATA_PATH):
        print(f"Error: Metadata file not found at '{METADATA_PATH}'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    print(f"Visualization GIFs will be saved to: '{VISUALIZATION_OUTPUT_DIR}'")

    df_metadata = pd.read_csv(METADATA_PATH)

    if MAX_SAMPLES_TO_PROCESS is not None and MAX_SAMPLES_TO_PROCESS < len(df_metadata):
        samples_to_process = df_metadata.sample(n=MAX_SAMPLES_TO_PROCESS, random_state=42)
        print(f"Processing a random subset of {MAX_SAMPLES_TO_PROCESS} samples.")
    else:
        samples_to_process = df_metadata
        print(f"Processing all {len(df_metadata)} samples found in metadata.")

    for _, sample_row in tqdm(samples_to_process.iterrows(), total=len(samples_to_process), desc="Generating GIFs"):
        video_id = sample_row['video_id']
        
        try:
            sequence_path = os.path.join(CNN_DATASET_DIR, sample_row['image_path'])
            sequence_array = np.load(sequence_path)
            num_frames, height, width = sequence_array.shape
            
            frames_for_gif = []
            
            for i in range(num_frames):
                fig, ax = plt.subplots(figsize=(5, 5))
                frame = sequence_array[i, :, :]
                min_val, max_val = frame.min(), frame.max()
                ax.imshow(frame, cmap='hot', vmin=min_val, vmax=max_val)
                main_title = (
                    f"ID: {video_id}\n"
                    f"Airflow: {sample_row['airflow_rate']:.2f}, Material: {sample_row['material']}"
                )
                ax.set_title(main_title, fontsize=10)
                ax.text(0.02, 0.02, f'Frame: {i+1}/{num_frames}', color='white', 
                        transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.5))
                ax.axis('off')
                fig.tight_layout()
                
                # --- CORRECTED IMAGE BUFFER HANDLING ---
                fig.canvas.draw()
                # Get the RGBA buffer
                rgba_buffer = fig.canvas.buffer_rgba()
                # Convert it to a numpy array
                image_rgba = np.asarray(rgba_buffer)
                # We only need the RGB channels for the GIF, so we slice off the Alpha channel
                image_rgb = image_rgba[:, :, :3]
                frames_for_gif.append(image_rgb)
                # --- END OF FIX ---
                
                plt.close(fig)

            save_filename = f"{video_id}.gif"
            save_path = os.path.join(VISUALIZATION_OUTPUT_DIR, save_filename)
            imageio.mimsave(save_path, frames_for_gif, fps=5)

        except Exception as e:
            print(f"\n[ERROR] Could not process sample {video_id}: {e}")
            continue
            
    print(f"\n--- Visualization Complete. Saved {len(samples_to_process)} GIFs. ---")

if __name__ == "__main__":
    visualize_roi_sequences()

# python -m src_cnn.vis_cnn_rois