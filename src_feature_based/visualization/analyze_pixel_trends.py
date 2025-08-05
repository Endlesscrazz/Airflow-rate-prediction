# analyze_pixel_trends.py
"""
BATCH SCRIPT to analyze all IR videos in a dataset. For each video, it:
1. Generates a Theil-Sen slope activity map to visualize heating regions.
2. Generates two specific pixel trend plots for known leak locations.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import argparse
import sys
import fnmatch
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import linregress, mstats

# --- Configuration ---
try:
    import config
except ImportError:
    class MockConfig: MAT_FRAMES_KEY = "TempFrames"
    config = MockConfig()

# --- Slope Calculation Functions (from our robust pipeline) ---
def _calculate_slope_for_row(row_data, t):
    """Helper for parallel Theil-Sen slope calculation."""
    W = row_data.shape[0]
    row_slopes = np.zeros(W, dtype=np.float64)
    for c in range(W):
        pixel_series = row_data[c, :]
        if not np.any(np.isnan(pixel_series)) and len(pixel_series) > 1:
            try: slope, _, _, _ = mstats.theilslopes(pixel_series, t, 0.95)
            except (ValueError, IndexError): slope = 0.0
            row_slopes[c] = slope
    return row_slopes

def calculate_theil_sen_slope_map_parallel(frames):
    """Calculates Theil-Sen slope for each pixel in parallel."""
    H, W, T = frames.shape
    if T < 2: return np.zeros((H, W), dtype=np.float64)
    t = np.arange(T)
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_slope_for_row)(frames[r, :, :], t) for r in tqdm(range(H), desc="  Calculating Slopes", leave=False, ncols=100)
    )
    slope_map = np.vstack(results)
    return slope_map

# --- Plotting Functions ---
def plot_activity_map(slope_map, output_path, base_filename):
    """Saves a visualization of the full slope map (heating and cooling)."""
    plt.figure(figsize=(12, 9))
    # Use a diverging colormap to clearly show heating (e.g., red) vs. cooling (e.g., blue)
    # The 'coolwarm' colormap is excellent for this.
    vmax = np.nanpercentile(np.abs(slope_map), 99) # Cap the color range for better contrast
    vmin = -vmax
    plt.imshow(slope_map, cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Heating/Cooling Rate (Theil-Sen Slope)")
    plt.title(f"Full Thermal Activity Map for: {base_filename}\n(Red = Heating, Blue = Cooling)")
    plt.savefig(output_path)
    plt.close()
    print(f"  - Saved activity map to: {os.path.basename(output_path)}")

def plot_pixel_time_series(frames, fps, pixel_coords, pixel_label, output_path, base_filename):
    """Generates and saves a plot of temperature time series for a specific pixel."""
    H, W, num_frames = frames.shape
    pixel_r, pixel_c = pixel_coords
    
    if not (0 <= pixel_r < H and 0 <= pixel_c < W):
        print(f"  - Warning: Pixel {pixel_label} {pixel_coords} is out of bounds for video of shape {(H,W)}. Skipping plot.")
        return

    time_vector_full = np.arange(num_frames) / fps
    series_full_raw = frames[pixel_r, pixel_c, :]
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_vector_full, series_full_raw, label=f'Pixel {pixel_label} {pixel_coords}')
    plt.xlabel(f"Time (s) - Full Duration ({num_frames/fps:.2f}s)")
    plt.ylabel("Raw Temperature")
    plt.title(f"Pixel Temperature Trend for: {base_filename}\nLocation: {pixel_label} {pixel_coords}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    try:
        plt.savefig(output_path)
        print(f"  - Saved trend plot for {pixel_label} to: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"  - Error saving plot {output_path}: {e}")
    plt.close()

# --- Main Batch Processing Logic ---
def main(args):
    """Main execution function to process all videos in a dataset."""
    print("--- Starting Batch Pixel Trend and Activity Analysis ---")
    print(f"Dataset Directory: {args.dataset_dir}")
    print(f"Base Output Directory: {args.base_output_dir}")
    print("-" * 30)
    
    # Define the fixed coordinates for the two leaks
    leak_locations = {
        "leaking_hole_1(center)": (302, 332),
        "leaking_hole_2(top_left)": (129, 254)
    }

    # Walk through the dataset directory
    for root, _, files in os.walk(args.dataset_dir):
        for video_filename in tqdm(fnmatch.filter(files, '*.mat'), desc="Processing Videos"):
            video_path = os.path.join(root, video_filename)
            base_filename = os.path.splitext(video_filename)[0]
            
            # Create a unique subfolder for each video's output
            video_output_dir = os.path.join(args.base_output_dir, base_filename)
            os.makedirs(video_output_dir, exist_ok=True)
            
            print(f"\n--- Analyzing: {video_filename} ---")

            try:
                mat_data = scipy.io.loadmat(video_path)
                frames = mat_data[args.mat_key].astype(np.float64)
            except Exception as e:
                print(f"  Error loading {video_filename}: {e}. Skipping.", file=sys.stderr)
                continue

            # 1. Generate and save the activity map
            # We calculate the full slope (positive and negative) for visualization
            full_slope_map = calculate_theil_sen_slope_map_parallel(frames)
            map_output_path = os.path.join(video_output_dir, f"{base_filename}_activity_map.png")
            plot_activity_map(full_slope_map, map_output_path, base_filename)

            # 2. Generate and save the trend plot for each defined leak location
            for label, coords in leak_locations.items():
                plot_output_path = os.path.join(video_output_dir, f"{base_filename}_{label}.png")
                plot_pixel_time_series(frames, args.fps, coords, label, plot_output_path, base_filename)
                
    print("\n--- Batch analysis complete. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch analyze all .mat files in a directory to visualize activity maps and specific pixel trends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Use directory arguments similar to run_SAM.py
    parser.add_argument("dataset_dir", help="Path to the root directory of the dataset (e.g., 'datasets/dataset_two_holes').")
    parser.add_argument("base_output_dir", help="Base directory to save all outputs. Subfolders will be created for each video.")
    
    parser.add_argument("--mat_key", default=getattr(config, 'MAT_FRAMES_KEY', 'TempFrames'),
                        help="Key in .mat file for temperature frames.")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frames per second of the recordings for accurate time axis.")

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Error: Dataset directory not found at {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)
        
    main(args)