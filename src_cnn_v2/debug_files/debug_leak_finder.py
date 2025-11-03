# src_cnn_v2/debug_leak_finder.py
"""
Standalone Debugging Script for Advanced Leak Detection.

Purpose:
  - Processes a SINGLE .mat video file.
  - Implements the fused (Temporal + Spatial) signal logic to find a leak's "epicenter".
  - Generates a rich verification plot showing the original frame, the fused score map,
    the detected peak, and the final crop area.
  - Saves the resulting fixed-size cropped sequence as a .npy file.

How to Run:
  python src_cnn_v2/debug_leak_finder.py \
    --video_path "/path/to/your/video.mat" \
    --output_dir "/path/to/your/debug_output_folder" \
    --crop_size 16
"""
import os
import sys
import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

# --- Core Logic Functions (Self-Contained) ---

def calculate_temporal_trend_map(frames, method='theil_sen'):
    """
    Calculates the temporal trend (slope) for each pixel over time.
    This version iterates pixel-by-pixel to ensure compatibility with older SciPy versions.
    """
    H, W, T = frames.shape
    if T < 2:
        return np.zeros((H, W), dtype=np.float32)

    t = np.arange(T)
    pixels_over_time = frames.reshape(-1, T)
    slopes = np.zeros(H * W, dtype=np.float32)

    # Loop through each pixel one-by-one instead of using the 'axis' argument
    for i, pixel_series in enumerate(pixels_over_time):
        # Theil-Sen on a single 1D array
        res = stats.theilslopes(pixel_series, t)
        slopes[i] = res[0] # res[0] is the slope

    # Reshape back to (H, W) and handle potential NaNs
    slope_map = slopes.reshape(H, W)
    np.nan_to_num(slope_map, copy=False, nan=0.0)

    # We only care about heating (positive slope)   # can be changed for cooling videos
    trend_map = np.maximum(slope_map, 0.0)
    
    return trend_map

def local_mean_std(img, k=31):
    """
    Calculates local mean/std with a kxk window (reflect padding). k must be odd.
    """
    assert k % 2 == 1, "Kernel size 'k' must be odd."
    r = k // 2
    pad = np.pad(img, ((r, r), (r, r)), mode="reflect")
    win = sliding_window_view(pad, (k, k))
    mean = win.mean(axis=(-1, -2))
    std = win.std(axis=(-1, -2)) + 1e-9
    return mean.astype(np.float32), std.astype(np.float32)

def calculate_local_heat_z_score(frames):
    """
    Calculates the spatial heat anomaly map (temp_z).
    """
    temp_mean = np.mean(frames, axis=2).astype(np.float32)
    loc_mu, loc_sd = local_mean_std(temp_mean, k=31)
    temp_z = np.maximum((temp_mean - loc_mu) / loc_sd, 0.0)
    return temp_z.astype(np.float32)

def save_verification_plot(base_frame, score_map, center_y, center_x, crop_size, save_path):
    """
    Saves a plot showing the score map, peak, and crop box.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Leak Detection Verification\nPeak at ({center_y}, {center_x})", fontsize=16)

    # Subplot 1: Original Frame with Peak and Crop Box
    ax[0].imshow(base_frame, cmap='inferno')
    ax[0].set_title("Original Frame with Crop Area")
    ax[0].scatter([center_x], [center_y], s=200, c='cyan', marker='*')
    rect = patches.Rectangle(
        (center_x - crop_size // 2, center_y - crop_size // 2),
        crop_size, crop_size,
        linewidth=2.5, edgecolor='lime', facecolor='none'
    )
    ax[0].add_patch(rect)
    ax[0].axis('off')

    # Subplot 2: The Fused Score Map
    im = ax[1].imshow(score_map, cmap='hot')
    ax[1].set_title("Fused Score Map (Temporal * Spatial)")
    ax[1].scatter([center_x], [center_y], s=200, c='cyan', marker='*')
    ax[1].axis('off')
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="Score")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  - Saved verification plot to: {save_path}")

def crop_sequence(frames, center_x, center_y, crop_size):
    """
    Crops each frame in a sequence around a given center point.
    """
    H, W, T = frames.shape
    half_crop = crop_size // 2
    
    # Calculate initial crop boundaries
    x_start = max(0, center_x - half_crop)
    x_end = min(W, x_start + crop_size)
    
    y_start = max(0, center_y - half_crop)
    y_end = min(H, y_start + crop_size)

    # Ensure the crop is exactly crop_size x crop_size, shifting if necessary
    if x_end - x_start < crop_size: x_start = x_end - crop_size
    if y_end - y_start < crop_size: y_start = y_end - crop_size

    return frames[y_start:y_end, x_start:x_end, :]


def main():
    parser = argparse.ArgumentParser(description="Debug script for advanced leak detection.")
    parser.add_argument("--video_path", required=True, help="Path to the single .mat video file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output plot and .npy file.")
    parser.add_argument("--crop_size", type=int, default=16, help="The size of the square crop (e.g., 16 for 16x16).")
    parser.add_argument("--heat_power", type=float, default=1.4, help="Exponent to raise the heat map to, emphasizing it.")
    args = parser.parse_args()

    print(f"--- Running Leak Detection Debug Script ---")
    print(f"  - Input Video: {args.video_path}")
    print(f"  - Output Dir:  {args.output_dir}")

    # --- 1. Setup and Load Data ---
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.video_path))[0]

    try:
        frames = scipy.io.loadmat(args.video_path)['TempFrames'].astype(np.float64)
        print(f"  - Loaded video with shape: {frames.shape}")
    except Exception as e:
        sys.exit(f"FATAL: Could not load video file. Error: {e}")

    # --- 2. Core Logic: Fused Signal Detection ---
    print("  - Calculating temporal trend map (Theil-Sen)...")
    temporal_map = calculate_temporal_trend_map(frames)

    print("  - Calculating spatial heat anomaly map (Z-Score)...")
    heat_map = calculate_local_heat_z_score(frames)

    print(f"  - Fusing signals (heat_power = {args.heat_power})...")
    score_map = temporal_map * (heat_map ** args.heat_power)
    
    # --- 3. Find Epicenter ---
    peak_idx = np.argmax(score_map)
    center_y, center_x = np.unravel_index(peak_idx, score_map.shape)
    print(f"  - Found leak epicenter at coordinates: (y={center_y}, x={center_x})")

    # --- 4. Generate Verification Plot ---
    plot_save_path = os.path.join(args.output_dir, f"{base_filename}_leak_verification.png")
    base_frame_for_plot = np.mean(frames, axis=2)
    save_verification_plot(base_frame_for_plot, score_map, center_y, center_x, args.crop_size, plot_save_path)

    # --- 5. Crop and Save Final .npy ---
    # For this debug script, we'll just crop the whole video sequence
    cropped_frames = crop_sequence(frames, center_x, center_y, args.crop_size)
    
    # The V2 pipeline expects (Time, Height, Width)
    cropped_sequence_to_save = cropped_frames.transpose(2, 0, 1)

    npy_save_path = os.path.join(args.output_dir, f"{base_filename}_crop_{args.crop_size}x{args.crop_size}.npy")
    np.save(npy_save_path, cropped_sequence_to_save)
    print(f"  - Saved final cropped sequence of shape {cropped_sequence_to_save.shape} to: {npy_save_path}")
    print("\n--- Debug Script Finished Successfully ---")


if __name__ == "__main__":
    main()

"""
python src_cnn_v2/debug_leak_finder.py \
  --video_path "/scratch/general/vast/u1527145/datasets/Fluke_BrickCladding_2holes_0805_2025_noshutter/T1.6V_2025-08-05-19-57-24_20_34_14_.mat" \
  --output_dir "debug_outputs/Fluke_BrickCladding_2holes_0805_2025_noshutter/vid-1" \
  --crop_size 16
"""
