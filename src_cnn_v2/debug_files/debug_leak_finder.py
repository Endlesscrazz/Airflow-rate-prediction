# src_cnn_v2/debug_leak_finder.py
"""
Standalone Debugging Script for Advanced Leak Detection. (Upgraded for Multi-Leak & Parallel Processing)

Purpose:
  - Processes a SINGLE .mat video file.
  - Implements the fused (Temporal + Spatial) signal logic to find leak "epicenters".
  - Uses scikit-image and joblib to robustly and quickly find the top N leaks.
  - Generates a rich verification plot showing all detected leaks.
  - Saves a separate cropped .npy file for each detected leak.

How to Run:
  python src_cnn_v2/debug_leak_finder.py \
    --video_path "/path/to/your/video.mat" \
    --output_dir "/path/to/your/debug_output_folder" \
    --num_leaks 2
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
from tqdm import tqdm
from skimage.feature import peak_local_max
from joblib import Parallel, delayed

# --- Core Logic Functions (Self-Contained and Importable) ---

def _calculate_slope_for_row(row_data, t):
    """Helper to calculate Theil-Sen slopes for a single row of pixels."""
    W = row_data.shape[0]
    row_slopes = np.zeros(W, dtype=np.float32)
    for c in range(W):
        try:
            res = stats.theilslopes(row_data[c, :], t)
            row_slopes[c] = res[0] if np.isfinite(res[0]) else 0.0
        except (ValueError, IndexError):
            row_slopes[c] = 0.0
    return row_slopes

def calculate_temporal_trend_map(frames):
    """
    Calculates the temporal trend (slope) for each pixel over time
    using parallel processing to speed up the calculation.
    """
    H, W, T = frames.shape
    if T < 2:
        return np.zeros((H, W), dtype=np.float32)
    t = np.arange(T)
    print("  - Calculating slopes in parallel across all CPU cores...")
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_slope_for_row)(frames[r, :, :], t) for r in tqdm(range(H), desc="  - Processing rows")
    )
    slope_map = np.vstack(results)
    np.nan_to_num(slope_map, copy=False, nan=0.0)
    trend_map = np.maximum(slope_map, 0.0)
    return trend_map.astype(np.float32)

def local_mean_std(img, k=31):
    """
    Calculates local mean/std with a kxk window (reflect padding).
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

def find_top_n_leaks(score_map, num_leaks, min_distance=50, threshold_abs=0.0):
    """
    Finds the top N leak candidates from a score map using peak_local_max.
    """
    coordinates = peak_local_max(score_map, min_distance=min_distance, threshold_abs=threshold_abs)
    if coordinates.size == 0:
        return []
    scores = score_map[coordinates[:, 0], coordinates[:, 1]]
    candidates = [{'coords': (r, c), 'score': s} for (r, c), s in zip(coordinates, scores)]
    candidates.sort(key=lambda p: p['score'], reverse=True)
    return candidates[:num_leaks]

def save_verification_plot(base_frame, score_map, leak_candidates, crop_size, save_path):
    """
    Saves a plot showing the score map and all found leak candidates.
    """
    num_found = len(leak_candidates)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Leak Detection Verification ({num_found} Leak(s) Found)", fontsize=16)
    ax[0].imshow(base_frame, cmap='inferno')
    ax[0].set_title("Original Frame with Crop Areas")
    im = ax[1].imshow(score_map, cmap='hot')
    ax[1].set_title("Fused Score Map (Temporal * Spatial)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="Score")
    for i, leak in enumerate(leak_candidates):
        center_y, center_x = leak['coords']
        ax[0].scatter([center_x], [center_y], s=200, c='cyan', marker='*')
        rect = patches.Rectangle((center_x - crop_size//2, center_y - crop_size//2), crop_size, crop_size, linewidth=2.5, edgecolor='lime', facecolor='none')
        ax[0].add_patch(rect)
        ax[0].text(center_x + 15, center_y, f"#{i+1}", color='white', fontsize=14, weight='bold')
        ax[1].scatter([center_x], [center_y], s=200, c='cyan', marker='*')
        ax[1].text(center_x + 15, center_y, f"#{i+1}", color='white', fontsize=14, weight='bold')
    ax[0].axis('off'); ax[1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  - Saved verification plot to: {save_path}")

def crop_sequence(frames, center_x, center_y, crop_size):
    H, W, T = frames.shape
    half_crop = crop_size // 2
    x_start = max(0, center_x - half_crop); x_end = min(W, x_start + crop_size)
    y_start = max(0, center_y - half_crop); y_end = min(H, y_start + crop_size)
    if x_end - x_start < crop_size: x_start = x_end - crop_size
    if y_end - y_start < crop_size: y_start = y_end - crop_size
    return frames[y_start:y_end, x_start:x_end, :]

def main():
    parser = argparse.ArgumentParser(description="Debug script for advanced leak detection.")
    parser.add_argument("--video_path", required=True, help="Path to the single .mat video file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output plot and .npy files.")
    parser.add_argument("--crop_size", type=int, default=16, help="The size of the square crop.")
    parser.add_argument("--heat_power", type=float, default=1.4, help="Exponent to raise the heat map to.")
    parser.add_argument("--num_leaks", type=int, default=1, help="Number of leaks to detect.")
    args = parser.parse_args()

    print(f"--- Running Leak Detection Debug Script ---")
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.video_path))[0]

    try:
        frames = scipy.io.loadmat(args.video_path)['TempFrames'].astype(np.float64)
    except Exception as e:
        sys.exit(f"FATAL: Could not load video file. Error: {e}")

    temporal_map = calculate_temporal_trend_map(frames)
    heat_map = calculate_local_heat_z_score(frames)
    score_map = temporal_map * (heat_map ** args.heat_power)
    
    min_score_threshold = score_map.max() * 0.05 
    leak_candidates = find_top_n_leaks(score_map, args.num_leaks, min_distance=50, threshold_abs=min_score_threshold)
    
    if not leak_candidates:
        sys.exit("  - No leaks found that meet the criteria. Exiting.")
    
    print(f"  - Found {len(leak_candidates)} leak candidate(s):")
    for i, leak in enumerate(leak_candidates):
        print(f"    - Leak #{i+1}: Coords={leak['coords']}, Score={leak['score']:.4f}")

    plot_save_path = os.path.join(args.output_dir, f"{base_filename}_leak_verification.png")
    base_frame_for_plot = np.mean(frames, axis=2)
    save_verification_plot(base_frame_for_plot, score_map, leak_candidates, args.crop_size, plot_save_path)

    for i, leak in enumerate(leak_candidates):
        center_y, center_x = leak['coords']
        cropped_frames = crop_sequence(frames, center_x, center_y, args.crop_size)
        cropped_sequence_to_save = cropped_frames.transpose(2, 0, 1)
        npy_save_path = os.path.join(args.output_dir, f"{base_filename}_crop_{args.crop_size}x{args.crop_size}_leak_{i+1}.npy")
        np.save(npy_save_path, cropped_sequence_to_save)
        print(f"  - Saved cropped sequence for leak #{i+1} to: {npy_save_path}")

    print("\n--- Debug Script Finished Successfully ---")

if __name__ == "__main__":
    main()

"""
python src_cnn_v2/debug_files/debug_leak_finder.py \
  --video_path "//Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter/T1.4V_2025-08-08-19-44-50_20_32_12_.mat" \
  --output_dir "debug_outputs/Fluke_BrickCladding_2holes_0808_2025_noshutter/vid-1" \
  --crop_size 16 \
  --num_leaks 2
"""
