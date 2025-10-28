# src_cnn_v2/find_leaking_holes.py
"""
Batch-Processing Script for Advanced Leak Detection. (VERSION 5 - FULLY PARALLEL)

Purpose:
  - Processes a directory of .mat video files in parallel using all available CPU cores.
  - Implements the fused (Temporal + Spatial) signal logic to find leak "epicenters".
  - Intelligently chooses the fastest available temporal analysis method.
  - Robustly assigns hole IDs using positional anchors where available, and falls back
    to a signal-strength-based method otherwise.
  - Saves precise (y, x) coordinates for each leak into a lightweight .json file.
  - Generates verification plots when run with the --debug flag.
"""
import os
import sys
import argparse
import json
import glob
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from skimage.feature import peak_local_max
from tqdm import tqdm
import traceback
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

# --- ANCHOR DATABASE ---
POSITIONAL_ANCHORS = {
    # "0616": { # Use the date as the unique key
    #     "1": np.array([302, 332]), # center hole
    #     "2": np.array([128, 251])  # top left corner hole
    # },
    # "0805": { # Use the date as the unique key
    #     "1": np.array([274, 328]), # center hole
    #     "2": np.array([360, 140])  # bottom_left hole
    # },
    "0808": { # Use the date as the unique key
        "1": np.array([269, 327]), # center hole
        "2": np.array([358, 142])  # Corrected bottom_left hole
    }
    # Add other datasets here as needed (e.g., 'hardyboard' using '0813')
}
    # Add other datasets here as needed (e.g., 'hardyboard')


# --- Core Logic Functions ---

def _calculate_slope_for_row(row_data, t):
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
    H, W, T = frames.shape
    if T < 2: return np.zeros((H, W), dtype=np.float32)
    t = np.arange(T)
    try:
        res = stats.theilslopes(frames, t, axis=2)
        slope_map = res[0]
    except TypeError:
        # This fallback is now also parallel, but within a single video's processing
        results = Parallel(n_jobs=-1)(
            delayed(_calculate_slope_for_row)(frames[r, :, :], t) for r in range(H)
        )
        slope_map = np.vstack(results)
    np.nan_to_num(slope_map, copy=False, nan=0.0)
    trend_map = np.maximum(slope_map, 0.0)
    return trend_map.astype(np.float32)

def local_mean_std(img, k=31):
    assert k % 2 == 1, "k must be odd."
    r = k // 2
    pad = np.pad(img, ((r, r), (r, r)), mode="reflect")
    win = sliding_window_view(pad, (k, k))
    mean = win.mean(axis=(-1, -2))
    std = win.std(axis=(-1, -2)) + 1e-9
    return mean.astype(np.float32), std.astype(np.float32)

def calculate_local_heat_z_score(frames):
    temp_mean = np.mean(frames, axis=2).astype(np.float32)
    loc_mu, loc_sd = local_mean_std(temp_mean, k=31)
    temp_z = np.maximum((temp_mean - loc_mu) / loc_sd, 0.0)
    return temp_z.astype(np.float32)

def find_top_n_leaks(score_map, num_leaks, min_distance=50):
    min_score_threshold = score_map.max() * 0.05 if score_map.max() > 0 else 0
    coordinates = peak_local_max(score_map, min_distance=min_distance, threshold_abs=min_score_threshold)
    if coordinates.size == 0: return []
    scores = score_map[coordinates[:, 0], coordinates[:, 1]]
    candidates = [{'coords': (r, c), 'score': s} for (r, c), s in zip(coordinates, scores)]
    candidates.sort(key=lambda p: p['score'], reverse=True)
    return candidates[:num_leaks]

def find_leaks_in_rois(score_map, anchor_points, search_radius=50):
    """
    Finds the strongest peak within a specific search radius around each anchor point.
    This is a robust method that ignores noise in other parts of the image.
    """
    assigned_leaks = []
    H, W = score_map.shape

    for hole_id, anchor_coord in anchor_points.items():
        # Define the Search Zone (ROI) boundaries
        center_y, center_x = anchor_coord
        y_min = max(0, center_y - search_radius)
        y_max = min(H, center_y + search_radius)
        x_min = max(0, center_x - search_radius)
        x_max = min(W, center_x + search_radius)

        # Create a mask for the entire image, with only the ROI set to True
        roi_mask = np.zeros_like(score_map, dtype=bool)
        roi_mask[y_min:y_max, x_min:x_max] = True

        # Find all peaks ONLY within the ROI
        roi_candidates = find_top_n_leaks(score_map * roi_mask, num_leaks=1, min_distance=1)
        
        if roi_candidates:
            # If a peak was found in the zone, it's our leak
            best_peak_in_roi = roi_candidates[0]
            assigned_leaks.append({
                "hole_id": int(hole_id),
                "center_y": int(best_peak_in_roi['coords'][0]),
                "center_x": int(best_peak_in_roi['coords'][1]),
                "score": float(best_peak_in_roi['score'])
            })

    return assigned_leaks

def save_verification_plot(base_frame, score_map, leak_candidates, crop_size, save_path):
    # This function remains unchanged
    num_found = len(leak_candidates)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Leak Detection Verification ({num_found} Leak(s) Found)", fontsize=16)
    # ... (rest of the plotting code is the same)
    ax[0].imshow(base_frame, cmap='inferno')
    ax[0].set_title("Original Frame with Crop Areas")
    im = ax[1].imshow(score_map, cmap='hot')
    ax[1].set_title("Fused Score Map (Temporal * Spatial)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="Score")
    for i, leak in enumerate(leak_candidates):
        center_y, center_x = leak['coords']
        ax[0].scatter([center_x], [center_y], s=200, c='cyan', marker='*')
        rect = patches.Rectangle((center_x - crop_size // 2, center_y - crop_size // 2), crop_size, crop_size, linewidth=2.5, edgecolor='lime', facecolor='none')
        ax[0].add_patch(rect)
        ax[0].text(center_x + 15, center_y, f"#{leak.get('hole_id', i+1)}", color='white', fontsize=14, weight='bold')
        ax[1].scatter([center_x], [center_y], s=200, c='cyan', marker='*')
        ax[1].text(center_x + 15, center_y, f"#{leak.get('hole_id', i+1)}", color='white', fontsize=14, weight='bold')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# --- NEW: Worker Function for Parallel Processing ---
def process_video(video_path, args):
    """
    Processes a single video file. This is the "workload" for each parallel job.
    """
    try:
        relative_path = os.path.relpath(os.path.dirname(video_path), args.dataset_dir)
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(args.base_output_dir, relative_path, base_filename)
        os.makedirs(video_output_dir, exist_ok=True)

        frames = scipy.io.loadmat(video_path)['TempFrames'].astype(np.float64)

        # --- ADD TEMPORAL NORMALIZATION HERE ---
        # Transpose to (T, H, W) for easier normalization
        frames_T_first = frames.transpose(2, 0, 1)
        # Calculate the mean of each frame
        frame_means = frames_T_first.mean(axis=(1, 2), keepdims=True)
        # Normalize and transpose back to (H, W, T)
        normalized_frames_T_first = frames_T_first / (frame_means + 1e-9)
        frames = normalized_frames_T_first.transpose(1, 2, 0)

        temporal_map = calculate_temporal_trend_map(frames)
        heat_map = calculate_local_heat_z_score(frames)
        score_map = temporal_map * (heat_map ** args.heat_power)
        
        all_candidates = find_top_n_leaks(score_map, num_leaks=10, min_distance=50)
        if not all_candidates:
            return # Silently skip if no peaks are found

        # --- SMART ASSIGNMENT LOGIC ---
        output_data = []
        anchor_set_key = next((key for key in POSITIONAL_ANCHORS if key in video_path), None)

        if anchor_set_key:
            # PATH A: Use robust "Targeted Search" in ROIs
            anchor_points = POSITIONAL_ANCHORS[anchor_set_key]
            output_data = find_leaks_in_rois(score_map, anchor_points, search_radius=50) # <-- NEW CALL
        else:
            # PATH B (FALLBACK): Use simple "top N by score"
            # This part remains unchanged
            tqdm.write(f"  - INFO: No positional anchors for '{base_filename}'. Using top {args.num_leaks} by score.")
            top_candidates = find_top_n_leaks(score_map, num_leaks=args.num_leaks, min_distance=50)
            for i, leak in enumerate(top_candidates):
                y, x = leak['coords']
                output_data.append({
                    "hole_id": i + 1,
                    "center_y": int(y), "center_x": int(x), "score": float(leak['score'])
                })
        
        if not output_data: return # Skip if assignment results in no leaks

        json_path = os.path.join(video_output_dir, f"{base_filename}_coordinates.json")
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        if args.debug:
            plot_save_path = os.path.join(video_output_dir, f"{base_filename}_leak_verification.png")
            base_frame_for_plot = np.mean(frames, axis=2)
            candidates_for_plot = [{'coords': (d['center_y'], d['center_x']), 'hole_id': d['hole_id']} for d in output_data]
            save_verification_plot(base_frame_for_plot, score_map, candidates_for_plot, args.crop_size, plot_save_path)

    except Exception as e:
        print(f"\n--- ERROR processing {os.path.basename(video_path)} ---")
        print(f"    Error: {e}")
        if args.debug:
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Batch script for advanced leak coordinate generation.")
    parser.add_argument("--dataset_dir", required=True, help="Path to the parent directory containing .mat files.")
    parser.add_argument("--base_output_dir", required=True, help="Directory to save the output coordinate files and plots.")
    parser.add_argument("--num_leaks", type=int, default=2, help="Number of leaks to detect (used in fallback mode).")
    parser.add_argument("--crop_size", type=int, default=16, help="Crop size to show in verification plots.")
    parser.add_argument("--heat_power", type=float, default=1.4, help="Exponent for the heat map.")
    parser.add_argument("--debug", action='store_true', help="Enable to generate verification plots for each video.")
    args = parser.parse_args()

    print("--- Starting Batch Leak Coordinate Generation ---")
    search_pattern = os.path.join(args.dataset_dir, '**', '*.mat')
    mat_file_paths = [p for p in glob.glob(search_pattern, recursive=True) if not os.path.basename(p).startswith('._')]
    
    if not mat_file_paths:
        sys.exit(f"Error: No .mat files found in {args.dataset_dir}")

    print(f"Found {len(mat_file_paths)} videos to process. Starting parallel processing...")

    # --- FULLY PARALLEL EXECUTION ---
    # This will now process all videos in parallel, showing a progress bar.
    Parallel(n_jobs=-1, verbose=10)(
        delayed(process_video)(video_path, args)
        for video_path in mat_file_paths
    )
            
    print("\n--- Batch Processing Complete ---")

if __name__ == "__main__":
    main()
"""
#BRICKCLADDING
python -m src_cnn_v2.find_leaking_holes \
    --dataset_dir /scratch/general/vast/u1527145/datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter \
    --base_output_dir /scratch/general/vast/u1527145/Airflow-rate-prediction/Output_SAM/datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter \
    --num_leaks 2 \
    --debug

"""