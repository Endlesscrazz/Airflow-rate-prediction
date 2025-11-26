# src_cnn_v2/find_leaking_holes.py
"""
Batch-Processing Script for Advanced Leak Detection with Anchor-Based Labeling.
(VERSION 8 - DYNAMIC DETECTION + ANCHOR LABELING)

- Step 1: Fuses Temporal Trend * Spatial Z-Score to create a "Likelihood Map".
- Step 2: Dynamically detects all significant peaks in the image.
- Step 3: Uses POSITIONAL ANCHORS to correctly ID the peaks (Hole 1 vs Hole 2).
- Prevents "Hole Swapping" while allowing the leak to drift/move naturally.
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
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import traceback
from scipy.ndimage import uniform_filter

# --- ANCHOR DATABASE ---
# Format: "UniqueSubstring": { "HoleID": np.array([Row(y), Col(x)]) }
POSITIONAL_ANCHORS = {
    # Fluke_HardyBoard_08132025_2holes_noshutter
    "0813": { 
        "1": np.array([322, 328]), 
        "2": np.array([130, 499])  
    },
    # Fluke_HardyBoard_03132025
    "0313": { 
        "1": np.array([235, 313])
    },
    # Legacy / Other datasets (Optional, keeping for compatibility)
    "0616": { "1": np.array([302, 332]), "2": np.array([128, 251]) },
    "0805": { "1": np.array([274, 328]), "2": np.array([360, 140]) },
    "0808": { "1": np.array([269, 327]), "2": np.array([358, 142]) },
}

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

def match_and_label_leaks(score_map, anchor_points, search_radius=60):
    """
    1. Detects ALL peaks in the score map (Dynamic).
    2. Matches peaks to the closest Anchor Point (Labeling).
    3. If no peak is found near an anchor, falls back to local max in that area.
    """
    H, W = score_map.shape
    final_leaks = []

    # 1. Dynamic Detection: Find all significant peaks in the image
    # We use a low threshold to catch even faint leaks
    global_max = score_map.max()
    threshold = global_max * 0.05 if global_max > 0 else 0
    
    # Get all candidates (y, x)
    candidates = peak_local_max(score_map, min_distance=10, threshold_abs=threshold)
    
    # 2. Match Candidates to Anchors
    for hole_id, anchor_coord in anchor_points.items():
        anchor_y, anchor_x = anchor_coord
        
        best_candidate = None
        best_score = -1.0
        
        # Look for the strongest candidate within the search radius of this anchor
        if len(candidates) > 0:
            # Calculate distances from this anchor to all candidates
            dists = cdist([anchor_coord], candidates)[0]
            
            # Filter by radius
            nearby_indices = np.where(dists <= search_radius)[0]
            
            if len(nearby_indices) > 0:
                # Of the nearby candidates, pick the one with the highest score
                nearby_candidates = candidates[nearby_indices]
                nearby_scores = score_map[nearby_candidates[:, 0], nearby_candidates[:, 1]]
                
                best_idx_local = np.argmax(nearby_scores)
                best_candidate = nearby_candidates[best_idx_local]
                best_score = nearby_scores[best_idx_local]

        # 3. Fallback: If dynamic detection missed it (too faint), force a look in the ROI
        if best_candidate is None:
            # Define ROI
            y_min, y_max = max(0, anchor_y - search_radius), min(H, anchor_y + search_radius)
            x_min, x_max = max(0, anchor_x - search_radius), min(W, anchor_x + search_radius)
            
            roi = score_map[y_min:y_max, x_min:x_max]
            if roi.size > 0 and roi.max() > 0:
                # Find max in ROI
                local_y, local_x = np.unravel_index(np.argmax(roi), roi.shape)
                best_candidate = np.array([y_min + local_y, x_min + local_x])
                best_score = roi.max()
            else:
                # Last resort: use the anchor itself
                best_candidate = anchor_coord
                best_score = 0.0

        final_leaks.append({
            "hole_id": int(hole_id),
            "center_y": int(best_candidate[0]),
            "center_x": int(best_candidate[1]),
            "score": float(best_score),
            "anchor_y": int(anchor_y), # Saved for debugging/plotting
            "anchor_x": int(anchor_x)
        })
    
    return final_leaks

def save_verification_plot(base_frame, score_map, leak_candidates, crop_size, save_path):
    num_found = len(leak_candidates)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Leak Detection (Dynamic + Anchored) - Found {num_found}", fontsize=16)
    
    ax[0].imshow(base_frame, cmap='inferno')
    ax[0].set_title("Original Frame (Mean)")
    
    im = ax[1].imshow(score_map, cmap='hot')
    ax[1].set_title("Score Map (Trend * Heat)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    
    for i, leak in enumerate(leak_candidates):
        cy, cx = leak['center_y'], leak['center_x']
        ay, ax_coord = leak.get('anchor_y'), leak.get('anchor_x')
        hid = leak['hole_id']
        
        # Plot Detected Point
        ax[0].scatter([cx], [cy], s=100, c='cyan', marker='x', label='Detected')
        rect = patches.Rectangle((cx - crop_size // 2, cy - crop_size // 2), crop_size, crop_size, linewidth=2, edgecolor='lime', facecolor='none')
        ax[0].add_patch(rect)
        
        # Plot Anchor Point (Ghost) to show drift
        if ay is not None:
            ax[0].scatter([ax_coord], [ay], s=50, c='white', marker='o', alpha=0.5, label='Anchor')
            # Draw line connecting anchor to detection
            ax[0].plot([ax_coord, cx], [ay, cy], c='white', linestyle='--', alpha=0.5)

        ax[0].text(cx + 10, cy, f"#{hid}", color='white', fontsize=12, weight='bold')
        
        # Plot on Score Map
        ax[1].scatter([cx], [cy], s=100, c='cyan', marker='x')
        ax[1].text(cx + 10, cy, f"#{hid}", color='white', fontsize=12, weight='bold')
        
    ax[0].legend(loc='upper right', fontsize='small')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

def process_video(video_path, args):
    try:
        # --- 1. SETUP PATHS ---
        relative_path = os.path.relpath(os.path.dirname(video_path), args.dataset_dir)
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(args.base_output_dir, relative_path, base_filename)
        os.makedirs(video_output_dir, exist_ok=True)

        # --- 2. LOAD & PREPROCESS ---
        frames = scipy.io.loadmat(video_path)['TempFrames'].astype(np.float32)
        
        # Temporal Normalization (Match Colleague / V2 Pipeline)
        frames_T_first = frames.transpose(2, 0, 1)
        frame_means = frames_T_first.mean(axis=(1, 2), keepdims=True)
        frame_means[frame_means < 1e-6] = 1.0
        frames_normalized = (frames_T_first / frame_means).transpose(1, 2, 0)

        # --- 3. GENERATE SCORE MAP ---
        temporal_map = calculate_temporal_trend_map(frames_normalized)
        heat_map = calculate_local_heat_z_score(frames_normalized)
        
        score_map = temporal_map * (heat_map ** args.heat_power)
        
        # Basic 3x3 smoothing to reduce single-pixel noise spikes
        score_map = uniform_filter(score_map, size=3)

        # --- 4. DETECT & LABEL ---
        output_data = []
        
        # Find which anchor set applies to this video
        anchor_set_key = next((key for key in POSITIONAL_ANCHORS if key in video_path), None)

        if anchor_set_key:
            # Use Anchors to Label the Dynamic Detections
            anchor_points = POSITIONAL_ANCHORS[anchor_set_key]
            output_data = match_and_label_leaks(score_map, anchor_points, search_radius=60)
        else:
            # Fallback: Just find top N (No ID guarantees)
            all_candidates = peak_local_max(score_map, min_distance=50, num_peaks=args.num_leaks)
            for i, (y, x) in enumerate(all_candidates):
                output_data.append({
                    "hole_id": i + 1,
                    "center_y": int(y), "center_x": int(x), "score": float(score_map[y, x])
                })
        
        if not output_data: return

        # --- 5. SAVE OUTPUTS ---
        json_path = os.path.join(video_output_dir, f"{base_filename}_coordinates.json")
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        if args.debug:
            plot_save_path = os.path.join(video_output_dir, f"{base_filename}_leak_verification.png")
            base_frame_for_plot = np.mean(frames, axis=2)
            save_verification_plot(base_frame_for_plot, score_map, output_data, args.crop_size, plot_save_path)

    except Exception as e:
        print(f"\n--- ERROR processing {os.path.basename(video_path)} ---")
        print(f"    Error: {e}")
        if args.debug:
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Batch script for leak detection with anchor labeling.")
    parser.add_argument("--dataset_dir", required=True, help="Path to the parent directory containing .mat files.")
    parser.add_argument("--base_output_dir", required=True, help="Directory to save the output coordinate files and plots.")
    parser.add_argument("--num_leaks", type=int, default=2, help="Fallback number of leaks.")
    parser.add_argument("--crop_size", type=int, default=16, help="Crop size to show in verification plots.")
    parser.add_argument("--heat_power", type=float, default=1.4, help="Exponent for the heat map.")
    parser.add_argument("--debug", action='store_true', help="Enable to generate verification plots.")
    args = parser.parse_args()

    print("--- Starting Batch Leak Detection (Dynamic + Anchored) ---")
    print(f"Loaded Anchors for keys: {list(POSITIONAL_ANCHORS.keys())}")
    
    search_pattern = os.path.join(args.dataset_dir, '**', '*.mat')
    mat_file_paths = [p for p in glob.glob(search_pattern, recursive=True) if not os.path.basename(p).startswith('._')]
    
    if not mat_file_paths:
        sys.exit(f"Error: No .mat files found in {args.dataset_dir}")

    print(f"Found {len(mat_file_paths)} videos. Processing...")

    Parallel(n_jobs=-1, verbose=5)(
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

# GYPSUM 10 Hole
python -m src_cnn_v2.find_leaking_holes \
    --dataset_dir /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_Gypsum_09032025_10holes_noshutter_Sameem \
    --base_output_dir Output_SAM/datasets/Fluke_Gypsum_09032025_10holes_noshutter_No-anchor \
    --num_leaks 10 \
    --debug
    
"""