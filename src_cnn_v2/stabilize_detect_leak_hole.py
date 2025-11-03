# src_cnn_v2/stabilize_detect_leak_hole.py
"""
Standalone Debugging and Validation Script for Leak Detection & Stabilization. (Version 3)

Purpose:
  - Processes a SINGLE .mat video file.
  - Artificially introduces shake to create a "shaky" version.
  - Runs a stabilization algorithm on the shaky video.
  - Runs the fused-signal leak detection on all three video versions.
  - Generates a comprehensive comparison plot AND a side-by-side comparison GIF.

How to Run:
  python src_cnn_v2/stabilize_detect_leak_hole.py \
    --video_path "/path/to/your/video.mat" \
    --output_dir "debug_stabilization_test" \
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
import imageio.v2 as imageio
import cv2

# --- Import the necessary functions from our stabilization POC ---
from poc_stabilization import add_gradual_shake, stabilize_video_phase_correlation

# --- Core Leak Detection Logic ---
def _calculate_slope_for_row(row_data, t):
    W = row_data.shape[0]
    row_slopes = np.zeros(W, dtype=np.float32)
    for c in range(W):
        try:
            res = stats.theilslopes(row_data[c, :], t)
            row_slopes[c] = res[0]
        except (ValueError, IndexError):
            row_slopes[c] = 0.0
    return row_slopes

def calculate_temporal_trend_map(frames):
    H, W, T = frames.shape
    if T < 2: return np.zeros((H, W), dtype=np.float32)
    t = np.arange(T)
    print("  - Calculating slopes in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_slope_for_row)(frames[r, :, :], t) for r in tqdm(range(H), desc="  - Processing rows")
    )
    slope_map = np.vstack(results)
    np.nan_to_num(slope_map, copy=False, nan=0.0)
    trend_map = np.maximum(slope_map, 0.0)
    return trend_map.astype(np.float32)

def local_mean_std(img, k=31):
    assert k % 2 == 1, "Kernel size 'k' must be odd."
    r = k // 2
    pad = np.pad(img, ((r, r), (r, r)), mode="reflect")
    win = sliding_window_view(pad, (k, k))
    mean = win.mean(axis=(-1, -2)); std = win.std(axis=(-1, -2)) + 1e-9
    return mean.astype(np.float32), std.astype(np.float32)

def calculate_local_heat_z_score(frames):
    temp_mean = np.mean(frames, axis=2).astype(np.float32)
    loc_mu, loc_sd = local_mean_std(temp_mean, k=31)
    temp_z = np.maximum((temp_mean - loc_mu) / loc_sd, 0.0)
    return temp_z.astype(np.float32)

def find_top_n_leaks(score_map, num_leaks, min_distance=20, threshold_abs=0.0):
    coordinates = peak_local_max(score_map, min_distance=min_distance, threshold_abs=threshold_abs)
    if coordinates.size == 0: return []
    scores = score_map[coordinates[:, 0], coordinates[:, 1]]
    candidates = [{'coords': (r, c), 'score': s} for (r, c), s in zip(coordinates, scores)]
    candidates.sort(key=lambda p: p['score'], reverse=True)
    return candidates[:num_leaks]

# --- Visualization Functions ---

def save_full_comparison_plot(original_frames, shaky_frames, stabilized_frames,
                              original_leaks, shaky_leaks, stabilized_leaks,
                              crop_size, save_path):
    fig, axs = plt.subplots(2, 3, figsize=(28, 16), dpi=120)
    fig.suptitle("Validation: Leak Detection with and without Stabilization", fontsize=24, y=0.97)
    videos = [original_frames, shaky_frames, stabilized_frames]
    leaks_list = [original_leaks, shaky_leaks, stabilized_leaks]
    titles = ["1. Original Video (Ground Truth)", "2. Shaky Video (Problem)", "3. Stabilized Video (Solution)"]
    colors = ['lime', 'red', 'lime']

    for i, ax_col in enumerate(axs.T):
        mean_frame = np.mean(videos[i], axis=2)
        ax_col[0].set_title(titles[i], fontsize=18)
        ax_col[0].imshow(mean_frame, cmap='inferno')
        
        score_map = calculate_temporal_trend_map(videos[i]) * \
                    (calculate_local_heat_z_score(videos[i]) ** 1.4)
        ax_col[1].set_title("Fused Score Map", fontsize=16)
        im = ax_col[1].imshow(score_map, cmap='hot')
        fig.colorbar(im, ax=ax_col[1], fraction=0.046, pad=0.04)

        if not leaks_list[i]:
            for row_ax in ax_col:
                row_ax.text(0.5, 0.5, "No Leaks Found", color='red', fontsize=20, ha='center', transform=row_ax.transAxes)
        else:
            for j, leak in enumerate(leaks_list[i]):
                center_y, center_x = leak['coords']
                ax_col[0].scatter([center_x], [center_y], s=300, c=colors[i], marker='*')
                rect = patches.Rectangle((center_x - crop_size//2, center_y - crop_size//2),
                                         crop_size, crop_size, linewidth=2.5, edgecolor=colors[i], facecolor='none')
                ax_col[0].add_patch(rect)
                ax_col[0].text(center_x + 15, center_y, f"#{j+1}", color='white', fontsize=16, weight='bold')
                ax_col[1].scatter([center_x], [center_y], s=300, c='cyan', marker='*')
                ax_col[1].text(center_x + 15, center_y, f"#{j+1}", color='white', fontsize=16, weight='bold')

    for ax in axs.flatten():
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nSaved full comparison plot to: {save_path}")

# --- NEW: GIF Visualization Function ---
def create_comparison_gif(original_frames, shaky_frames, stabilized_frames, save_path, fps=10):
    """
    Generates a 3-panel GIF showing Original, Shaky, and Stabilized videos side-by-side
    with corrected high-contrast grayscale.
    """
    print(f"  - Creating 3-panel comparison GIF: {os.path.basename(save_path)}")
    H, W, T = original_frames.shape
    
    # --- START OF FIX ---
    # 1. Use robust percentile-based normalization to find the desired min/max range
    vmin = np.percentile(original_frames, 1)
    vmax = np.percentile(original_frames, 99)
    
    # 2. Create a robust normalization and conversion function
    def normalize_for_viz(frame, vmin, vmax):
        # Clip the data to the desired percentile range to remove outliers
        clipped_frame = np.clip(frame, vmin, vmax)
        
        # Use cv2.normalize to safely scale the clipped data to the 0-255 range
        # This is more robust than manual scaling for converting data types.
        frame_8bit = cv2.normalize(clipped_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Convert the single-channel grayscale to a 3-channel RGB image for text drawing
        return cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2RGB)
    # --- END OF FIX ---

    with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
        for i in range(T):
            # Normalize all three frames using the same robust function and limits
            orig_viz = normalize_for_viz(original_frames[:, :, i], vmin, vmax)
            shaky_viz = normalize_for_viz(shaky_frames[:, :, i], vmin, vmax)
            stabilized_viz = normalize_for_viz(stabilized_frames[:, :, i], vmin, vmax)
            
            combined_frame = np.hstack((orig_viz, shaky_viz, stabilized_viz))
            
            # Add text labels
            cv2.putText(combined_frame, '1. Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_frame, '2. Shaky', (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combined_frame, '3. Stabilized', (2 * W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(combined_frame, f'Frame: {i}/{T}', (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            writer.append_data(combined_frame)
            
    print(f"  - Saved comparison GIF to: {save_path}")

def create_heatmap_gif(shaky_frames, stabilized_frames, save_path, fps=10, cmap='inferno'):
    """
    Generates a 2-panel GIF showing Shaky and Stabilized videos side-by-side
    in a heatmap colormap for better thermal visualization.
    """
    print(f"  - Creating 2-panel heatmap GIF: {os.path.basename(save_path)}")
    H, W, T = shaky_frames.shape
    
    # Get the colormap from matplotlib
    colormap = plt.get_cmap(cmap)
    
    # Find robust global min/max across both videos for consistent coloring
    vmin = np.percentile(shaky_frames, 1)
    vmax = np.percentile(shaky_frames, 99)
    
    # Normalize the data to [0, 1] based on these limits
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
        for i in range(T):
            # Apply normalization and colormap to each frame
            # The colormap returns an (H, W, 4) RGBA image
            shaky_rgba = colormap(norm(shaky_frames[:, :, i]))
            stabilized_rgba = colormap(norm(stabilized_frames[:, :, i]))
            
            # Convert RGBA to RGB by discarding the alpha channel and scaling to 8-bit
            shaky_rgb = (shaky_rgba[:, :, :3] * 255).astype(np.uint8)
            stabilized_rgb = (stabilized_rgba[:, :, :3] * 255).astype(np.uint8)
            
            # Stack the two frames side-by-side
            combined_frame = np.hstack((shaky_rgb, stabilized_rgb))
            
            # Add text labels
            cv2.putText(combined_frame, '1. Shaky', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_frame, '2. Stabilized', (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_frame, f'Frame: {i}/{T}', (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            writer.append_data(combined_frame)
            
    print(f"  - Saved heatmap GIF to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Debug script for leak detection and stabilization.")
    parser.add_argument("--video_path", required=True, help="Path to the single .mat video file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output visualizations.")
    parser.add_argument("--crop_size", type=int, default=16, help="The size of the square crop.")
    parser.add_argument("--num_leaks", type=int, default=2, help="Number of leaks to detect.")
    parser.add_argument("--heat_power", type=float, default=1.4, help="Exponent for the heat map fusion.")
    parser.add_argument("--shift1_frame", type=int, default=75, help="Frame to start first shift.")
    parser.add_argument("--shift1_x", type=float, default=-5.0, help="First X shift (%%).")
    parser.add_argument("--shift1_y", type=float, default=0.0, help="First Y shift (%%).")
    parser.add_argument("--shift2_frame", type=int, default=150, help="Frame to start second shift.")
    parser.add_argument("--shift2_x", type=float, default=10.0, help="Second X shift (relative, %%).")
    parser.add_argument("--shift2_y", type=float, default=0.0, help="Second Y shift (relative, %%).")
    args = parser.parse_args()

    print(f"--- Running Full Detection & Stabilization Validation ---")
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.video_path))[0]

    try:
        original_frames = scipy.io.loadmat(args.video_path)['TempFrames'].astype(np.float64)
        original_frames = original_frames[:,:,:150]
        print(f"  - Loaded video with shape: {original_frames.shape}")
    except Exception as e:
        sys.exit(f"FATAL: Could not load video file. Error: {e}")

    shaky_frames= add_gradual_shake(original_frames, args)
    stabilized_frames= stabilize_video_phase_correlation(shaky_frames)

    videos_to_test = {"Original": original_frames, "Shaky": shaky_frames, "Stabilized": stabilized_frames}
    results = {}
    for name, frames in videos_to_test.items():
        print(f"\n--- Detecting leaks in '{name}' video ---")
        score_map = calculate_temporal_trend_map(frames) * (calculate_local_heat_z_score(frames) ** args.heat_power)
        min_score_threshold = score_map.max() * 0.05
        leaks = find_top_n_leaks(score_map, args.num_leaks, min_distance=50, threshold_abs=min_score_threshold)
        results[name] = leaks
        if leaks:
            print(f"  - Found {len(leaks)} leak(s):")
            for i, leak in enumerate(leaks):
                print(f"    - Leak #{i+1}: Coords={leak['coords']}, Score={leak['score']:.4f}")
        else:
            print("  - No leaks found.")

    # --- Generate Visualizations ---
    # 1. Static 6-panel plot (unchanged)
    plot_save_path = os.path.join(args.output_dir, f"{base_filename}_full_comparison.png")
    save_full_comparison_plot(original_frames, shaky_frames, stabilized_frames,
                              results["Original"], results["Shaky"], results["Stabilized"],
                              args.crop_size, plot_save_path)
                              
    # 2. Grayscale 3-panel GIF (unchanged)
    gif_save_path = os.path.join(args.output_dir, f"{base_filename}_comparison_grayscale.gif")
    create_comparison_gif(original_frames, shaky_frames, stabilized_frames, gif_save_path)
    
    # --- NEW: Call the heatmap GIF function ---
    # 3. Heatmap 2-panel GIF
    heatmap_gif_save_path = os.path.join(args.output_dir, f"{base_filename}_comparison_heatmap.gif")
    create_heatmap_gif(shaky_frames, stabilized_frames, heatmap_gif_save_path)
    
    print("\n--- Validation Script Finished Successfully ---")

if __name__ == "__main__":
    main()

"""
python src_cnn_v2/stabilize_detect_leak_hole.py \
  --video_path "/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0805_2025_noshutter/T1.6V_2025-08-05-19-57-24_20_34_14_.mat" \
  --output_dir "debug_stabilized_outputs/Fluke_BrickCladding_2holes_0805_2025_noshutter/vid-1-stablisied" \
  --crop_size 16 \
  --num_leaks 2 \
  --shift1_frame 75 --shift1_x -10 --shift1_y 0 \
  --shift2_frame 150 --shift2_x 0 --shift2_y -10

python src_cnn_v2/stabilize_detect_leak_hole.py \
  --video_path "/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_HardyBoard_08132025_2holes_noshutter/T1.4V_2025-08-14-15-47-12_21_34_13_.mat" \
  --output_dir "debug_outputs/Fluke_HardyBoard_08132025_2holes_noshutter/vid-1" \
  --crop_size 16 \
  --num_leaks 2
"""