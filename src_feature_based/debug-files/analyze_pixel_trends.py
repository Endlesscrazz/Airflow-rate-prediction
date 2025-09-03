# analyze_pixel_trends_interactive.py
"""
INTERACTIVE SCRIPT to analyze a SINGLE IR video. It:
1. Displays the first frame and the calculated activity map side-by-side.
2. Allows the user to click on points of interest on either image.
3. Generates and saves time-series plots for the selected pixels.
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import argparse
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import kendalltau, mstats

# --- Calculation Functions ---
def _calculate_value_for_row(row_data, t, method):
    """Helper for parallel calculation of Theil-Sen slope or Kendall's Tau."""
    W = row_data.shape[0]
    row_values = np.zeros(W, dtype=np.float64)
    for c in range(W):
        pixel_series = row_data[c, :]
        if not np.any(np.isnan(pixel_series)) and len(pixel_series) > 1:
            try:
                if method == 'theil_sen':
                    val, _, _, _ = mstats.theilslopes(pixel_series, t, 0.95)
                elif method == 'kendall_tau':
                    val, _ = kendalltau(t, pixel_series)
                else:
                    val = 0.0
            except (ValueError, IndexError):
                val = 0.0
            row_values[c] = val if np.isfinite(val) else 0.0
    return row_values

def calculate_activity_map_parallel(frames, method):
    """Calculates an activity map for each pixel in parallel using the specified method."""
    H, W, T = frames.shape
    if T < 2: return np.zeros((H, W), dtype=np.float64)
    t = np.arange(T)
    
    desc = f"Calculating {method.replace('_', ' ').title()}"
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_value_for_row)(frames[r, :, :], t, method) for r in tqdm(range(H), desc=desc, leave=False, ncols=100)
    )
    activity_map = np.vstack(results)
    return activity_map

# --- Plotting Functions ---
def plot_pixel_time_series(frames, fps, pixel_coords, output_path, base_filename):
    """Generates and saves a single plot with time series for all selected pixels."""
    plt.figure(figsize=(14, 7))
    time_vector_full = np.arange(frames.shape[2]) / fps
    
    for label, (r, c) in pixel_coords.items():
        series = frames[r, c, :]
        plt.plot(time_vector_full, series, label=f'{label} (Pixel: {r}, {c})')
        
    plt.xlabel(f"Time (s)")
    plt.ylabel("Raw Temperature")
    plt.title(f"Pixel Temperature Trend for: {base_filename}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  - Saved combined trend plot to: {os.path.basename(output_path)}")

# --- Main Interactive Logic ---
def main(args):
    video_path = args.video_path
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    video_output_dir = os.path.join(args.base_output_dir, base_filename)
    os.makedirs(video_output_dir, exist_ok=True)
    
    print(f"--- Analyzing: {base_filename} ---")
    print(f"Using activity method: {args.activity_method}")

    try:
        mat_data = scipy.io.loadmat(video_path)
        frames = mat_data[args.mat_key].astype(np.float64)
        print(f"  - Loaded video with shape: {frames.shape}")
    except Exception as e:
        print(f"  Error loading {video_path}: {e}. Aborting.", file=sys.stderr)
        return

    # 1. Calculate the activity map BEFORE showing the plot
    activity_map = calculate_activity_map_parallel(frames, args.activity_method)
    
    # 2. Interactively get pixel coordinates with side-by-side plots
    first_frame = frames[:, :, 0]
    selected_points = {}

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            col, row = int(event.xdata), int(event.ydata)
            label = f"Point_{len(selected_points) + 1}"
            selected_points[label] = (row, col)
            print(f"Selected {label}: (row={row}, col={col})")
            # Draw marker on both plots for visual confirmation
            ax1.plot(col, row, 'g+', markersize=12, markeredgewidth=2)
            ax2.plot(col, row, 'g+', markersize=12, markeredgewidth=2)
            fig.canvas.draw()

    # --- NEW: Side-by-side interactive plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Grayscale First Frame
    ax1.imshow(first_frame, cmap='gray')
    ax1.set_title("First Frame (Click to select points)")
    
    # Plot 2: Activity Map
    vmax = np.nanpercentile(np.abs(activity_map), 99.5)
    im = ax2.imshow(activity_map, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax2.set_title(f"Activity Map ({args.activity_method.replace('_', ' ').title()})")
    fig.colorbar(im, ax=ax2, label=f"Activity ({args.activity_method})")

    fig.suptitle("Click on points of interest on EITHER image, then close this window to generate plots.", fontsize=16)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if not selected_points:
        print("No points selected. Exiting.")
        return
    
    def plot_activity_map(slope_map, output_path, base_filename):
        plt.figure(figsize=(12, 9))
        vmax = np.nanpercentile(np.abs(slope_map), 99.5)
        vmin = -vmax
        plt.imshow(slope_map, cmap='coolwarm', vmin=vmin, vmax=vmax)
        plt.colorbar(label=f"Heating/Cooling Rate {args.activity_method}")
        plt.title(f"Thermal Activity Map for: {base_filename}")
        plt.savefig(output_path)
        plt.close()
        print(f"  - Saved activity map to: {os.path.basename(output_path)}")

    # 3. Generate and save the trend plot for all selected points
    full_slope_map = calculate_activity_map_parallel(frames, args.activity_method)
    map_output_path = os.path.join(video_output_dir, f"{base_filename}_activity_map.png")
    plot_activity_map(full_slope_map, map_output_path, base_filename)

    plot_output_path = os.path.join(video_output_dir, f"{base_filename}_selected_trends.png")
    plot_pixel_time_series(frames, args.fps, selected_points, plot_output_path, base_filename)
    
    print(f"\n--- Analysis complete for {base_filename}. Outputs are in '{video_output_dir}' ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively analyze a single .mat file to debug pixel trends and activity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("video_path", help="Path to the SINGLE .mat video file you want to analyze.")
    parser.add_argument("base_output_dir", help="Base directory to save outputs.")
    
    parser.add_argument("--mat_key", default='TempFrames', help="Key in .mat file for temperature frames.")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second of the recordings.")
    parser.add_argument("--activity_method", type=str, default="kendall_tau", choices=['theil_sen', 'kendall_tau'],
                        help="Method to use for calculating the activity map.")
    
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Error: Video file not found at {args.video_path}", file=sys.stderr)
        sys.exit(1)
        
    main(args)

"""
Fluke_BrickCladding_2holes_0808_2025_noshutter
    vid-6:
    python -m src_feature_based.debug-files.analyze_pixel_trends \
        datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter/T1.8V_2025-08-08-19-04-55_20_28_8_.mat \
        debug_ouputs/Fluke_BrickCladding_2holes_0808_2025_noshutter/vid-6
"""

"""
Fluke_BrickCladding_2holes_0616_2025_noshutter
    vid-6:
    python -m src_feature_based.debug-files.analyze_pixel_trends \
        datasets/Fluke_BrickCladding_2holes_0616_2025_noshutter/T1.4V_2.2Pa_2025-6-16-16-33-25_20_34_14_.mat \
        debug_ouputs/Fluke_BrickCladding_2holes_0616_2025_noshutter/vid-1
"""

"""
Fluke_BrickCladding_2holes_0805_2025_noshutter
    vid-6:
    python -m src_feature_based.debug-files.analyze_pixel_trends \
        datasets/Fluke_BrickCladding_2holes_0805_2025_noshutter/T1.6V_2025-08-05-20-22-42_20_26_6_.mat \
        debug_ouputs/Fluke_BrickCladding_2holes_0805_2025_noshutter/vid-3

Fluke_HardyBoard_08132025_2holes_noshutter
    vid-4
    python -m src_feature_based.debug-files.analyze_pixel_trends \
        /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_HardyBoard_08132025_2holes_noshutter/T1.4V_2025-08-15-18-45-57_21_32_11_.mat \
        debug_ouputs/Fluke_HardyBoard_08132025_2holes_noshutter/vid-4
"""