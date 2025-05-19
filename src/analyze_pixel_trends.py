# analyze_pixel_trends.py
"""
Script to analyze a SINGLE IR video recording (.mat file), generate a pixel
temperature time series plot for a representative/specified pixel, and store
this plot. Helps in observing the trend (heating/cooling) for setting
'envir_para' or understanding individual file behavior.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import argparse
import sys
import traceback # Kept for debugging if needed

try:
    import config # Assumes config.py is in the same directory or Python path
except ImportError:
    print("Warning: Failed to import config.py. MAT_FRAMES_KEY might use default.")
    class MockConfig: # Basic fallback
        MAT_FRAMES_KEY = "TempFrames"
    config = MockConfig()

def plot_pixel_time_series_for_file(frames_raw, fps, focus_duration_sec,
                                    pixel_r, pixel_c, smooth_window,
                                    output_plot_path, base_filename_for_title):
    """
    Generates and saves a plot of temperature time series for a specific pixel
    from the provided raw frames.
    """
    H, W, num_frames = frames_raw.shape
    
    # Ensure pixel coordinates are within bounds (already done before calling, but good practice)
    pixel_r = int(min(max(0, pixel_r), H - 1))
    pixel_c = int(min(max(0, pixel_c), W - 1))

    # Determine focus duration in frames, ensuring it's valid
    focus_duration_frames = min(int(focus_duration_sec * fps), num_frames)
    if focus_duration_frames < 2 : # Need at least 2 frames for any focus plot
        print(f"Warning: Focus duration ({focus_duration_sec}s at {fps}fps = {focus_duration_frames} frames) is too short. Will only plot full duration.")
        focus_duration_frames = 0 # effectively disable focus plot part if too short

    time_vector_full = (np.arange(num_frames) / fps).astype(np.float64)
    
    series_full_raw = frames_raw[pixel_r, pixel_c, :]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False) # Create 2 subplots
    ax1, ax2 = axes.ravel() # Unpack axes

    # Plot 1: Raw Full Duration
    ax1.plot(time_vector_full, series_full_raw, label=f'Pixel ({pixel_r},{pixel_c}) - Raw Full')
    ax1.set_xlabel(f"Time (s) - Full Duration (Total {num_frames/fps:.2f}s)")
    ax1.set_ylabel("Raw Temperature") # Assuming °C or similar unit
    ax1.set_title(f"Raw Full Duration Trend for Pixel ({pixel_r},{pixel_c})")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Raw Focus Duration (and optional smoothed version)
    if focus_duration_frames >= 2:
        time_vector_focus = (np.arange(focus_duration_frames) / fps).astype(np.float64)
        series_focus_raw = frames_raw[pixel_r, pixel_c, :focus_duration_frames]
        ax2.plot(time_vector_focus, series_focus_raw, label=f'Pixel ({pixel_r},{pixel_c}) - Raw Focus')

        if smooth_window > 1 and focus_duration_frames >= smooth_window:
            series_to_smooth = series_focus_raw
            # convolve mode 'valid' returns an array of length max(M, N) - min(M, N) + 1.
            # For a moving average, M = len(series_to_smooth), N = smooth_window.
            # Length = len(series_to_smooth) - smooth_window + 1
            if len(series_to_smooth) >= smooth_window :
                smoothed_focus_series = np.convolve(series_to_smooth, np.ones(smooth_window)/smooth_window, mode='valid')
                
                # To plot this correctly against time_vector_focus, we need to align it.
                # The center of the first window of convolution aligns with index (smooth_window-1)//2 of original.
                start_idx_for_smoothed_time = (smooth_window - 1) // 2
                end_idx_for_smoothed_time = start_idx_for_smoothed_time + len(smoothed_focus_series)
                
                if end_idx_for_smoothed_time <= len(time_vector_focus):
                    time_vector_for_smoothed = time_vector_focus[start_idx_for_smoothed_time:end_idx_for_smoothed_time]
                    ax2.plot(time_vector_for_smoothed, smoothed_focus_series,
                             label=f'Smoothed (win={smooth_window})', linestyle='--')
                else:
                    print(f"  Warning: Smoothed series alignment issue for pixel ({pixel_r},{pixel_c}). Plotting raw focus only.")
            else:
                print(f"  Warning: Not enough data points in focus window ({len(series_to_smooth)}) to smooth with window {smooth_window} for pixel ({pixel_r},{pixel_c}).")


        ax2.set_xlabel(f"Time (s) - Focus Duration ({focus_duration_sec}s / {focus_duration_frames} frames)")
        ax2.set_ylabel("Raw Temperature")
        ax2.set_title(f"Initial Trend During Focus (Smoothed if win={smooth_window}>1)")
        ax2.legend()
        ax2.grid(True)
    else: # If focus_duration_frames was too short
        ax2.text(0.5, 0.5, "Focus duration too short to plot.", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title("Focus Duration Plot (Not Generated)")
    
    fig.suptitle(f"Pixel Temperature Time Series for: {base_filename_for_title}\nPixel: ({pixel_r},{pixel_c})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    try:
        plt.savefig(output_plot_path)
        print(f"  Saved time series plot to: {output_plot_path}")
    except Exception as e:
        print(f"  Error saving plot {output_plot_path}: {e}")
    plt.close(fig)


def analyze_single_video_file(mat_filepath, output_plot_dir, mat_key, fps,
                              focus_duration_sec, roi_border_percent, smooth_window,
                              fixed_coords=None):
    """
    Loads a single .mat file, determines pixel coordinates, and generates its time series plot.
    """
    base_filename = os.path.splitext(os.path.basename(mat_filepath))[0]
    print(f"\n--- Analyzing File: {base_filename} ---")
    print(f"  Output Directory: {output_plot_dir}")
    print(f"  Parameters: FPS={fps}, FocusDuration={focus_duration_sec}s, ROI Border={roi_border_percent*100 if roi_border_percent is not None else 'None'}%, SmoothWin={smooth_window}")
    if fixed_coords:
        print(f"  Using fixed coordinates for plotting: ROW={fixed_coords[0]}, COL={fixed_coords[1]}")
    
    os.makedirs(output_plot_dir, exist_ok=True) # Ensure output directory exists

    try:
        mat_data = scipy.io.loadmat(mat_filepath)
        frames_raw = mat_data[mat_key].astype(np.float64)
        if frames_raw.ndim != 3 or frames_raw.shape[2] < 2:
            print(f"  Error: Invalid frame data in {mat_filepath}. Skipping.")
            return False # Indicate failure
        H, W, _ = frames_raw.shape

        pixel_r_to_plot, pixel_c_to_plot = H // 2, W // 2 # Default to center

        if fixed_coords:
            pixel_r_to_plot, pixel_c_to_plot = fixed_coords[0], fixed_coords[1]
            print(f"  Using fixed pixel coordinates: ({pixel_r_to_plot}, {pixel_c_to_plot})")
        elif roi_border_percent is not None and 0.0 <= roi_border_percent < 0.5:
            border_h = int(H * roi_border_percent)
            border_w = int(W * roi_border_percent)
            # Center of the ROI
            if H - 2 * border_h > 0 and W - 2 * border_w > 0:
                pixel_r_to_plot = border_h + (H - 2 * border_h) // 2
                pixel_c_to_plot = border_w + (W - 2 * border_w) // 2
                print(f"  Using center of ROI ({roi_border_percent*100}%) pixel: ({pixel_r_to_plot}, {pixel_c_to_plot})")

            else:
                print(f"  Warning: ROI border too large for {mat_filepath}, using full frame center pixel: ({pixel_r_to_plot}, {pixel_c_to_plot}).")
        else:
             print(f"  Using center of full frame pixel: ({pixel_r_to_plot}, {pixel_c_to_plot})")
        
        # Ensure coordinates are valid
        pixel_r_to_plot = int(max(0, min(pixel_r_to_plot, H - 1)))
        pixel_c_to_plot = int(max(0, min(pixel_c_to_plot, W - 1)))

        plot_filename = f"{base_filename}_pixel_({pixel_r_to_plot}-{pixel_c_to_plot})_timeseries.png"
        output_plot_path = os.path.join(output_plot_dir, plot_filename)

        plot_pixel_time_series_for_file(frames_raw, fps, focus_duration_sec,
                                        pixel_r_to_plot, pixel_c_to_plot,
                                        smooth_window, output_plot_path, base_filename)
        return True # Indicate success

    except Exception as e:
        print(f"  Error processing {mat_filepath}: {e}")
        traceback.print_exc()
        return False # Indicate failure

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a pixel temperature time series plot for a SINGLE .mat file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("mat_file", help="Path to the specific .mat file to analyze.")
    parser.add_argument("output_plot_dir", help="Directory where the output plot will be saved.")
    
    parser.add_argument("--mat_key", default=getattr(config, 'MAT_FRAMES_KEY', 'TempFrames'),
                        help="Key in .mat file for temperature frames.")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frames per second of the original recording (for time axis calculation).")
    parser.add_argument("--focus_duration_sec", type=float, default=5.0,
                        help="Duration (seconds) of the initial 'focus' window to also plot in detail.")
    parser.add_argument("--roi_border_percent", type=float, default=None, # Default to None, meaning use full center or fixed_coords
                        help="Optional: Percent border (0.0-0.49) to define a central ROI for pixel selection. "
                             "If not provided or invalid, center of full frame is used unless --pixel_coords is set.")
    parser.add_argument("--smooth_window", type=int, default=1,
                        help="Temporal smoothing window for the focus duration plot (1 = no smoothing).")
    parser.add_argument("--pixel_coords", type=int, nargs=2, default=None, metavar=('ROW', 'COL'),
                        help="Optional: Fixed (ROW, COL) coordinates of the pixel to plot. Overrides ROI/center logic.")

    if len(sys.argv) < 3: # mat_file and output_plot_dir are mandatory
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if not os.path.isfile(args.mat_file):
        print(f"Error: MAT file not found at {args.mat_file}")
        sys.exit(1)
    
    # Validate ROI border percent
    valid_roi_border = None
    if args.roi_border_percent is not None:
        if 0.0 <= args.roi_border_percent < 0.5:
            valid_roi_border = args.roi_border_percent
        else:
            print(f"Warning: Invalid roi_border_percent ({args.roi_border_percent}). Will use full frame center or fixed_coords if provided.")
            
    if args.smooth_window < 1:
        print("Warning: Smooth window must be >= 1. Setting to 1 (no smoothing).")
        args.smooth_window = 1
    if args.fps <= 0:
        print("Error: FPS must be positive.")
        sys.exit(1)
    if args.focus_duration_sec < 0: # Allow 0 to effectively disable focus plot part
        print("Warning: Focus duration cannot be negative. Setting to 0.")
        args.focus_duration_sec = 0


    analyze_single_video_file(
        mat_filepath=args.mat_file,
        output_plot_dir=args.output_plot_dir,
        mat_key=args.mat_key,
        fps=args.fps,
        focus_duration_sec=args.focus_duration_sec,
        roi_border_percent=valid_roi_border, # Pass validated or None
        smooth_window=args.smooth_window,
        fixed_coords=args.pixel_coords
    )

    print("\nPixel trend analysis for single file finished.")