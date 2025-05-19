# visualize_thermal_data.py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import cv2 # For bounding box and potentially video writing
import os
import argparse
import sys
import config # For MAT_FRAMES_KEY and DATASET_FOLDER

# --- Configuration ---
DEFAULT_FPS_DISPLAY = 5  # For saving animations
DEFAULT_PADDING_ZOOM = 10 # Pixels padding around mask for zoomed video
DEFAULT_HOTSPOT_PIXEL_COUNT = 1
DEFAULT_BACKGROUND_PIXEL_COUNT = 1

def load_mat_and_mask(mat_filepath, mask_filepath, mat_key):
    """Loads .mat video and .npy mask."""
    if not os.path.exists(mat_filepath):
        print(f"Error: MAT file not found: {mat_filepath}")
        return None, None
    if not os.path.exists(mask_filepath):
        print(f"Error: Mask file not found: {mask_filepath}")
        return None, None

    try:
        mat_data = scipy.io.loadmat(mat_filepath)
        frames = mat_data[mat_key].astype(np.float64)
        if frames.ndim != 3 or frames.shape[2] < 1:
            raise ValueError("Invalid frame data in .mat file.")
    except Exception as e:
        print(f"Error loading MAT file {mat_filepath}: {e}")
        return None, None

    try:
        mask = np.load(mask_filepath)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if mask.ndim != 2:
            raise ValueError("Invalid mask dimensions.")
        if mask.shape != frames.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} mismatch with frame shape {frames.shape[:2]}.")
    except Exception as e:
        print(f"Error loading mask file {mask_filepath}: {e}")
        return None, None

    return frames, mask

def get_mask_bounding_box(mask, padding=0):
    """Finds the bounding box of True values in a boolean mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None # No True values in mask

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding
    rmin = max(0, rmin - padding)
    cmin = max(0, cmin - padding)
    rmax = min(mask.shape[0] - 1, rmax + padding)
    cmax = min(mask.shape[1] - 1, cmax + padding)

    return rmin, rmax, cmin, cmax

def create_zoomed_masked_video(frames, mask, output_filename="zoomed_hotspot_video.mp4",
                               padding=DEFAULT_PADDING_ZOOM, fps=DEFAULT_FPS_DISPLAY):
    """Creates and saves a video focusing on the masked area."""
    bbox = get_mask_bounding_box(mask, padding)
    if bbox is None:
        print("No hotspot found in mask, cannot create zoomed video.")
        return

    rmin, rmax, cmin, cmax = bbox
    height, width = rmax - rmin + 1, cmax - cmin + 1

    if height <= 0 or width <= 0:
        print("Invalid bounding box dimensions after padding, cannot create zoomed video.")
        return

    # Ensure mask is uint8 for drawing contours or overlaying
    mask_uint8 = mask.astype(np.uint8) * 255

    # --- Using Matplotlib Animation ---
    fig, ax = plt.subplots(figsize=(width/50, height/50)) # Adjust figure size
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    ims = []
    for i in range(frames.shape[2]):
        frame_data = frames[:, :, i]
        cropped_frame = frame_data[rmin:rmax+1, cmin:cmax+1]
        cropped_mask = mask_uint8[rmin:rmax+1, cmin:cmax+1] # For potential overlay

        # Normalize for display
        display_frame_norm = cv2.normalize(cropped_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        display_frame_color = cv2.cvtColor(display_frame_norm, cv2.COLOR_GRAY2BGR)

        # Overlay mask (optional, simple alpha blend)
        # You can make this more sophisticated, e.g., drawing contours
        overlay_color = np.zeros_like(display_frame_color)
        overlay_color[cropped_mask[rmin:rmax+1, cmin:cmax+1] > 0] = [0, 255, 0] # Green mask
        blended = cv2.addWeighted(overlay_color, 0.3, display_frame_color, 0.7, 0)

        im = ax.imshow(blended, animated=True) # Use blended or display_frame_color
        ims.append([im])

    if not ims:
        print("No frames processed for animation.")
        return

    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True, repeat_delay=1000)
    try:
        ani.save(output_filename, writer='ffmpeg', fps=fps)
        print(f"Zoomed video saved to {output_filename}")
    except Exception as e:
        print(f"Error saving animation (ensure ffmpeg is installed and in PATH): {e}")
    plt.close(fig)


def plot_pixel_time_series(frames, mask, mat_filename, num_hotspot_pixels=DEFAULT_HOTSPOT_PIXEL_COUNT,
                           num_background_pixels=DEFAULT_BACKGROUND_PIXEL_COUNT, fps_data=5.0):
    """Plots temperature time series for selected hotspot and background pixels."""
    H, W, T = frames.shape
    time_vector = np.arange(T) / fps_data

    hotspot_coords = np.argwhere(mask)
    background_coords = np.argwhere(~mask)

    if hotspot_coords.size == 0:
        print("No hotspot pixels found in mask.")
        return
    if background_coords.size == 0:
        print("No background pixels found (mask covers entire image?).")
        return

    plt.figure(figsize=(12, 6))
    plt.title(f"Pixel Temperature Time Series for {os.path.basename(mat_filename)}")
    plt.xlabel(f"Time (seconds, data fps={fps_data})")
    plt.ylabel("Temperature (°C or arb. units)")

    # Plot hotspot pixels
    for i in range(min(num_hotspot_pixels, len(hotspot_coords))):
        # Pick somewhat randomly or first few
        idx = np.random.choice(len(hotspot_coords)) if len(hotspot_coords) > num_hotspot_pixels else i
        r, c = hotspot_coords[idx]
        plt.plot(time_vector, frames[r, c, :], label=f'Hotspot Pixel ({r},{c})', alpha=0.8)

    # Plot background pixels
    for i in range(min(num_background_pixels, len(background_coords))):
        idx = np.random.choice(len(background_coords)) if len(background_coords) > num_background_pixels else i
        r, c = background_coords[idx]
        plt.plot(time_vector, frames[r, c, :], label=f'Background Pixel ({r},{c})', linestyle='--', alpha=0.8)

    plt.legend()
    plt.grid(True)
    plot_filename = f"pixel_series_{os.path.splitext(os.path.basename(mat_filename))[0]}.png"
    plt.savefig(plot_filename)
    print(f"Time series plot saved to {plot_filename}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize thermal data: zoomed video or pixel time series.")
    parser.add_argument("mat_file", help="Path to the .mat video file.")
    parser.add_argument("mask_file", help="Path to the corresponding _mask.npy file.")
    parser.add_argument("--mode", choices=['video', 'timeseries', 'both'], default='both',
                        help="Visualization mode: 'video' for zoomed video, 'timeseries', or 'both'.")
    parser.add_argument("--mat_key", default=config.MAT_FRAMES_KEY, help="Key for frame data in .mat file.")
    parser.add_argument("--fps_display", type=float, default=DEFAULT_FPS_DISPLAY, help="FPS for the output video animation.")
    parser.add_argument("--fps_data", type=float, default=5.0, help="Original FPS of the data (for time axis in timeseries).")
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING_ZOOM, help="Padding around mask for zoomed video.")
    parser.add_argument("--hotspot_pixels", type=int, default=DEFAULT_HOTSPOT_PIXEL_COUNT, help="Number of hotspot pixels for time series plot.")
    parser.add_argument("--background_pixels", type=int, default=DEFAULT_BACKGROUND_PIXEL_COUNT, help="Number of background pixels for time series plot.")
    parser.add_argument("--output_video_name", default="zoomed_hotspot_video.mp4", help="Filename for the output zoomed video.")


    if len(sys.argv) < 3 : # mat_file and mask_file are positional
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    frames, mask = load_mat_and_mask(args.mat_file, args.mask_file, args.mat_key)

    if frames is None or mask is None:
        sys.exit(1)

    output_dir = "visualized_thermal_data" # You can make this configurable
    os.makedirs(output_dir, exist_ok=True)


    if args.mode == 'video' or args.mode == 'both':
        video_save_path = os.path.join(output_dir, f"zoom_{os.path.splitext(os.path.basename(args.mat_file))[0]}.mp4")
        create_zoomed_masked_video(frames, mask, output_filename=video_save_path,
                                   padding=args.padding, fps=args.fps_display)

    if args.mode == 'timeseries' or args.mode == 'both':
        # Change CWD for saving plots or prefix with output_dir
        original_cwd = os.getcwd()
        os.chdir(output_dir)
        plot_pixel_time_series(frames, mask, args.mat_file,
                               num_hotspot_pixels=args.hotspot_pixels,
                               num_background_pixels=args.background_pixels,
                               fps_data=args.fps_data)
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()