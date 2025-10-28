# scripts/generate_presentation_visuals.py
"""
Generates a suite of high-quality visuals for a single video to explain
the automated segmentation pipeline for presentations.

Includes contrast enhancement (CLAHE) for clearer raw frame visualization.
"""
import os
import sys
import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import kendalltau
import imageio # For creating GIFs
import cv2 # Import OpenCV for contrast enhancement

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Calculation Functions (simplified for this script) ---
def _calculate_kendall_for_row(row_data, t):
    W = row_data.shape[0]
    row_values = np.zeros(W, dtype=np.float64)
    for c in range(W):
        pixel_series = row_data[c, :]
        if not np.any(np.isnan(pixel_series)) and len(pixel_series) > 1:
            try:
                val, _ = kendalltau(t, pixel_series)
            except (ValueError, IndexError):
                val = 0.0
            row_values[c] = val if np.isfinite(val) else 0.0
    return row_values

def calculate_activity_map_parallel(frames):
    H, W, T = frames.shape
    if T < 2: return np.zeros((H, W), dtype=np.float64)
    t = np.arange(T)
    desc = "Calculating Kendall Tau Activity Map"
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_kendall_for_row)(frames[r, :, :], t) for r in tqdm(range(H), desc=desc, ncols=100)
    )
    return np.vstack(results)

# --- NEW: Helper function for contrast enhancement ---
def enhance_frame(frame):
    """Applies CLAHE to a single frame for better visualization."""
    # Normalize frame to 0-255 range for 8-bit processing
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Create a CLAHE object (these are good default values)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE
    enhanced_frame = clahe.apply(frame_normalized)
    
    return enhanced_frame

def main(args):
    video_path = args.video_path
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Generating visuals for: {base_filename} ---")

    try:
        mat_data = scipy.io.loadmat(video_path)
        frames = mat_data[args.mat_key].astype(np.float64)
    except Exception as e:
        print(f"Error loading {video_path}: {e}", file=sys.stderr)
        return

        # --- 1. Save ENHANCED First and Last Frames Side-by-Side (Grayscale + Inferno) ---

    first_frame = frames[:, :, 0]
    last_frame = frames[:, :, -1]

    enhanced_first = enhance_frame(first_frame)
    enhanced_last = enhance_frame(last_frame)

    # --- 1a. Grayscale version ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(enhanced_first, cmap='gray')
    axes[0].set_title("First Frame")
    axes[0].axis('off')

    axes[1].imshow(enhanced_last, cmap='gray')
    axes[1].set_title("Last Frame")
    axes[1].axis('off')

    gray_output_path = os.path.join(args.output_dir, f"{base_filename}_01a_raw_first_last_gray.png")
    plt.suptitle("Step 1: Raw IR Frames - Grayscale")
    plt.tight_layout()
    plt.savefig(gray_output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved side-by-side grayscale frame to: {gray_output_path}")

    # --- 1b. Inferno version ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(enhanced_first, cmap='inferno')
    axes[0].set_title("First Frame")
    axes[0].axis('off')

    axes[1].imshow(enhanced_last, cmap='inferno')
    axes[1].set_title("Last Frame")
    axes[1].axis('off')

    inferno_output_path = os.path.join(args.output_dir, f"{base_filename}_01b_raw_first_last_inferno.png")
    plt.suptitle("Step 1: Raw IR Frames - Inferno")
    plt.tight_layout()
    plt.savefig(inferno_output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved side-by-side inferno frame to: {inferno_output_path}")


    # --- 3. Save ENHANCED Video GIF ---
    gif_path = os.path.join(args.output_dir, f"{base_filename}_02_raw_video_enhanced.gif")
    clip_to_enhance = frames[:, :, :args.gif_frames]
    
    enhanced_frames_for_gif = []
    for i in range(clip_to_enhance.shape[2]):
        enhanced_frames_for_gif.append(enhance_frame(clip_to_enhance[:, :, i]))
        
    imageio.mimsave(gif_path, enhanced_frames_for_gif, fps=args.fps)
    print(f"Saved enhanced video GIF to: {gif_path}")

    # --- 4. Calculate and Save the Activity Map ---
    activity_map = calculate_activity_map_parallel(frames)
    activity_map[activity_map < 0] = 0
    
    activity_map_path = os.path.join(args.output_dir, f"{base_filename}_03_activity_map.png")
    plt.figure(figsize=(8, 6))
    plt.imshow(activity_map, cmap='hot')
    plt.title("Step 2: Thermal Activity Map")
    plt.axis('off')
    plt.savefig(activity_map_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved activity map to: {activity_map_path}")
    
    print("\n--- Visuals Generated Successfully ---")
    print(f"Please now run the `run_SAM.py` script on this video to generate the final two visuals (prompt map and segmentation).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a suite of visuals for a single video to explain the segmentation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("video_path", help="Path to the SINGLE .mat video file.")
    parser.add_argument("output_dir", help="Directory to save the generated visual assets.")
    
    parser.add_argument("--mat_key", default='TempFrames', help="Key in .mat file for temperature frames.")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second for GIF generation.")
    parser.add_argument("--gif_frames", type=int, default=50, help="Number of frames to include in the GIF (e.g., 50 frames at 5fps = 10 seconds).")
    
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Error: Video file not found at {args.video_path}", file=sys.stderr)
        sys.exit(1)
        
    main(args)

"""

python -m scripts.vis-scripts.generate_vis \
    datasets/Fluke_BrickCladding_2holes_0616_2025_noshutter/T1.4V_2.2Pa_2025-6-16-16-33-25_20_34_14_.mat \
    presentation_assets
python -m scripts.vis-scripts.generate_vis \
    datasets/Fluke_Gypsum_07252025_noshutter/T1.6V_2025-07-28-17-36-55_20_34_14_.mat \
    presentation_assets
    
"""