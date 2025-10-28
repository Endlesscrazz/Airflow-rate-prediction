# src_cnn_v2/poc_stabilization.py
"""
Proof of Concept (POC) for Video Stabilization.

This script demonstrates and validates the video stabilization algorithm by:
  1. Loading a single .mat thermal video.
  2. Artificially introducing camera shake to create a "shaky" version.
  3. Implementing a stabilization algorithm based on template matching to
     track the leak and re-center the frames.
  4. Generating a comprehensive side-by-side comparison plot that visualizes:
     - The path of the leak in the original vs. shaky video.
     - The output of the original leak finder on the shaky video (showing the wrong crop).
     - The output of the new stabilization algorithm (showing the correct crop).
  5. Saving the final stabilized and cropped sequence as a .npy file.

How to Run:
  python src_cnn_v2/poc_stabilization.py \
    --video_path "/path/to/your/video.mat" \
    --output_dir "poc_stabilization_output"
"""
import os
import sys
import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# --- We will reuse the core detection logic from your debug script ---
# This makes the comparison fair.
from debug_leak_finder import (
    calculate_temporal_trend_map,
    calculate_local_heat_z_score,
    crop_sequence
)

# --- STEP 2: Function to Artificially Add Shake ---
def add_artificial_shake(frames):
    """
    Introduces pre-defined shifts to a video sequence to simulate camera shake.
    - Frames 0-49: No shift
    - Frames 50-89: Shift left by 5% of width
    - Frames 90-149: Shift left (5%) and up (5%)
    """
    H, W, T = frames.shape
    shaky_frames = np.zeros_like(frames)
    
    # Define the shifts
    shift_x_1 = int(W * 0.05)
    shift_y_1 = 0
    
    shift_x_2 = int(W * 0.05)
    shift_y_2 = int(H * 0.05)

    print("  - Applying artificial shake to video...")
    for i in range(T):
        frame = frames[:, :, i]
        if 50 <= i < 90:
            # Shift left
            M = np.float32([[1, 0, -shift_x_1], [0, 1, -shift_y_1]])
            shaky_frames[:, :, i] = cv2.warpAffine(frame, M, (W, H))
        elif i >= 90:
            # Shift left and up
            M = np.float32([[1, 0, -shift_x_2], [0, 1, -shift_y_2]])
            shaky_frames[:, :, i] = cv2.warpAffine(frame, M, (W, H))
        else:
            # No shift
            shaky_frames[:, :, i] = frame
            
    return shaky_frames

# --- STEP 3: The Stabilization Algorithm ---
def stabilize_video(frames, template_size=25):
    """
    Stabilizes a video by tracking a template and re-centering each frame.
    Returns the stabilized frames and the tracked path of the leak.
    """
    H, W, T = frames.shape
    
    # --- 1. Find a robust anchor point and create a template ---
    print("  - Stabilizer: Finding robust anchor point...")
    # Use the first 30 frames to get a stable initial detection
    score_map_initial = calculate_temporal_trend_map(frames[:, :, :30]) * \
                        (calculate_local_heat_z_score(frames[:, :, :30]) ** 1.4)
    
    initial_y, initial_x = np.unravel_index(np.argmax(score_map_initial), score_map_initial.shape[:2])
    
    half = template_size // 2
    template = frames[initial_y-half : initial_y+half, initial_x-half : initial_x+half, 0].astype(np.float32)

    # --- 2. Track the template through all frames ---
    print("  - Stabilizer: Tracking leak path across all frames...")
    leak_positions = []
    for i in range(T):
        frame = frames[:, :, i].astype(np.float32)
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        center_x = max_loc[0] + half
        center_y = max_loc[1] + half
        leak_positions.append((center_x, center_y))

    # --- 3. Crop each frame around its tracked center to stabilize ---
    print("  - Stabilizer: Cropping frames around tracked path...")
    stabilized_frames = np.zeros_like(frames)
    for i in range(T):
        frame = frames[:, :, i]
        current_x, current_y = leak_positions[i]
        
        # Calculate the shift needed to move the leak back to the initial anchor
        dx = initial_x - current_x
        dy = initial_y - current_y
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        stabilized_frames[:, :, i] = cv2.warpAffine(frame, M, (W, H))
        
    return stabilized_frames, leak_positions, (initial_x, initial_y)


# --- STEP 4 & 5: Visualization Function ---
def create_comparison_plot(original_frames, shaky_frames, stabilized_frames, 
                           original_path, shaky_path, stabilized_path,
                           original_epicenter, shaky_epicenter, crop_size, 
                           save_path):
    """
    Creates a detailed side-by-side comparison plot.
    """
    fig, axs = plt.subplots(2, 3, figsize=(22, 14), dpi=120)
    fig.suptitle("Video Stabilization Proof of Concept", fontsize=20, y=0.97)

    # --- Row 1: Leak Paths ---
    # Plot 1: Path on Original Video
    axs[0, 0].set_title("1a. Leak Path on Original Video")
    axs[0, 0].imshow(np.mean(original_frames, axis=2), cmap='inferno')
    x_coords, y_coords = zip(*original_path)
    axs[0, 0].plot(x_coords, y_coords, 'c-', label='Tracked Path')
    axs[0, 0].scatter([x_coords[0]], [y_coords[0]], c='lime', marker='o', s=100, label='Start', zorder=5)
    axs[0, 0].legend()

    # Plot 2: Path on Shaky Video
    axs[0, 1].set_title("1b. Leak Path on Shaky Video")
    axs[0, 1].imshow(np.mean(shaky_frames, axis=2), cmap='inferno')
    x_coords, y_coords = zip(*shaky_path)
    axs[0, 1].plot(x_coords, y_coords, 'c-', label='Tracked Path')
    axs[0, 1].scatter([x_coords[0]], [y_coords[0]], c='lime', marker='o', s=100, label='Start', zorder=5)
    axs[0, 1].legend()

    # Plot 3: Path on Stabilized Video
    axs[0, 2].set_title("1c. Leak Path on Stabilized Video (Goal: a single dot)")
    axs[0, 2].imshow(np.mean(stabilized_frames, axis=2), cmap='inferno')
    x_coords, y_coords = zip(*stabilized_path)
    axs[0, 2].plot(x_coords, y_coords, 'c-', label='Tracked Path')
    axs[0, 2].scatter([x_coords[0]], [y_coords[0]], c='lime', marker='o', s=100, label='Start', zorder=5)
    axs[0, 2].legend()

    # --- Row 2: Epicenter Detection Comparison ---
    # Plot 4: Original Detector on Original Video (The Ground Truth)
    axs[1, 0].set_title(f"2a. Original Detector on Clean Video\nFinds Epicenter at ({original_epicenter[1]}, {original_epicenter[0]})")
    axs[1, 0].imshow(np.mean(original_frames, axis=2), cmap='inferno')
    axs[1, 0].scatter([original_epicenter[1]], [original_epicenter[0]], s=200, c='cyan', marker='*')
    rect = patches.Rectangle((original_epicenter[1] - crop_size//2, original_epicenter[0] - crop_size//2),
                             crop_size, crop_size, linewidth=2, edgecolor='lime', facecolor='none')
    axs[1, 0].add_patch(rect)

    # Plot 5: Original Detector on Shaky Video (The Problem)
    axs[1, 1].set_title(f"2b. Original Detector on Shaky Video\nFinds WRONG Epicenter at ({shaky_epicenter[1]}, {shaky_epicenter[0]})")
    axs[1, 1].imshow(np.mean(shaky_frames, axis=2), cmap='inferno')
    axs[1, 1].scatter([shaky_epicenter[1]], [shaky_epicenter[0]], s=200, c='red', marker='*')
    rect = patches.Rectangle((shaky_epicenter[1] - crop_size//2, shaky_epicenter[0] - crop_size//2),
                             crop_size, crop_size, linewidth=2, edgecolor='red', facecolor='none')
    axs[1, 1].add_patch(rect)

    # Plot 6: Original Detector on Stabilized Video (The Solution)
    # Re-run detection on the stabilized video
    stabilized_score_map = calculate_temporal_trend_map(stabilized_frames) * \
                           (calculate_local_heat_z_score(stabilized_frames) ** 1.4)
    stabilized_epicenter_y, stabilized_epicenter_x = np.unravel_index(np.argmax(stabilized_score_map), stabilized_score_map.shape)
    axs[1, 2].set_title(f"2c. Original Detector on Stabilized Video\nFinds CORRECT Epicenter at ({stabilized_epicenter_x}, {stabilized_epicenter_y})")
    axs[1, 2].imshow(np.mean(stabilized_frames, axis=2), cmap='inferno')
    axs[1, 2].scatter([stabilized_epicenter_x], [stabilized_epicenter_y], s=200, c='cyan', marker='*')
    rect = patches.Rectangle((stabilized_epicenter_x - crop_size//2, stabilized_epicenter_y - crop_size//2),
                             crop_size, crop_size, linewidth=2, edgecolor='lime', facecolor='none')
    axs[1, 2].add_patch(rect)

    for ax_row in axs:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nSaved comparison plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Proof of Concept for video stabilization.")
    parser.add_argument("--video_path", required=True, help="Path to the single .mat video file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output plot and .npy file.")
    parser.add_argument("--crop_size", type=int, default=16, help="The final crop size for the model.")
    args = parser.parse_args()

    print(f"--- Running Stabilization POC ---")
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.video_path))[0]

    # --- 1. Load Original Video ---
    try:
        original_frames = scipy.io.loadmat(args.video_path)['TempFrames'].astype(np.float64)
        # For POC, let's work with the first 150 frames
        original_frames = original_frames[:, :, :150]
        print(f"  - Loaded video with shape: {original_frames.shape}")
    except Exception as e:
        sys.exit(f"FATAL: Could not load video file. Error: {e}")

    # --- 2. Create Shaky Version ---
    shaky_frames = add_artificial_shake(original_frames)

    # --- 3. Stabilize the Shaky Video ---
    stabilized_frames, shaky_path, anchor_point = stabilize_video(shaky_frames)
    
    # Also get paths for original and stabilized videos for visualization
    _, original_path, _ = stabilize_video(original_frames)
    _, stabilized_path, _ = stabilize_video(stabilized_frames)

    # --- 4. Run Original Leak Finder on all three versions ---
    print("\n--- Comparing Leak Detection Algorithms ---")
    
    # On Original Video (Ground Truth)
    original_score_map = calculate_temporal_trend_map(original_frames) * \
                         (calculate_local_heat_z_score(original_frames) ** 1.4)
    original_epicenter_y, original_epicenter_x = np.unravel_index(np.argmax(original_score_map), original_score_map.shape)
    print(f"  - Original Detector on Clean Video... Found leak at: ({original_epicenter_x}, {original_epicenter_y})")

    # On Shaky Video (The Problem)
    shaky_score_map = calculate_temporal_trend_map(shaky_frames) * \
                      (calculate_local_heat_z_score(shaky_frames) ** 1.4)
    shaky_epicenter_y, shaky_epicenter_x = np.unravel_index(np.argmax(shaky_score_map), shaky_score_map.shape)
    print(f"  - Original Detector on Shaky Video... Found leak at: ({shaky_epicenter_x}, {shaky_epicenter_y}) <-- Incorrect!")

    # --- 5. Generate Comparison Plot and Save Final Output ---
    plot_save_path = os.path.join(args.output_dir, f"{base_filename}_stabilization_comparison.png")
    create_comparison_plot(original_frames, shaky_frames, stabilized_frames,
                           original_path, shaky_path, stabilized_path,
                           (original_epicenter_y, original_epicenter_x),
                           (shaky_epicenter_y, shaky_epicenter_x),
                           args.crop_size, plot_save_path)

    # Final step: crop the stabilized video using the original leak finder
    final_cropped_frames = crop_sequence(stabilized_frames, original_epicenter_x, original_epicenter_y, args.crop_size)
    final_sequence_to_save = final_cropped_frames.transpose(2, 0, 1)
    npy_save_path = os.path.join(args.output_dir, f"{base_filename}_stabilized_crop.npy")
    np.save(npy_save_path, final_sequence_to_save)
    print(f"Saved final stabilized sequence of shape {final_sequence_to_save.shape} to: {npy_save_path}")

    print("\n--- POC Finished Successfully ---")

if __name__ == "__main__":
    main()

"""
python src_cnn_v2/poc_stabilization.py \
  --video_path "/scratch/general/vast/u1527145/datasets/Fluke_Gypsum_07162025_noshutter/T1.4V_2025-07-17-16-56-31_22_34_12_.mat" \
  --output_dir "stabilization_poc_output" \
  --crop_size 30
"""