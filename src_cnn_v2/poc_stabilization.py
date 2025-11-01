# src_cnn_v2/poc_stabilization.py
"""
Proof of Concept (POC) for Global Video Stabilization. (Version 12 - Multi-Mode)

This script serves two purposes, controlled by the '--mode' flag:

1. --mode generate:
   - Takes a single clean .mat video as input.
   - Artificially introduces gradual camera shake.
   - Saves the new "shaky" video as a new .mat file.

2. --mode compare:
   - Takes TWO .mat videos as input: an original clean version and a shaky version.
   - Runs the stabilization algorithm on the shaky video.
   - Generates detailed visualizations comparing the results.

How to Run:

# MODE 1: Generate a shaky video
python src_cnn_v2/poc_stabilization.py \
  --mode generate \
  --video_path "/path/to/clean_video.mat" \
  --output_path "/path/to/save/shaky_video.mat"

# MODE 2: Compare a shaky video to its original
python src_cnn_v2/poc_stabilization.py \
  --mode compare \
  --video_path "/path/to/clean_video.mat" \
  --shaky_video_path "/path/to/shaky_video.mat" \
  --output_dir "comparison_output"
"""
import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- Core Functions ---

def add_gradual_shake(frames, args):
    H, W, T = frames.shape
    shaky_frames = np.zeros_like(frames)
    start_shift = (0.0, 0.0)
    shift1_target = (int(W * (args.shift1_x / 100.0)), int(H * (args.shift1_y / 100.0)))
    shift2_target = (shift1_target[0] + int(W * (args.shift2_x / 100.0)), 
                     shift1_target[1] + int(H * (args.shift2_y / 100.0)))
    total_x_shifts, total_y_shifts = [], []
    phase1_len = args.shift1_frame
    if phase1_len > 0:
        total_x_shifts.extend(list(np.linspace(start_shift[0], shift1_target[0], num=phase1_len)))
        total_y_shifts.extend(list(np.linspace(start_shift[1], shift1_target[1], num=phase1_len)))
    phase2_len = args.shift2_frame - args.shift1_frame
    if phase2_len > 0:
        total_x_shifts.extend(list(np.linspace(total_x_shifts[-1], shift2_target[0], num=phase2_len + 1))[1:])
        total_y_shifts.extend(list(np.linspace(total_y_shifts[-1], shift2_target[1], num=phase2_len + 1))[1:])
    phase3_len = T - args.shift2_frame
    if phase3_len > 0:
        total_x_shifts.extend([total_x_shifts[-1]] * phase3_len)
        total_y_shifts.extend([total_y_shifts[-1]] * phase3_len)
    print("  - Applying gradual artificial shake to video...")
    for i in range(T):
        M = np.float32([[1, 0, total_x_shifts[i]], [0, 1, total_y_shifts[i]]])
        shaky_frames[:, :, i] = cv2.warpAffine(frames[:, :, i], M, (W, H))
    actual_shifts = list(zip(total_x_shifts, total_y_shifts))
    return shaky_frames, actual_shifts

def stabilize_video_phase_correlation(shaky_frames):
    H, W, T = shaky_frames.shape
    stabilized_frames = np.zeros_like(shaky_frames)
    detected_shifts = [(0.0, 0.0)] 
    reference_frame = shaky_frames[:, :, 0].astype(np.float32)
    stabilized_frames[:, :, 0] = shaky_frames[:, :, 0]
    print("  - Stabilizing video using Phase Correlation...")
    for i in range(1, T):
        current_frame = shaky_frames[:, :, i].astype(np.float32)
        shift, _ = cv2.phaseCorrelate(reference_frame, current_frame)
        dx, dy = shift
        detected_shifts.append((-dx, -dy))
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stabilized_frames[:, :, i] = cv2.warpAffine(shaky_frames[:, :, i], M, (W, H))
    return stabilized_frames, detected_shifts

def find_and_apply_stable_crop(frames, shifts):
    H, W, T = frames.shape
    max_shift_right = max(0, max(dx for dx, dy in shifts))
    max_shift_left = max(0, max(-dx for dx, dy in shifts))
    max_shift_down = max(0, max(dy for dx, dy in shifts))
    max_shift_up = max(0, max(-dy for dx, dy in shifts))
    print("\n  - Maximum corrective shifts (L, R, U, D): "
          f"({max_shift_left:.1f}, {max_shift_right:.1f}, {max_shift_up:.1f}, {max_shift_down:.1f}) pixels")
    crop_x1 = int(np.ceil(max_shift_left))
    crop_x2 = int(W - np.ceil(max_shift_right))
    crop_y1 = int(np.ceil(max_shift_up))
    crop_y2 = int(H - np.ceil(max_shift_down))
    print(f"  - Applying stable crop at coordinates: (x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2})")
    return frames[crop_y1:crop_y2, crop_x1:crop_x2, :]

# --- Visualization Functions ---
def create_stabilization_gif(original_frames, shaky_frames, stabilized_frames, final_cropped_frames, save_path, fps=10):
    print(f"  - Creating 3-panel high-contrast grayscale GIF: {os.path.basename(save_path)}")
    H, W, T = shaky_frames.shape
    vmin = np.percentile(original_frames, 1)
    vmax = np.percentile(original_frames, 99)
    def normalize_for_viz_gray(frame, vmin, vmax):
        frame = np.clip(frame, vmin, vmax)
        frame_norm = (frame.astype(np.float32) - vmin) / (vmax - vmin)
        frame_8bit = (frame_norm * 255).astype(np.uint8)
        return cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2RGB)
    with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
        for i in range(T):
            orig_viz = normalize_for_viz_gray(original_frames[:,:,i], vmin, vmax)
            shaky_viz = normalize_for_viz_gray(shaky_frames[:, :, i], vmin, vmax)
            stabilized_viz = normalize_for_viz_gray(stabilized_frames[:, :, i], vmin, vmax)
            cropped_viz = normalize_for_viz_gray(final_cropped_frames[:, :, i], vmin, vmax)
            resized_crop = cv2.resize(cropped_viz, (W, H), interpolation=cv2.INTER_NEAREST)
            combined_frame = np.hstack((orig_viz, shaky_viz, stabilized_viz, resized_crop))
            cv2.putText(combined_frame, '1. Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_frame, '2. Shaky', (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combined_frame, '3. Stabilized', (2*W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(combined_frame, '4. Final Crop', (3*W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_frame, f'Frame: {i}/{T}', (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.append_data(combined_frame)
    print(f"  - Saved 4-panel grayscale GIF to: {save_path}")

def run_generate_mode(args):
    """Logic for creating a shaky video."""
    print(f"--- Running in GENERATE mode ---")
    try:
        mat_data = scipy.io.loadmat(args.video_path)
        original_frames = mat_data['TempFrames'].astype(np.float64)
        original_frames = original_frames[:, :, :args.num_frames]
        print(f"  - Loaded video with shape: {original_frames.shape}")
    except Exception as e:
        sys.exit(f"FATAL: Could not load video file. Error: {e}")

    shaky_frames, _ = add_gradual_shake(original_frames, args)
    
    # Save the new shaky video as a .mat file, preserving other data
    mat_data['TempFrames'] = shaky_frames
    scipy.io.savemat(args.output_path, mat_data)
    print(f"\nSuccessfully saved shaky video to: {args.output_path}")

def run_compare_mode(args):
    """Logic for stabilizing a shaky video and creating visualizations."""
    print(f"--- Running in COMPARE mode ---")
    try:
        original_frames = scipy.io.loadmat(args.video_path)['TempFrames'].astype(np.float64)
        original_frames = original_frames[:, :, :args.num_frames]
        
        shaky_frames = scipy.io.loadmat(args.shaky_video_path)['TempFrames'].astype(np.float64)
        shaky_frames = shaky_frames[:, :, :args.num_frames]
        
        print(f"  - Loaded original video shape: {original_frames.shape}")
        print(f"  - Loaded shaky video shape:  {shaky_frames.shape}")
    except Exception as e:
        sys.exit(f"FATAL: Could not load video files. Error: {e}")

    stabilized_frames, detected_shifts = stabilize_video_phase_correlation(shaky_frames)
    final_cropped_frames = find_and_apply_stable_crop(stabilized_frames, detected_shifts)
    
    print("\n--- Generating Visualizations ---")
    base_filename = os.path.splitext(os.path.basename(args.shaky_video_path))[0]
    
    gif_save_path = os.path.join(args.output_dir, f"{base_filename}_stabilization_process.gif")
    create_stabilization_gif(original_frames, shaky_frames, stabilized_frames, final_cropped_frames, gif_save_path)

def main():
    parser = argparse.ArgumentParser(description="POC for Global Video Stabilization (Multi-Mode).")
    parser.add_argument("--mode", type=str, required=True, choices=["generate", "compare"], help="The mode of operation.")
    parser.add_argument("--video_path", required=True, help="Path to the original (clean) .mat video file.")
    parser.add_argument("--num_frames", type=int, default=150, help="Total number of frames to process.")

    # --- Arguments for 'generate' mode ---
    parser.add_argument("--output_path", type=str, help="[Generate Mode] Path to save the new shaky .mat file.")
    parser.add_argument("--shift1_frame", type=int, default=75, help="[Generate Mode] Frame number to END the first gradual shift.")
    parser.add_argument("--shift1_x", type=float, default=-5.0, help="[Generate Mode] Total X shift at the end of phase 1 (%% of width).")
    parser.add_argument("--shift1_y", type=float, default=0.0, help="[Generate Mode] Total Y shift at the end of phase 1 (%% of height).")
    parser.add_argument("--shift2_frame", type=int, default=150, help="[Generate Mode] Frame number to END the second gradual shift.")
    parser.add_argument("--shift2_x", type=float, default=10.0, help="[Generate Mode] X shift RELATIVE to phase 1's end (%% of width).")
    parser.add_argument("--shift2_y", type=float, default=0.0, help="[Generate Mode] Y shift RELATIVE to phase 1's end (%% of height).")

    # --- Arguments for 'compare' mode ---
    parser.add_argument("--shaky_video_path", type=str, help="[Compare Mode] Path to the shaky .mat video file to be stabilized.")
    parser.add_argument("--output_dir", type=str, help="[Compare Mode] Directory to save the output visualizations.")
    
    args = parser.parse_args()

    if args.mode == 'generate':
        if not args.output_path:
            parser.error("--output_path is required for 'generate' mode.")
        run_generate_mode(args)
    elif args.mode == 'compare':
        if not args.shaky_video_path or not args.output_dir:
            parser.error("--shaky_video_path and --output_dir are required for 'compare' mode.")
        os.makedirs(args.output_dir, exist_ok=True)
        run_compare_mode(args)

    print("\n--- POC Finished Successfully ---")

if __name__ == "__main__":
    main()
"""
python src_cnn_v2/poc_stabilization.py \
  --video_path "/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_Gypsum_07162025_noshutter/T1.4V_2025-07-17-16-56-31_22_34_12_.mat" \
  --output_dir "stabilization_poc_output/Fluke_Gypsum_07162025_noshutter/vid-1-left-top-shake-stabilized" \
  --num_frames 150 \
  --shift1_frame 75 --shift1_x -5 \
  --shift2_frame 150 --shift2_x 10 

python src_cnn_v2/poc_stabilization.py \
  --video_path "/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0616_2025_noshutter/T1.4V_2.2Pa_2025-6-16-16-33-25_20_34_14_.mat" \
  --output_dir "stabilization_poc_output/Fluke_BrickCladding_2holes_0616_2025_noshutter/vid-1-left-top-shake-stabilized" \
  --num_frames 150 \
  --shift1_frame 75 --shift1_x -5 \
  --shift2_frame 150 --shift2_x 10 

#  A "Circular" Drift (Right, then Down)
python src_cnn_v2/poc_stabilization.py \
  --video_path "/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_Gypsum_07162025_noshutter/T1.4V_2025-07-17-16-56-31_22_34_12_.mat" \
  --output_dir "stabilization_poc_output/Fluke_Gypsum_07162025_noshutter/vid-1-shake_output_circular" \
  --num_frames 150 \
  --shift1_frame 75 --shift1_x 10 --shift1_y 0 \
  --shift2_frame 150 --shift2_x 0 --shift2_y 10 \
  --stabilization_method template

python src_cnn_v2/poc_stabilization.py \
  --video_path "/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_Gypsum_07162025_noshutter/T1.4V_2025-07-17-16-56-31_22_34_12_.mat" \
  --output_dir "stabilization_poc_output/Fluke_Gypsum_07162025_noshutter/vid-1-right-down-shake" \
  --shift1_frame 40 --shift1_x 20 --shift1_y 0 \
  --shift2_frame 100 --shift2_x 20 --shift2_y 15
"""