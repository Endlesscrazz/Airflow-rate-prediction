# src_cnn_v2/poc_stabilization.py
"""
Proof of Concept (POC) for Global Video Stabilization. (Final Version)

This script performs an end-to-end demonstration and produces three GIFs,
all rendered with robust, high-contrast grayscale visualization.
"""
import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import imageio.v2 as imageio

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
    return shaky_frames

def stabilize_video_phase_correlation(shaky_frames):
    H, W, T = shaky_frames.shape
    stabilized_frames = np.zeros_like(shaky_frames)
    reference_frame = shaky_frames[:, :, 0].astype(np.float32)  #first frame taken as reference
    stabilized_frames[:, :, 0] = shaky_frames[:, :, 0]
    print("  - Stabilizing shaky video using Phase Correlation...")
    for i in range(1, T):
        current_frame = shaky_frames[:, :, i].astype(np.float32)
        shift, _ = cv2.phaseCorrelate(reference_frame, current_frame)
        dx, dy = shift
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stabilized_frames[:, :, i] = cv2.warpAffine(shaky_frames[:, :, i], M, (W, H))
    return stabilized_frames

# --- Visualization Functions ---

def normalize_for_gif(frame, vmin, vmax):
    clipped_frame = np.clip(frame, vmin, vmax)
    frame_8bit = cv2.normalize(clipped_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2RGB)

def create_comparison_gif(frame_set1, frame_set2, label1, label2, color1, color2, vmin, vmax, save_path, fps=10):
    print(f"  - Creating GIF: {os.path.basename(save_path)}")
    H, W, T = frame_set1.shape
    with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
        for i in range(T):
            viz1 = normalize_for_gif(frame_set1[:, :, i], vmin, vmax)
            viz2 = normalize_for_gif(frame_set2[:, :, i], vmin, vmax)
            combined_frame = np.hstack((viz1, viz2))
            cv2.putText(combined_frame, label1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color1, 2)
            cv2.putText(combined_frame, label2, (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color2, 2)
            cv2.putText(combined_frame, f'Frame: {i}/{T}', (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.append_data(combined_frame)
    print(f"  - Saved GIF to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="POC for Global Video Stabilization.")
    parser.add_argument("--video_path", required=True, help="Path to the original (clean) .mat video file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output visualizations.")
    parser.add_argument("--num_frames", type=int, default=150, help="Total number of frames to process.")
    parser.add_argument("--shift1_frame", type=int, default=75, help="Frame number to END the first gradual shift.")
    parser.add_argument("--shift1_x", type=float, default=-5.0, help="Total X shift at the end of phase 1 (%% of width).")
    parser.add_argument("--shift1_y", type=float, default=0.0, help="Total Y shift at the end of phase 1 (%% of height).")
    parser.add_argument("--shift2_frame", type=int, default=150, help="Frame number to END the second gradual shift.")
    parser.add_argument("--shift2_x", type=float, default=10.0, help="X shift RELATIVE to phase 1's end (%% of width).")
    parser.add_argument("--shift2_y", type=float, default=0.0, help="Y shift RELATIVE to phase 1's end (%% of height).")
    args = parser.parse_args()

    print(f"--- Running Stabilization POC ---")
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.video_path))[0]

    try:
        original_frames = scipy.io.loadmat(args.video_path)['TempFrames'].astype(np.float64)
        original_frames = original_frames[:, :, :args.num_frames]
        print(f"  - Loaded video with shape: {original_frames.shape}")
    except Exception as e:
        sys.exit(f"FATAL: Could not load video file. Error: {e}")

    shaky_frames = add_gradual_shake(original_frames, args)
    stabilized_frames = stabilize_video_phase_correlation(shaky_frames)

    print("\n--- Generating Visualizations ---")
    
    # --- START OF FIX ---
    # 1. Calculate the robust, global contrast limits ONCE from the original video
    vmin = np.percentile(original_frames, 1)
    vmax = np.percentile(original_frames, 99)

    # 2. Pass these limits to all GIF creation functions
    # GIF 1: The Problem (Original vs. Shaky)
    gif1_path = os.path.join(args.output_dir, f"{base_filename}_1_problem.gif")
    create_comparison_gif(original_frames, shaky_frames, 
                          "1. Original", "2. Shaky", 
                          (0, 255, 0), (0, 0, 255), 
                          vmin, vmax, gif1_path)
    
    # GIF 2: The Solution (Shaky vs. Stabilized)
    gif2_path = os.path.join(args.output_dir, f"{base_filename}_2_solution.gif")
    create_comparison_gif(shaky_frames, stabilized_frames, 
                          "1. Shaky", "2. Stabilized", 
                          (0, 0, 255), (255, 255, 0), 
                          vmin, vmax, gif2_path)

    # GIF 3: The Final Comparison (Original vs. Stabilized)
    gif3_path = os.path.join(args.output_dir, f"{base_filename}_3_final_comparison.gif")
    create_comparison_gif(original_frames, stabilized_frames, 
                          "1. Original", "2. Stabilized", 
                          (0, 255, 0), (255, 255, 0), 
                          vmin, vmax, gif3_path)
    # --- END OF FIX ---

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


"""
