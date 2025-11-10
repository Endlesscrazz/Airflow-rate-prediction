# src_cnn_v2/poc_realtime_stabilization.py
"""
Proof of Concept (POC) for an END-TO-END Real-Time Stabilization & Detection Pipeline. (Version 2)
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cv2
import imageio.v2 as imageio
import time
from collections import deque
import matplotlib.cm as cm

try:
    from debug_files.debug_leak_finder import calculate_temporal_trend_map, calculate_local_heat_z_score, find_top_n_leaks
except ImportError:
    sys.exit("FATAL: Could not import from 'debug_leak_finder.py'.")

# --- Core Functions ---

def add_gradual_shake(frames, args):
    # This function is correct and remains unchanged
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
    for i in range(T):
        M = np.float32([[1, 0, total_x_shifts[i]], [0, 1, total_y_shifts[i]]])
        shaky_frames[:, :, i] = cv2.warpAffine(frames[:, :, i], M, (W, H))
    return shaky_frames

class RealTimeStabilizer:
    # This class is also correct and remains unchanged. It's for the "live view".
    def __init__(self, smoothing_window=30):
        self.smoothing_window = smoothing_window
        self.transforms = deque(maxlen=smoothing_window)
        self.previous_frame = None

    def process_frame(self, frame):
        current_frame_float = frame.astype(np.float32)
        if self.previous_frame is None:
            self.previous_frame = current_frame_float
            self.transforms.append((0.0, 0.0))
            return frame
        shift, _ = cv2.phaseCorrelate(self.previous_frame, current_frame_float)
        dx, dy = shift; self.transforms.append((dx, dy))
        avg_dx = np.mean([t[0] for t in self.transforms]); avg_dy = np.mean([t[1] for t in self.transforms])
        correction_dx = avg_dx - dx; correction_dy = avg_dy - dy
        H, W = frame.shape
        M = np.float32([[1, 0, correction_dx], [0, 1, correction_dy]])
        stabilized_frame = cv2.warpAffine(frame, M, (W, H))
        self.previous_frame = current_frame_float
        return stabilized_frame.astype(frame.dtype)

# --- NEW: Offline, "Static" Stabilizer for Leak Detection ---
def stabilize_buffer_statically(frame_buffer):
    """
    Takes a buffer of frames and stabilizes all of them to the first frame in the buffer.
    This creates a perfectly static video required for the leak detection algorithm.
    """
    buffer_array = np.stack(list(frame_buffer), axis=-1)
    H, W, T = buffer_array.shape
    
    static_stabilized_frames = np.zeros_like(buffer_array)
    reference_frame = buffer_array[:, :, 0].astype(np.float32)
    static_stabilized_frames[:, :, 0] = reference_frame
    
    for i in range(1, T):
        current_frame = buffer_array[:, :, i].astype(np.float32)
        shift, _ = cv2.phaseCorrelate(reference_frame, current_frame)
        dx, dy = shift
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        static_stabilized_frames[:, :, i] = cv2.warpAffine(buffer_array[:, :, i], M, (W, H))
        
    return static_stabilized_frames.astype(buffer_array.dtype)

# --- Visualization (Simplified for clarity) ---
def create_final_gif(original_frames, stabilized_frames_with_leaks, save_path, fps=10):
    print(f"\n--- Creating Final Visualization GIF ---")
    H, W, T = original_frames.shape
    vmin = np.percentile(original_frames, 1); vmax = np.percentile(original_frames, 99)
    cmap = cm.get_cmap('inferno')
    def normalize(frame, vmin, vmax):
        clipped = np.clip(frame, vmin, vmax)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(clipped))
        return (rgba[:, :, :3] * 255).astype(np.uint8)

    with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
        for i in range(T):
            orig_viz = normalize(original_frames[:, :, i], vmin, vmax)
            stabilized_viz = stabilized_frames_with_leaks[i]
            combined = np.hstack((orig_viz, stabilized_viz))
            cv2.putText(combined, '1. Original (for reference)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, '2. Stabilized with Live Detection', (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, f'Frame: {i}/{T}', (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.append_data(combined)
    print(f"  - Saved final GIF to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="POC for End-to-End Real-Time Pipeline.")
    # (Parser arguments are the same)
    parser.add_argument("--video_path", required=True, help="Path to the original (clean) .mat video file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output visualizations.")
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames to process.")
    parser.add_argument("--buffer_size", type=int, default=150, help="Number of frames to keep in the stabilization buffer.")
    parser.add_argument("--detection_interval", type=int, default=10, help="Run leak detection every N frames.")
    parser.add_argument("--shift1_frame", type=int, default=150, help="Frame to end first shift.")
    parser.add_argument("--shift1_x", type=float, default=-5.0, help="X shift at end of phase 1 (%%).")
    parser.add_argument("--shift1_y", type=float, default=0.0, help="Y shift at end of phase 1 (%%).")
    parser.add_argument("--shift2_frame", type=int, default=300, help="Frame to end second shift.")
    parser.add_argument("--shift2_x", type=float, default=10.0, help="X shift relative to phase 1 (%%).")
    parser.add_argument("--shift2_y", type=float, default=0.0, help="Y shift relative to phase 1 (%%).")
    args = parser.parse_args()

    print(f"--- Running End-to-End Real-Time Pipeline Simulation ---")
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.video_path))[0]

    try:
        original_frames = scipy.io.loadmat(args.video_path)['TempFrames'].astype(np.uint16)
        original_frames = original_frames[:, :, :args.num_frames]
    except Exception as e:
        sys.exit(f"FATAL: Could not load video file. Error: {e}")

    shaky_frames = add_gradual_shake(original_frames, args)
    stabilizer = RealTimeStabilizer(smoothing_window=30)
    
    # This buffer now holds the RAW shaky frames
    raw_frame_buffer = deque(maxlen=args.buffer_size)
    
    output_visualization_frames = []
    total_stabilization_time, total_detection_time, detection_runs, latest_leaks = 0, 0, 0, []

    print("\n--- Processing Simulated Real-Time Stream ---")
    vmin = np.percentile(original_frames, 1); vmax = np.percentile(original_frames, 99)
    cmap = cm.get_cmap('inferno')

    for i in range(args.num_frames):
        current_shaky_frame = shaky_frames[:, :, i]
        
        # --- Step A: Run the "Live View" Stabilizer ---
        start_time_stabilize = time.time()
        live_stabilized_frame = stabilizer.process_frame(current_shaky_frame)
        total_stabilization_time += (time.time() - start_time_stabilize)
        
        # --- Step B: Add the RAW shaky frame to a separate buffer for detection ---
        raw_frame_buffer.append(current_shaky_frame)
        
        # --- Step C: Periodically run leak detection ---
        if (i + 1) % args.detection_interval == 0 and len(raw_frame_buffer) == args.buffer_size:
            print(f"\nFrame {i+1}: Running leak detection on the latest {args.buffer_size} frames...")
            start_time_detect = time.time()

            # --- START OF FIX ---
            # 1. Statically stabilize the buffer before detection
            static_buffer = stabilize_buffer_statically(raw_frame_buffer)
            
            # 2. Run detection on the STATICALLY STABILIZED buffer
            score_map = calculate_temporal_trend_map(static_buffer) * \
                        (calculate_local_heat_z_score(static_buffer) ** 1.4)
            latest_leaks = find_top_n_leaks(score_map, num_leaks=2, min_distance=50)
            # --- END OF FIX ---
            
            detection_time_ms = (time.time() - start_time_detect) * 1000
            total_detection_time += detection_time_ms; detection_runs += 1
            print(f"  -> Detection finished in {detection_time_ms:.2f} ms. Found {len(latest_leaks)} leak(s).")

        # --- Step D: Create the visualization frame (using the "Live View" stabilizer output) ---
        viz_frame = (cmap((np.clip(live_stabilized_frame, vmin, vmax) - vmin) / (vmax - vmin))[:, :, :3] * 255).astype(np.uint8)
        if latest_leaks:
            for j, leak in enumerate(latest_leaks):
                center_y, center_x = leak['coords']
                cv2.circle(viz_frame, (center_x, center_y), 15, (0, 255, 255), 2)
                cv2.putText(viz_frame, f"#{j+1}", (center_x + 20, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        output_visualization_frames.append(viz_frame)

    # (Performance reporting and GIF creation are the same)
    print("\n--- Performance Summary ---")
    avg_stabilize_time = (total_stabilization_time * 1000) / args.num_frames
    avg_detect_time = total_detection_time / detection_runs if detection_runs > 0 else 0
    print(f"  - Average stabilization time per frame: {avg_stabilize_time:.2f} ms")
    print(f"  - Average leak detection time (when run): {avg_detect_time:.2f} ms")
    print(f"  - Max real-time FPS supported by stabilization: {1000 / avg_stabilize_time:.1f}")
    gif_save_path = os.path.join(args.output_dir, f"{base_filename}_end_to_end_simulation.gif")
    create_final_gif(original_frames, output_visualization_frames, gif_save_path)
    print("\n--- POC Finished Successfully ---")

if __name__ == "__main__":
    main()
"""
python src_cnn_v2/poc_realtime_stabilization.py \
  --video_path "/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_HardyBoard_08132025_2holes_noshutter/T1.4V_2025-08-14-15-47-12_21_34_13_.mat" \
  --output_dir "realtime_poc_output/Fluke_HardyBoard_08132025_2holes_noshutter/vid-1-circular" \
  --num_frames 150 \
  --shift1_frame 75 --shift1_x -5 --shift1_y -10 \
  --shift2_frame 150 --shift2_x 10 --shift2_y 20

python src_cnn_v2/poc_realtime_stabilization.py \
    --video_path "/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_Gypsum_07292025_noshutter/T1.4V_2025-08-01-18-45-09_20_32_12_.mat" \
    --output_dir "realtime_poc_output/Fluke_Gypsum_07292025_noshutter/vid-1"
"""