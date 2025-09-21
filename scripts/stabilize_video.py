# scripts/stabilize_video_poc.py
"""
A Proof-of-Concept script to demonstrate and test video stabilization.
1. Optionally adds artificial shake to a stable source video.
2. Implements a feature-based Electronic Image Stabilization (EIS) pipeline.
3. Saves the shaky (optional) and stabilized videos for comparison.
"""
import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
from tqdm import tqdm
import skvideo.io

def add_artificial_shake(frames, max_dx, max_dy, max_da):
    """Adds random jitter to a stable video to simulate handheld shake."""
    print(f"\n--- Adding Artificial Shake (dx:±{max_dx}, dy:±{max_dy}, da:±{max_da}) ---")
    H, W, T = frames.shape
    shaky_frames = np.zeros_like(frames)
    
    dx = np.cumsum(np.random.uniform(-max_dx, max_dx, T))
    dy = np.cumsum(np.random.uniform(-max_dy, max_dy, T))
    da = np.cumsum(np.random.uniform(-max_da, max_da, T))
    
    # Smooth the random path to make it look more natural
    smooth_kernel = np.ones(10)/10
    dx = np.convolve(dx, smooth_kernel, mode='same')
    dy = np.convolve(dy, smooth_kernel, mode='same')
    da = np.convolve(da, smooth_kernel, mode='same')

    for i in tqdm(range(T), desc="Generating shaky frames"):
        M_rot = cv2.getRotationMatrix2D((W/2, H/2), da[i], 1)
        M_rot[0, 2] += dx[i]
        M_rot[1, 2] += dy[i]
        shaky_frames[:, :, i] = cv2.warpAffine(frames[:, :, i], M_rot, (W, H))
        
    return shaky_frames

def get_motion(prev_frame, curr_frame):
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    if prev_pts is None: return np.array([0.0, 0.0, 0.0]) # dx, dy, da
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_pts, None)
    idx = np.where(status==1)[0]
    if len(idx) < 4: return np.array([0.0, 0.0, 0.0])
    prev_pts, curr_pts = prev_pts[idx], curr_pts[idx]
    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    
    if m is None: return np.array([0.0, 0.0, 0.0])
    
    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0]) * (180 / np.pi)
    return np.array([dx, dy, da])

def smooth_motion_path(transforms, smoothing_window):
    smoothed_transforms = np.copy(transforms)
    for i in range(transforms.shape[1]):
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed_series = np.convolve(transforms[:, i], kernel, mode='valid')
        pad = (len(transforms) - len(smoothed_series)) // 2
        smoothed_transforms[:, i] = np.pad(smoothed_series, (pad, len(transforms) - len(smoothed_series) - pad), mode='edge')
    return smoothed_transforms

def main(args):
    print("--- Starting Video Stabilization POC ---")
    base_filename = os.path.splitext(os.path.basename(args.input_video))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    
    frames = scipy.io.loadmat(args.input_video)['TempFrames'].astype(np.float64)
    H, W, T = frames.shape
    print(f"Video loaded. Shape: {W}x{H}, Frames: {T}")

    frames_uint8 = np.array([cv2.normalize(frames[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) for i in range(T)]).transpose(1, 2, 0)

    if args.add_shake:
        shaky_frames_uint8 = add_artificial_shake(frames_uint8, args.shake_x, args.shake_y, args.shake_angle)
        shaky_output_path = os.path.join(args.output_dir, f"{base_filename}.shaky.mp4")
        
        # CORRECTED SAVING LOGIC
        shaky_frames_for_saving = shaky_frames_uint8.transpose(2, 0, 1)[:, :, :, np.newaxis]
        skvideo.io.vwrite(shaky_output_path, shaky_frames_for_saving, outputdict={'-vcodec': 'libx264', '-pix_fmt': 'gray', '-r': str(args.fps)})
        print(f"  - Saved artificially shaken video to: {shaky_output_path}")
        input_frames_for_stabilization = shaky_frames_uint8
    else:
        input_frames_for_stabilization = frames_uint8

    print("\nStep 1: Estimating camera motion...")
    transforms = np.zeros((T, 3), np.float64) # dx, dy, da
    for i in tqdm(range(T - 1), desc="Analyzing frames"):
        transforms[i+1] = get_motion(input_frames_for_stabilization[:, :, i], input_frames_for_stabilization[:, :, i+1])
    
    trajectory = np.cumsum(transforms, axis=0)

    print("\nStep 2: Smoothing the motion path...")
    smoothed_trajectory = smooth_motion_path(trajectory, args.smoothing_window)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    print("\nStep 3: Applying correction and generating stabilized video...")
    stabilized_output_path = os.path.join(args.output_dir, f"{base_filename}.stabilized.mp4")
    writer = skvideo.io.FFmpegWriter(stabilized_output_path, outputdict={'-vcodec': 'libx264', '-pix_fmt': 'gray', '-r': str(args.fps)})

    for i in tqdm(range(T), desc="Applying transforms"):
        dx, dy, da = transforms_smooth[i]
        M = cv2.getRotationMatrix2D((W/2, H/2), da, 1)
        M[0, 2] += dx
        M[1, 2] += dy
        
        frame_stabilized = cv2.warpAffine(input_frames_for_stabilization[:, :, i], M, (W, H))
        writer.writeFrame(np.expand_dims(frame_stabilized, axis=-1))

    writer.close()
    print(f"\n POC Complete. Stabilized video saved to: {stabilized_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proof-of-concept for thermal video stabilization.")
    parser.add_argument("input_video", help="Path to the input .mat video file.")
    parser.add_argument("output_dir", help="Directory to save the output videos.")
    parser.add_argument("--smoothing_window", type=int, default=30, help="Size of the moving average window. Larger is smoother.")
    parser.add_argument("--add_shake", action='store_true', help="Add artificial shake to the input video before stabilizing.")
    parser.add_argument("--shake_x", type=float, default=2.0, help="Max horizontal shake per frame (pixels).")
    parser.add_argument("--shake_y", type=float, default=2.0, help="Max vertical shake per frame (pixels).")
    parser.add_argument("--shake_angle", type=float, default=0.5, help="Max angular shake per frame (degrees).")
    parser.add_argument("--fps", type=int, default=10, help="FPS for the output videos.")
    args = parser.parse_args()
    main(args)
"""
gypsum:
python -m scripts.stabilize_video \
/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_Gypsum_07162025_noshutter/T1.4V_2025-07-17-16-56-31_22_34_12_.mat \
stabilization_poc_output \
--add_shake

brickcladding
python -m scripts.stabilize_video \
/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter/T1.4V_2025-08-08-19-44-50_20_32_12_.mat \
stabilization_poc_output/brickcladding \
--add_shake
"""