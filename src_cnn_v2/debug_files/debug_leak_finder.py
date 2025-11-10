# src_cnn_v2/debug_files/debug_leak_finder.py
"""
Standalone Debugging Script for Advanced Leak Detection.
(Version 3.1: Corrected np.save arguments)
"""
import os
import sys
import argparse
import cv2
import time
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from tqdm import tqdm
from skimage.feature import peak_local_max
from joblib import Parallel, delayed
from sklearn.linear_model import HuberRegressor

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- Core Logic Functions ---

def _calculate_theilsen_slope_for_row(row_data, t):
    W = row_data.shape[0]; row_slopes = np.zeros(W, dtype=np.float32)
    for c in range(W):
        try:
            res = stats.theilslopes(row_data[c, :], t)
            row_slopes[c] = res[0] if np.isfinite(res[0]) else 0.0
        except (ValueError, IndexError): row_slopes[c] = 0.0
    return row_slopes

def _calculate_huber_slope_for_row_cpu(row_data, t):
    W = row_data.shape[0]; row_slopes = np.zeros(W, dtype=np.float32)
    t_reshaped = t.reshape(-1, 1); huber = HuberRegressor()
    for c in range(W):
        try:
            huber.fit(t_reshaped, row_data[c, :])
            row_slopes[c] = huber.coef_[0] if np.isfinite(huber.coef_[0]) else 0.0
        except ValueError: row_slopes[c] = 0.0
    return row_slopes

def calculate_huber_slope_gpu(frames):
    device = torch.device("cuda")
    print(f"  - PyTorch detected. Using GPU: {torch.cuda.get_device_name(0)}")
    H, W, T = frames.shape
    frames_gpu = torch.from_numpy(frames).to(device, dtype=torch.float32)
    t_gpu = torch.arange(T, device=device, dtype=torch.float32).view(1, 1, -1)
    
    t_mean = torch.mean(t_gpu)
    frames_mean = torch.mean(frames_gpu, dim=2, keepdim=True)
    numerator = torch.sum((t_gpu - t_mean) * (frames_gpu - frames_mean), dim=2)
    denominator = torch.sum((t_gpu - t_mean)**2, dim=2)
    slope = numerator / denominator
    intercept = frames_mean.squeeze(2) - slope * t_mean

    for _ in range(5):
        predicted = slope.unsqueeze(2) * t_gpu + intercept.unsqueeze(2)
        residuals = frames_gpu - predicted
        mad = torch.median(torch.abs(residuals - torch.median(residuals)))
        delta = 1.345 * mad
        weights = torch.ones_like(residuals)
        outliers = torch.abs(residuals) > delta
        weights[outliers] = delta / torch.abs(residuals[outliers])
        
        w_sum = torch.sum(weights, dim=2)
        wx_sum = torch.sum(weights * t_gpu, dim=2)
        wy_sum = torch.sum(weights * frames_gpu, dim=2)
        wxy_sum = torch.sum(weights * t_gpu * frames_gpu, dim=2)
        wx2_sum = torch.sum(weights * t_gpu**2, dim=2)
        w_denom = w_sum * wx2_sum - wx_sum**2
        
        slope = (w_sum * wxy_sum - wx_sum * wy_sum) / w_denom
        intercept = (wy_sum * wx2_sum - wx_sum * wxy_sum) / w_denom

    return torch.clamp(slope, min=0).cpu().numpy()

def calculate_temporal_trend_map(frames, regressor='theilsen', use_gpu=False):
    H, W, T = frames.shape; t = np.arange(T)
    if T < 2: return np.zeros((H, W), dtype=np.float32), "N/A"

    slope_map = None
    device_used = "CPU (Parallel)"

    if regressor == 'huber' and use_gpu:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            slope_map = calculate_huber_slope_gpu(frames)
            device_used = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            print("\n!!! WARNING: --gpu flag was used, but a CUDA-enabled GPU was not found by PyTorch. Falling back to CPU. !!!\n")
            regressor = 'huber' # Ensure we still use huber on CPU
            device_used = "CPU (Parallel)"
    
    if slope_map is None: # Fallback for Theil-Sen or if GPU Huber failed
        print(f"  - Calculating slopes on CPU using '{regressor}' regressor...")
        if regressor == 'huber':
            func_to_parallelize = _calculate_huber_slope_for_row_cpu
        else:
            func_to_parallelize = _calculate_theilsen_slope_for_row
        
        results = Parallel(n_jobs=-1)(
            delayed(func_to_parallelize)(frames[r, :, :], t) for r in tqdm(range(H), desc="  - Processing rows")
        )
        slope_map = np.vstack(results)

    np.nan_to_num(slope_map, copy=False, nan=0.0)
    trend_map = np.maximum(slope_map, 0.0)
    return trend_map, device_used

def calculate_local_heat_z_score(frames):
    temp_mean = np.mean(frames, axis=2).astype(np.float32)
    loc_mu = cv2.GaussianBlur(temp_mean, (31, 31), 0)
    temp_mean_sq = cv2.GaussianBlur(temp_mean**2, (31, 31), 0)
    loc_sd = np.sqrt(np.maximum(temp_mean_sq - loc_mu**2, 0)) + 1e-9
    return np.maximum((temp_mean - loc_mu) / loc_sd, 0.0)

def find_top_n_leaks(score_map, num_leaks, min_distance=50):
    threshold_abs = score_map.max() * 0.05
    coordinates = peak_local_max(score_map, min_distance=min_distance, threshold_abs=threshold_abs)
    if coordinates.size == 0: return []
    scores = score_map[coordinates[:, 0], coordinates[:, 1]]
    candidates = [{'coords': (r, c), 'score': s} for (r, c), s in zip(coordinates, scores)]
    candidates.sort(key=lambda p: p['score'], reverse=True)
    return candidates[:num_leaks]

def save_verification_plot(base_frame, score_map, leak_candidates, crop_size, save_path):
    num_found = len(leak_candidates)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Leak Detection Verification ({num_found} Leak(s) Found)", fontsize=16)
    ax[0].imshow(base_frame, cmap='inferno'); ax[0].set_title("Original Frame")
    im = ax[1].imshow(score_map, cmap='hot'); ax[1].set_title("Fused Score Map")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="Score")
    for i, leak in enumerate(leak_candidates):
        center_y, center_x = leak['coords']
        for axis in ax:
            axis.scatter([center_x], [center_y], s=200, c='cyan', marker='*')
            axis.text(center_x + 15, center_y, f"#{i+1}", color='white', fontsize=14, weight='bold')
        rect = patches.Rectangle((center_x - crop_size//2, center_y - crop_size//2), crop_size, crop_size, linewidth=2.5, edgecolor='lime', facecolor='none')
        ax[0].add_patch(rect)
    ax[0].axis('off'); ax[1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  - Saved verification plot to: {save_path}")

def crop_sequence(frames, center_x, center_y, crop_size):
    H, W, T = frames.shape
    half_crop = crop_size // 2
    x_start = max(0, center_x - half_crop); x_end = min(W, x_start + crop_size)
    y_start = max(0, center_y - half_crop); y_end = min(H, y_start + crop_size)
    if x_end - x_start < crop_size: x_start = x_end - crop_size
    if y_end - y_start < crop_size: y_start = y_end - crop_size
    return frames[y_start:y_end, x_start:x_end, :]

def main():
    parser = argparse.ArgumentParser(description="Debug script for advanced leak detection.")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--crop_size", type=int, default=16)
    parser.add_argument("--heat_power", type=float, default=1.4)
    parser.add_argument("--num_leaks", type=int, default=1)
    parser.add_argument("--regressor", type=str, default="theilsen", choices=["theilsen", "huber"])
    parser.add_argument("--gpu", action="store_true", help="Use GPU for Huber regression if available.")
    args = parser.parse_args()

    print(f"--- Running Leak Detection Debug Script (Regressor: {args.regressor.upper()}) ---")
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.video_path))[0]

    try:
        frames = scipy.io.loadmat(args.video_path)['TempFrames'].astype(np.float64)
    except Exception as e:
        sys.exit(f"FATAL: Could not load video file. Error: {e}")

    start_time = time.perf_counter()
    temporal_map, device_used = calculate_temporal_trend_map(frames, args.regressor, args.gpu)
    end_time = time.perf_counter()
    temporal_map_time = end_time - start_time
    
    heat_map = calculate_local_heat_z_score(frames)
    score_map = temporal_map * (heat_map ** args.heat_power)
    
    leak_candidates = find_top_n_leaks(score_map, args.num_leaks)
    
    if not leak_candidates:
        print("  - No leaks found that meet the criteria. Exiting.")
    else:
        plot_save_path = os.path.join(args.output_dir, f"{base_filename}_{args.regressor}_leak_verification.png")
        save_verification_plot(np.mean(frames, axis=2), score_map, leak_candidates, args.crop_size, plot_save_path)
        for i, leak in enumerate(leak_candidates):
            center_y, center_x = leak['coords']
            cropped_frames = crop_sequence(frames, center_x, center_y, args.crop_size)
            npy_save_path = os.path.join(args.output_dir, f"{base_filename}_{args.regressor}_crop_{args.crop_size}x{args.crop_size}_leak_{i+1}.npy")
            np.save(npy_save_path, cropped_frames.transpose(2, 0, 1))
            print(f"  - Saved cropped sequence for leak #{i+1} to: {npy_save_path}")

    print("\n" + "="*50)
    print("--- COMPARISON LOG ---")
    print(f"{'Regressor Used:':<25} {args.regressor.upper()}")
    print(f"{'Computation Device:':<25} {device_used}")
    print(f"{'Temporal Map Time:':<25} {temporal_map_time:.2f} seconds")
    print("-" * 25)
    if leak_candidates:
        print(f"Found {len(leak_candidates)} leak candidate(s):")
        for i, leak in enumerate(leak_candidates):
            print(f"  - Leak #{i+1}: Coords={leak['coords']}, Score={leak['score']:.6f}")
    else:
        print("  - No leaks found.")
    print("="*50 + "\n")
    
    print("\n--- Debug Script Finished Successfully ---")

if __name__ == "__main__":
    main()
"""
# Sameem 11-holes brickcladding
python src_cnn_v2/debug_files/debug_leak_finder.py \
  --video_path "/Users/shreyas/Downloads/Fluke_Brickcladding_10112025_11holes_noshutter_Sameem/T15P_2025-11-10-10-11-50_23_29_6_.mat" \
  --output_dir "debug_outputs/Fluke_Brickcladding_10112025_11holes_noshutter_Sameem/vid-1" \
  --crop_size 16 \
  --num_leaks 11 \
  --regressor theilsen 


# Brickcladding-CHPC
python src_cnn_v2/debug_files/debug_leak_finder.py \
  --video_path "/scratch/general/vast/u1527145/datasets/Fluke_BrickCladding_2holes_0805_2025_noshutter/T1.6V_2025-08-05-19-57-24_20_34_14_.mat" \
  --output_dir "debug_outputs/Fluke_BrickCladding_2holes_0808_2025_noshutter/vid-1" \
  --crop_size 16 \
  --num_leaks 2 \
  --regressor huber \
  --gpu
"""
