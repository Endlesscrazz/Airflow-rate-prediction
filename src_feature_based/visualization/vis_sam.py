# src/visualize_pipeline.py
"""
A diagnostic script to visualize each step of the thermal anomaly detection pipeline.

This script takes an IR video and processing parameters as input and generates a
single, comprehensive image that shows the data at each stage of the analysis,
from raw input to final prompt selection. It does NOT run the SAM model.

Example Usage:

gypsum2:
1.4/vid-3
python src/visualization/vis_sam.py \
  --input datasets/dataset_gypsum2/FanPower_1.4V/T1.4V_2025-07-17-17-36-27_22_26_4_.mat \
  --output_dir output_SAM/pipeline_viz/gypsum2/1.4/vid-3/iter-1-roi0 \
  --temporal_smooth_window 3 \
  --roi_erosion 2 \
  --activity_method kendall_tau \
  --spatial_filter bilateral \
  --tophat_filter_size 1
  --num_leaks 1


vid-2:
python src/visualization/vis_sam.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-34-30_20_34_14_.mat \
  --output_dir output_SAM/pipeline_viz/vid-2/iter-1-tophat3 \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --activity_method kendall_tau \
  --spatial_filter bilateral \
  --tophat_filter_size 3

vid-4:
python src/visualization/vis_sam.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-44-18_20_30_10_.mat \
  --output_dir output_SAM/pipeline_viz/vid-4/iter-1-tophat \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --activity_method kendall_tau \
  --spatial_filter bilateral \
  --tophat_filter_size 5

vid-6:
python src/visualization/vis_sam.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-17-0-30_20_26_6_.mat \
  --output_dir output_SAM/pipeline_viz/vid-6/kernel9_roi7_minarea5_q995 \
  --spatial_filter tophat \
  --spatial_kernel_size 9 \
  --activity_method kendall_tau  \
  --activity_quantile 0.995 \
  --roi_erosion 7 \
  --min_area 5

vid-5:
python src/visualization/vis_sam.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-59-16_20_26_6_.mat \
  --output_dir output_SAM/pipeline_viz/vid-5/kernel9_roi7_minarea5_q995 \
  --spatial_filter tophat \
  --spatial_kernel_size 9 \
  --activity_method kendall_tau  \
  --activity_quantile 0.995 \
  --roi_erosion 7 \
  --min_area 5

vid-9:
python src/visualization/vis_sam.py \
  --input datasets/dataset_two_holes/T2.0V_6.3Pa_2025-6-16-17-25-13_20_30_10_.mat \
  --output_dir output_SAM/pipeline_viz/vid-9/kernel7_roi7_minarea5_q995 \
  --spatial_filter tophat \
  --spatial_kernel_size 7 \
  --activity_method kendall_tau  \
  --activity_quantile 0.995 \
  --roi_erosion 7 \
  --min_area 5
"""
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
vis_sam.py

Visualize each preprocessing and prompt-selection step from debug_simple_SAM.py in one panel,
without performing the SAM segmentation.
"""
import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, mstats
from joblib import Parallel, delayed
from tqdm import tqdm

# --- STAGE 1: PRE-PROCESSING FUNCTIONS ---

def apply_spatial_filter(frames, filter_type='bilateral', d=9, sigmaColor=75, sigmaSpace=75):
    if filter_type.lower() == 'none':
        return frames
    H, W, T = frames.shape
    filtered = np.zeros_like(frames)
    for i in tqdm(range(T), desc="Spatial Filter", leave=False):
        frame8 = cv2.normalize(frames[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if filter_type.lower() == 'bilateral':
            f = cv2.bilateralFilter(frame8, d, sigmaColor, sigmaSpace)
        else:
            f = frame8
        filtered[:, :, i] = cv2.normalize(f.astype(np.float64), None,
                                            np.min(frames), np.max(frames),
                                            cv2.NORM_MINMAX)
    return filtered


def apply_temporal_smoothing(frames, window_size):
    if window_size <= 1:
        return frames
    H, W, T = frames.shape
    smoothed = np.zeros_like(frames, dtype=np.float64)
    kernel = np.ones(window_size) / window_size
    for r in tqdm(range(H), desc="Temporal Smooth", leave=False):
        for c in range(W):
            series = frames[r, c, :]
            conv = np.convolve(series, kernel, mode='valid')
            pad_before = (T - len(conv)) // 2
            pad_after = T - len(conv) - pad_before
            smoothed[r, c, :] = np.pad(conv, (pad_before, pad_after), mode='edge')
    return smoothed

# --- STAGE 2: ACTIVITY MAP GENERATION ---

def _calculate_slope_for_row(row_data, t, method):
    W = row_data.shape[0]
    out = np.zeros(W, dtype=np.float64)
    for c in range(W):
        series = row_data[c, :]
        if not np.any(np.isnan(series)) and len(series) > 1:
            try:
                if method == 'theil_sen':
                    val, _, _, _ = mstats.theilslopes(series, t, 0.95)
                else:
                    val, _ = kendalltau(t, series)
            except Exception:
                val = 0.0
            out[c] = val if np.isfinite(val) else 0.0
    return out


def generate_activity_map(frames, method, env_para):
    H, W, T = frames.shape
    if T < 2:
        return np.zeros((H, W), dtype=np.float64)
    t = np.arange(T)
    rows = Parallel(n_jobs=-1)(
        delayed(_calculate_slope_for_row)(frames[r, :, :], t, method)
        for r in tqdm(range(H), desc="Activity Map", leave=False)
    )
    raw = np.vstack(rows)
    if env_para == 1:
        m = np.copy(raw)
        m[m < 0] = 0
    elif env_para == -1:
        m = -raw
        m[m < 0] = 0
    else:
        m = np.abs(raw)
    return m

# --- STAGE 3: POST-PROCESSING ---

def apply_tophat_filter(activity_map, kernel_size):
    if kernel_size <= 0:
        return activity_map
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mx = np.max(activity_map)
    if mx == 0:
        return activity_map
    im8 = (activity_map / mx * 255).astype(np.uint8)
    th = cv2.morphologyEx(im8, cv2.MORPH_TOPHAT, kernel)
    return th.astype(np.float64) / 255 * mx


def create_panel_roi_mask(median_frame, erosion_iterations):
    frame8 = cv2.normalize(median_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, thresh = cv2.threshold(frame8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    if num <= 1:
        return np.ones_like(median_frame, dtype=np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8)
    if erosion_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erosion_iterations)
    return mask

# --- STAGE 4: PROMPT SELECTION ---

def find_prompts_by_peak_activity(activity_map, num_prompts, quantile_thresh):
    active = activity_map[activity_map > 1e-9]
    if active.size == 0:
        return [], []
    thresh = np.quantile(active, quantile_thresh)
    bm = (activity_map >= thresh).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, kernel)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bm, 8)
    if num <= 1:
        return [], []
    candidates = []
    for i in range(1, num):
        mask = (labels == i)
        peak = np.max(activity_map[mask])
        candidates.append({'centroid': cents[i], 'stat': stats[i], 'peak': peak})
    sorted_c = sorted(candidates, key=lambda x: x['peak'], reverse=True)
    return sorted_c[:num_prompts], candidates

# --- MAIN SCRIPT ---

def main():
    parser = argparse.ArgumentParser(description="Visualize preprocessing and prompt selection steps")
    parser.add_argument("--input", required=True, help=".mat file with TempFrames")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_leaks", type=int, default=2)
    parser.add_argument("--env_para", type=int, default=0, choices=[1,-1,0])
    parser.add_argument("--activity_method", type=str, default="kendall_tau", choices=["theil_sen","kendall_tau"])
    parser.add_argument("--spatial_filter", type=str, default="bilateral", choices=["none","bilateral"])
    parser.add_argument("--temporal_smooth_window", type=int, default=3)
    parser.add_argument("--tophat_filter_size", type=int, default=0)
    parser.add_argument("--activity_quantile", type=float, default=0.995)
    parser.add_argument("--roi_erosion", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mat = scipy.io.loadmat(args.input)
    frames = mat['TempFrames'].astype(np.float64)
    # Execute steps
    sf = apply_spatial_filter(frames, args.spatial_filter)
    ts = apply_temporal_smoothing(sf, args.temporal_smooth_window)
    am = generate_activity_map(ts, args.activity_method, args.env_para)
    pt = apply_tophat_filter(am, args.tophat_filter_size)
    median = np.median(frames, axis=2)
    roi = create_panel_roi_mask(median, args.roi_erosion)
    masked = pt * roi
    top, all_c = find_prompts_by_peak_activity(masked, args.num_leaks, args.activity_quantile)

    # Plot all in one panel
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    ax[0,0].imshow(sf[:,:,0], cmap='gray'); ax[0,0].set_title('Spatially Filtered Frame')
    ax[0,1].imshow(ts[:,:,0], cmap='gray'); ax[0,1].set_title('Temporally Smoothed Frame')
    im1 = ax[0,2].imshow(am, cmap='hot'); ax[0,2].set_title('Activity Map'); fig.colorbar(im1, ax=ax[0,2])
    im2 = ax[1,0].imshow(pt, cmap='hot'); ax[1,0].set_title('After Top-Hat'); fig.colorbar(im2, ax=ax[1,0])
    ax[1,1].imshow(roi, cmap='gray'); ax[1,1].set_title('Panel ROI Mask')
    im3 = ax[1,2].imshow(masked, cmap='hot'); ax[1,2].set_title('Masked Activity & Prompts')
    # Overlay prompts
    if all_c:
        pts = np.array([c['centroid'] for c in all_c])
        ax[1,2].scatter(pts[:,0], pts[:,1], s=50, facecolors='none', edgecolors='cyan', label='All Candidates')
    if top:
        pts2 = np.array([p['centroid'] for p in top])
        ax[1,2].scatter(pts2[:,0], pts2[:,1], s=200, marker='*', edgecolors='yellow', facecolors='none', label='Top Prompts')
    ax[1,2].legend()
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, 'vis_steps.png')
    plt.savefig(out_path)
    print(f"Saved visualization panel to: {out_path}")

if __name__ == '__main__':
    main()
