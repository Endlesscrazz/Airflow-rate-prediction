#!/usr/bin/env python3
"""
visualize_focus_area_two_stage.py

Standalone script to visualize both STATIC and DYNAMIC focus areas for a single thermal .mat file,
by first extracting the static ROI (the actual hole) from the mean frame, then computing a slope map
and extracting the dynamic hotspot within that ROI.

Workflow:
1. STATIC ROI: extract hotspot from the time-averaged (mean) frame using quantile thresholding and center selection.
2. DYNAMIC ROI: compute per-pixel linear slope map, mask to STATIC ROI, then extract hotspot from slope map via quantile + center selection.

Outputs in `output_dir`:
- orig_gray/color, mean_gray/color, static_mask/overlay
- slope_gray/color, dynamic_mask/overlay

Usage example:
    python visualize_focus_area_two_stage.py \
      input.mat output_dir \
      --static_quantile 0.95 --dynamic_quantile 0.95 \
      --morph_kernel 5 --orig_colormap hot --grad_colormap inferno --fps 30
"""
import os, sys, argparse
import numpy as np
import cv2
import scipy.io

# --- Helper functions ---

def compute_mean_frame(frames):
    return None if frames is None or frames.ndim != 3 else np.mean(frames, axis=2, dtype=np.float64)


def compute_slope_map(frames, fps=30.0):
    T = frames.shape[2]
    dt = 1.0 / fps
    t = np.arange(T, dtype=np.float64) * dt
    t_mean = t.mean()
    var_t = ((t - t_mean)**2).sum()
    f = frames.astype(np.float64)
    mean_pix = f.mean(axis=2, keepdims=True)
    diff = f - mean_pix
    cov = np.tensordot(diff, (t - t_mean), axes=([2], [0]))
    slopes = cov / var_t
    return np.abs(slopes)


def extract_hotspot_from_map(A, threshold_quantile, morph_kernel):
    """
    Threshold A at the given quantile, clean and pick the component closest to image center.
    Returns (mask, area).
    """
    vals = A[~np.isnan(A)]
    if vals.size == 0:
        return np.zeros_like(A, bool), 0.0
    thr = np.percentile(vals, threshold_quantile * 100)
    thr = max(thr, 1e-6)
    binm = (np.nan_to_num(A, nan=-np.inf) >= thr).astype(np.uint8)
    k = max(1, morph_kernel)
    kernel = np.ones((k, k), np.uint8)
    clean = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    n_lbl, labels, stats, cents = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if n_lbl <= 1:
        return np.zeros_like(A, bool), 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    # center-based selection
    h, w = A.shape
    cx, cy = w/2, h/2
    dists = [((x-cx)**2 + (y-cy)**2) for (x, y) in cents[1:]]
    idx = int(np.argmin(dists)) + 1
    mask = (labels == idx)
    return mask, float(areas[idx-1])


def normalize_to_uint8(img):
    mn, mx = np.nanmin(img), np.nanmax(img)
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def save_visuals(orig, mean_map, slope_map, static_mask, dyn_mask, args):
    os.makedirs(args.output_dir, exist_ok=True)
    og = normalize_to_uint8(orig)
    cv2.imwrite(f"{args.output_dir}/orig_gray.png", og)
    cv2.imwrite(f"{args.output_dir}/orig_color.png", cv2.applyColorMap(og, args.orig_cmap))
    mg = normalize_to_uint8(mean_map)
    cv2.imwrite(f"{args.output_dir}/mean_gray.png", mg)
    cv2.imwrite(f"{args.output_dir}/mean_color.png", cv2.applyColorMap(mg, args.orig_cmap))
    sm = static_mask.astype(np.uint8) * 255
    cv2.imwrite(f"{args.output_dir}/static_mask.png", sm)
    bgr = cv2.cvtColor(og, cv2.COLOR_GRAY2BGR)
    bgr[static_mask] = (0, 255, 255)
    ov = cv2.addWeighted(bgr, 0.4, cv2.cvtColor(og, cv2.COLOR_GRAY2BGR), 0.6, 0)
    cv2.imwrite(f"{args.output_dir}/static_overlay.png", ov)
    sp = normalize_to_uint8(slope_map)
    cv2.imwrite(f"{args.output_dir}/slope_gray.png", sp)
    cv2.imwrite(f"{args.output_dir}/slope_color.png", cv2.applyColorMap(sp, args.grad_cmap))
    dm = dyn_mask.astype(np.uint8) * 255
    cv2.imwrite(f"{args.output_dir}/dynamic_mask.png", dm)
    b2 = cv2.cvtColor(og, cv2.COLOR_GRAY2BGR)
    b2[dyn_mask] = (0, 255, 255)
    dov = cv2.addWeighted(b2, 0.4, cv2.cvtColor(og, cv2.COLOR_GRAY2BGR), 0.6, 0)
    cv2.imwrite(f"{args.output_dir}/dynamic_overlay.png", dov)


def main():
    p = argparse.ArgumentParser(description="Visualize static and dynamic leak regions.")
    p.add_argument('mat_file')
    p.add_argument('output_dir')
    p.add_argument('--key', default='TempFrames')
    p.add_argument('--static_quantile', type=float, default=0.95,
                   help='Quantile to extract static hotspot from mean frame')
    p.add_argument('--dynamic_quantile', type=float, default=0.95,
                   help='Quantile to extract dynamic hotspot from slope map')
    p.add_argument('--morph_kernel', type=int, default=5,
                   help='Morphology kernel size for cleanup')
    p.add_argument('--orig_colormap', default='hot')
    p.add_argument('--grad_colormap', default='inferno')
    p.add_argument('--fps', type=float, default=30.0)
    args = p.parse_args()

    data = scipy.io.loadmat(args.mat_file)
    frames = data.get(args.key)
    if frames is None or frames.ndim != 3:
        print('Invalid frames'); sys.exit(1)

    orig = frames[:, :, 0].astype(np.float64)
    mean_map = compute_mean_frame(frames)

    # STATIC ROI: real hotspot on mean frame
    static_mask, _ = extract_hotspot_from_map(
        mean_map,
        threshold_quantile=args.static_quantile,
        morph_kernel=args.morph_kernel
    )

    # DYNAMIC ROI: slope map, then hotspot
    slope_map = compute_slope_map(frames, fps=args.fps)
    masked_slope = slope_map.copy()
    masked_slope[~static_mask] = np.nan
    dyn_mask, _ = extract_hotspot_from_map(
        masked_slope,
        threshold_quantile=args.dynamic_quantile,
        morph_kernel=args.morph_kernel
    )

    # prepare colormaps
    args.orig_cmap = getattr(cv2, f'COLORMAP_{args.orig_colormap.upper()}', cv2.COLORMAP_HOT)
    args.grad_cmap = getattr(cv2, f'COLORMAP_{args.grad_colormap.upper()}', cv2.COLORMAP_INFERNO)

    save_visuals(orig, mean_map, slope_map, static_mask, dyn_mask, args)
    print('Done.')

if __name__ == '__main__':
    main()
