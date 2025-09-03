# run_SAM.py
"""
FINAL UNIVERSAL BATCH-PROCESSING SCRIPT (v2 - Corrected)
- Re-integrates the optional spatial filter (--spatial_filter).
- Contains the complete, flexible, and modular pipeline.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import torch
import matplotlib.pyplot as plt
import fnmatch
from scipy.stats import kendalltau, mstats
from tqdm import tqdm
from joblib import Parallel, delayed

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    try: from segment_anything import sam_model_registry, SamPredictor
    except ImportError: print(f"Error: segment_anything library not found.", file=sys.stderr); sys.exit(1)
    class MockConfig: MAT_FRAMES_KEY = 'TempFrames'
    old_cfg = MockConfig()


def save_parameters(args, output_dir):
    """Saves the command-line arguments to a text file."""
    params_path = os.path.join(output_dir, "parameters.txt")
    with open(params_path, 'w') as f:
        f.write("--- Run Parameters ---\n")
        for arg, value in sorted(vars(args).items()): f.write(f"{arg}: {value}\n")

def apply_spatial_filter(frames, filter_type='none', d=9, sigmaColor=75, sigmaSpace=75):
    """(Optional) Applies a spatial filter to each frame to reduce noise."""
    if filter_type.lower() == 'none':
        print("\n  - Step 1: Skipping spatial filter.")
        return frames
    print(f"  - Step 1: Applying '{filter_type}' spatial filter to each frame...")
    H, W, T = frames.shape
    filtered_frames = np.zeros_like(frames, dtype=np.float64)
    for i in tqdm(range(T), desc="    Filtering Frames Spatially", leave=False, ncols=100):
        frame_8bit = cv2.normalize(frames[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if filter_type.lower() == 'bilateral':
            filtered_frame_8bit = cv2.bilateralFilter(frame_8bit, d, sigmaColor, sigmaSpace)
        else: filtered_frame_8bit = frame_8bit
        min_val, max_val = np.min(frames[:,:,i]), np.max(frames[:,:,i])
        filtered_frames[:, :, i] = cv2.normalize(filtered_frame_8bit.astype(np.float64), None, min_val, max_val, cv2.NORM_MINMAX)
    return filtered_frames

def apply_temporal_smoothing(frames, window_size):
    """Applies a moving average filter along the time axis."""
    if window_size <= 1: return frames
    print(f"\n  - Step 2: Applying temporal smoothing (window: {window_size})...")
    H, W, T = frames.shape
    smoothed_frames = np.zeros_like(frames, dtype=np.float64)
    kernel = np.ones(window_size) / window_size
    for r in tqdm(range(H), desc="    Smoothing Frames Temporally", leave=False, ncols=100):
        for c in range(W):
            pixel_series = frames[r, c, :]
            smoothed_series = np.convolve(pixel_series, kernel, mode='valid')
            pad_before = (T - len(smoothed_series)) // 2
            pad_after = T - len(smoothed_series) - pad_before
            smoothed_frames[r, c, :] = np.pad(smoothed_series, (pad_before, pad_after), mode='edge')
    return smoothed_frames

def _calculate_slope_for_row(row_data, t, method):
    """Helper for parallel calculation."""
    W = row_data.shape[0]
    row_values = np.zeros(W, dtype=np.float64)
    for c in range(W):
        pixel_series = row_data[c, :]
        if not np.any(np.isnan(pixel_series)) and len(pixel_series) > 1:
            try:
                if method == 'theil_sen': val, _, _, _ = mstats.theilslopes(pixel_series, t, 0.95)
                elif method == 'kendall_tau': val, _ = kendalltau(t, pixel_series)
                else: val = 0.0
            except (ValueError, IndexError): val = 0.0
            row_values[c] = val if np.isfinite(val) else 0.0
    return row_values

def generate_activity_map(frames, method, env_para):
    """Generates an activity map using the specified method."""
    H, W, T = frames.shape
    if T < 2: return np.zeros((H, W), dtype=np.float64)
    t = np.arange(T)
    print(f"\n  - Step 3: Calculating activity map using '{method}'...")
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_slope_for_row)(frames[r, :, :], t, method) for r in tqdm(range(H), desc=f"    Processing Rows ({method})", leave=False, ncols=100)
    )
    raw_activity_map = np.vstack(results)
    if env_para == 1: activity_map = raw_activity_map.copy(); activity_map[activity_map < 0] = 0
    elif env_para == -1: activity_map = -raw_activity_map; activity_map[activity_map < 0] = 0
    else: activity_map = np.abs(raw_activity_map)
    return activity_map

def create_border_roi_mask(frame_shape, border_percent):
    """Creates a simple binary mask that is False at the borders and True in the center."""
    print(f"\n  - Step 4: Creating {border_percent*100:.0f}% border ROI mask...")
    if not (0 <= border_percent < 0.5):
        print("    Warning: Invalid border percent. Using full frame for ROI.")
        return np.ones(frame_shape, dtype=np.uint8)
    H, W = frame_shape
    border_h, border_w = int(H * border_percent), int(W * border_percent)
    roi_mask = np.zeros(frame_shape, dtype=np.uint8)
    roi_mask[border_h : H - border_h, border_w : W - border_w] = 1
    return roi_mask

def create_panel_roi_mask(median_frame, erosion_iterations):
    """Creates a binary mask for the panel via Otsu's thresholding."""
    print("  - Step 4: Creating Panel ROI Mask via Otsu's Thresholding...")
    frame_norm = cv2.normalize(median_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, thresh = cv2.threshold(frame_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    if num_labels <= 1:
        print("    Warning: Otsu's method could not find a distinct object. Using full frame.", file=sys.stderr)
        return np.ones_like(median_frame, dtype=np.uint8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    panel_mask = np.zeros_like(thresh); panel_mask[labels == largest_label] = 1
    if erosion_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        panel_mask = cv2.erode(panel_mask, kernel, iterations=erosion_iterations)
    return panel_mask

def find_prompts_by_peak_activity(activity_map, num_prompts, quantile_thresh):
    """Finds prompts by selecting the N blobs with the highest peak activity."""
    print(f"\n  - Step 5: Finding the {num_prompts} most intense blobs (quantile: {quantile_thresh:.3f})...")
    if activity_map is None or not np.any(activity_map > 1e-9): return [], []
    active_pixels = activity_map[activity_map > 1e-9]
    if active_pixels.size == 0: return [], []
    activity_threshold = np.quantile(active_pixels, quantile_thresh)
    binary_mask = (activity_map >= activity_threshold).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1: return [], []
    all_candidates = []
    for i in range(1, num_labels):
        component_mask = (labels == i)
        peak_activity_value = np.max(activity_map[component_mask])
        all_candidates.append({'centroid': centroids[i], 'stat': stats[i], 'peak_activity': peak_activity_value})
    sorted_by_intensity = sorted(all_candidates, key=lambda x: x['peak_activity'], reverse=True)
    top_candidates = sorted_by_intensity[:num_prompts]
    return top_candidates, all_candidates

def visualize_prompts(activity_map, all_candidates, final_prompts, save_path, method):
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(activity_map, cmap='hot', vmin=0)
    label = 'Activity (Theil-Sen Slope)' if method == 'theil_sen' else 'Activity (Kendall Tau Corr.)'
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title('Debug Map: Prompts from Most Intense Blobs')
    if all_candidates: ax.scatter([c['centroid'][0] for c in all_candidates], [c['centroid'][1] for c in all_candidates], s=100, facecolors='none', edgecolors='cyan', lw=1.5, label='All Found Blobs')
    if final_prompts: ax.scatter(np.array([p['centroid'] for p in final_prompts])[:, 0], np.array([p['centroid'] for p in final_prompts])[:, 1], s=600, c='yellow', marker='*', edgecolor='black', label='Final Selected Prompts', zorder=5)
    ax.legend(); fig.savefig(save_path); plt.close(fig)

# def run_sam_with_box_prompts(frame_rgb, top_candidates, predictor):
#     all_final_masks = []; predictor.set_image(frame_rgb)
#     for cand in top_candidates:
#         stat = cand['stat']
#         x1, y1 = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
#         x2, y2 = x1 + stat[cv2.CC_STAT_WIDTH], y1 + stat[cv2.CC_STAT_HEIGHT]
#         input_box = np.array([x1, y1, x2, y2])
#         masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False)
#         if len(masks) > 0: all_final_masks.append(masks[0])
#     return all_final_masks

## Less strict SAM prompts
def run_sam_with_box_and_point_prompts(frame_rgb, top_candidates, predictor):
    """
    Runs SAM using both a bounding box and a center point prompt.
    This encourages SAM to segment the entire plume within the box, not just the core.
    """
    all_final_masks = []; predictor.set_image(frame_rgb)
    for cand in top_candidates:
        stat = cand['stat']
        
        x1, y1 = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
        x2, y2 = x1 + stat[cv2.CC_STAT_WIDTH], y1 + stat[cv2.CC_STAT_HEIGHT]
        input_box = np.array([x1, y1, x2, y2])
        
        # Center Point (the centroid of the blob)
        point_coords = cand['centroid'].reshape(1, 2)
        point_labels = np.array([1]) # 1 indicates a positive foreground prompt

        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box[None, :], 
            multimask_output=False,
        )
        if len(masks) > 0:
            all_final_masks.append(masks[0])
            
    return all_final_masks

def main(args):
    """Main execution function to process all videos in a dataset."""
    print("--- Starting Automated Hotspot Segmentation Batch Process ---")
    os.makedirs(args.base_output_dir, exist_ok=True)

    print("Loading SAM model...")
    try:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"; sam.to(device=device)
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"Error loading SAM model: {e}", file=sys.stderr); sys.exit(1)

    for root, _, files in os.walk(args.dataset_dir):
        # to ignore macos metadata files:
        mat_files = [f for f in fnmatch.filter(files, '*.mat') if not f.startswith('._')]
        for video_filename in tqdm(mat_files, desc="Processing Videos"):
            video_path = os.path.join(root, video_filename)
            base_filename = os.path.splitext(video_filename)[0]
            
            relative_dir = os.path.relpath(root, args.dataset_dir)
            video_output_dir = os.path.join(args.base_output_dir, relative_dir, base_filename)
            os.makedirs(video_output_dir, exist_ok=True)
            
            save_parameters(args, video_output_dir)

            print(f"\n--- Processing: {video_filename} ---")

            try:
                mat_data = scipy.io.loadmat(video_path)
                frames = mat_data["TempFrames"].astype(np.float64)
                H, W, T = frames.shape
                if T < 2: raise ValueError("Not enough frames.")
            except Exception as e:
                print(f"  Error loading {video_filename}: {e}. Skipping.", file=sys.stderr); continue

            # Execute Pipeline 
            spatially_filtered_frames = apply_spatial_filter(frames, args.spatial_filter)
            temporally_smoothed_frames = apply_temporal_smoothing(spatially_filtered_frames, args.temporal_smooth_window)
            activity_map = generate_activity_map(temporally_smoothed_frames, args.activity_method, args.env_para)
            
            target_frame = np.median(frames, axis=2)
            
            # Select ROI Method 
            if args.roi_method == 'border':
                roi_mask = create_border_roi_mask((H, W), args.roi_border_percent)
            else: # Default to 'otsu'
                roi_mask = create_panel_roi_mask(target_frame, args.roi_erosion)

            masked_activity_map = activity_map * roi_mask
            
            top_candidates, all_candidates = find_prompts_by_peak_activity(
                masked_activity_map, args.num_leaks, args.activity_quantile
            )

            if not top_candidates:
                print(f"  No significant prompts found for {video_filename}. Skipping.", file=sys.stderr)
                continue

            debug_plot_path = os.path.join(video_output_dir, f"{base_filename}_prompt_verification.png")
            visualize_prompts(masked_activity_map, all_candidates, top_candidates, debug_plot_path, args.activity_method)

            print("  Running SAM segmentation...")
            frame_normalized_8bit = cv2.normalize(target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_rgb = cv2.cvtColor(frame_normalized_8bit, cv2.COLOR_GRAY2BGR)
            #all_final_masks = run_sam_with_box_prompts(frame_rgb, top_candidates, predictor)

            all_final_masks = run_sam_with_box_and_point_prompts(frame_rgb, top_candidates, predictor)

            if not all_final_masks:
                print(f"  SAM did not generate masks for {video_filename}. Skipping save.", file=sys.stderr)
                continue

            print("  Saving final outputs...")
            final_visualization_image = frame_rgb.copy()
            colors = [[255, 255, 0], [0, 255, 255], [255, 0, 255]]
            for i, mask in enumerate(all_final_masks):
                color = colors[i % len(colors)]
                color_overlay = np.zeros_like(final_visualization_image); color_overlay[mask] = color
                final_visualization_image = cv2.addWeighted(final_visualization_image, 1, color_overlay, 0.6, 0)
            plot_save_path = os.path.join(video_output_dir, f"{base_filename}_sam_segmentation.png")
            cv2.imwrite(plot_save_path, final_visualization_image)
            
            for i, mask in enumerate(all_final_masks):
                mask_save_path = os.path.join(video_output_dir, f"{base_filename}_mask_{i}.npy")
                np.save(mask_save_path, mask)
            print(f"  Saved {len(all_final_masks)} individual masks to {video_output_dir}")

    print("\n--- Batch Process Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process a dataset of IR videos to automatically segment thermal leaks.")
    
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--base_output_dir", required=True, type=str)
    
    param_group = parser.add_argument_group('Pipeline Control Parameters')
    param_group.add_argument("--num_leaks", type=int, default=2)
    param_group.add_argument("--env_para", type=int, default=0, choices=[1, -1, 0])
    param_group.add_argument("--activity_method", type=str, default="kendall_tau", choices=['theil_sen', 'kendall_tau'])
    param_group.add_argument("--spatial_filter", type=str, default="bilateral", choices=['none', 'bilateral'],
                              help="Optional spatial filter to apply to each frame before other processing.")
    param_group.add_argument("--temporal_smooth_window", type=int, default=3)
    param_group.add_argument("--activity_quantile", type=float, default=0.995)
    param_group.add_argument("--roi_method", type=str, default="otsu", choices=['otsu', 'border'])
    param_group.add_argument("--roi_border_percent", type=float, default=0.1)
    param_group.add_argument("--roi_erosion", type=int, default=3)

    default_checkpoint = os.path.join('SAM', 'sam_checkpoints', 'sam_vit_b_01ec64.pth')
    parser.add_argument("--checkpoint_path", type=str, default=default_checkpoint)
    parser.add_argument("--model_type", type=str, default="vit_b")

    args = parser.parse_args()
    
    if not os.path.isdir(args.dataset_dir):
        print(f"Error: Dataset directory not found at '{args.dataset_dir}'", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(args.checkpoint_path):
        print(f"Error: SAM checkpoint not found at '{args.checkpoint_path}'", file=sys.stderr); sys.exit(1)
        
    main(args)
"""
python -m scripts.run_SAM --dataset_dir datasets/Fluke_Gypsum_07292025_noshutter \
    --base_output_dir output_SAM/datasets/Fluke_Gypsum_07292025_noshutter \
    --num_leaks 1 \
    --roi_method border 
    
python -m scripts.run_SAM --dataset_dir datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter \
    --base_output_dir output_SAM/datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter \
    --num_leaks 2 

python -m scripts.run_SAM --dataset_dir /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_HardyBoard_08132025_2holes_noshutter \
    --base_output_dir output_SAM/datasets/Fluke_HardyBoard_08132025_2holes_noshutter \
    --num_leaks 2 \
    --roi_method border  
"""