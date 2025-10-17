# run_SAM.py
"""
FINAL UNIVERSAL BATCH-PROCESSING SCRIPT (v7 - With Multi-Hole Dynamic Mode)
- Maintains all original functionality (manual, quantile, local_peak modes).
- Adds a 'multi_hole_filter' processing mode for complex datasets.
- This mode dynamically finds all leaks, filters unwanted ones, and assigns correct hole IDs.
- Adds a new visualization for verifying hole ID assignment in the new mode.
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
from scipy.spatial.distance import cdist

try:
    from segment_anything import sam_model_registry, SamPredictor
    from skimage.feature import peak_local_max
except ImportError:
    print("Error: Required libraries not found. Please run:", file=sys.stderr)
    print("pip install scikit-image segment-anything-py imageio", file=sys.stderr)
    sys.exit(1)

# --- Coordinates for Multi-Hole Filtering & Assignment ---
MANUAL_PROMPTS = {
        "Fluke_BrickCladding_2holes_0808_2025_noshutter": [
            {"name": "center_hole", "coord": (272, 328), "box_size": 30},
            {"name": "bottom_left_hole", "coord": (360, 140), "box_size": 15}
        ],
        "Fluke_BrickCladding_2holes_0805_2025_noshutter": [
            {"name": "center_hole", "coord": (274, 328), "box_size": 30},
            {"name": "bottom_left_hole", "coord": (360, 140), "box_size": 15}
        ]
    }

SLIT_COORDS = {
    'Hole_1': np.array([233, 553]),
    'Hole_10': np.array([414, 353]),
}

TARGET_HOLE_COORDS = {
    2: np.array([79, 507]),
    3: np.array([53, 344]),
    4: np.array([78, 181]),
    5: np.array([213, 219]),
    6: np.array([332, 149]),
    7: np.array([326, 307]),
    8: np.array([325, 453]),
    9: np.array([201, 355]),
}

# ========================================================================================
# --- HELPER & CORE FUNCTIONS ---
# ========================================================================================

def save_parameters(args, output_dir):
    params_path = os.path.join(output_dir, "parameters.txt")
    with open(params_path, 'w') as f:
        f.write("--- Run Parameters ---\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")

def apply_spatial_filter(frames, filter_type='none', d=9, sigmaColor=75, sigmaSpace=75):
    if filter_type.lower() == 'none':
        return frames
    print(f"\n  - Applying '{filter_type}' spatial filter...\n")
    H, W, T = frames.shape
    filtered_frames = np.zeros_like(frames, dtype=np.float64)
    for i in tqdm(range(T), desc="    Filtering Spatially", leave=False, ncols=100):
        frame_8bit = cv2.normalize(frames[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if filter_type.lower() == 'bilateral':
            filtered_frame_8bit = cv2.bilateralFilter(frame_8bit, d, sigmaColor, sigmaSpace)
        else:
            filtered_frame_8bit = frame_8bit
        min_val, max_val = np.min(frames[:, :, i]), np.max(frames[:, :, i])
        filtered_frames[:, :, i] = cv2.normalize(filtered_frame_8bit.astype(np.float64), None, min_val, max_val, cv2.NORM_MINMAX)
    return filtered_frames

def apply_temporal_smoothing(frames, window_size):
    if window_size <= 1:
        return frames
    print(f"\n  - Applying temporal smoothing (window: {window_size})...\n")
    H, W, T = frames.shape
    smoothed_frames = np.zeros_like(frames, dtype=np.float64)
    kernel = np.ones(window_size) / window_size
    for r in tqdm(range(H), desc="    Smoothing Temporally", leave=False, ncols=100):
        for c in range(W):
            pixel_series = frames[r, c, :]
            smoothed_series = np.convolve(pixel_series, kernel, mode='valid')
            pad_before = (T - len(smoothed_series)) // 2
            pad_after = T - len(smoothed_series) - pad_before
            smoothed_frames[r, c, :] = np.pad(smoothed_series, (pad_before, pad_after), mode='edge')
    return smoothed_frames

def _calculate_slope_for_row(row_data, t, method):
    W = row_data.shape[0]
    row_values = np.zeros(W, dtype=np.float64)
    for c in range(W):
        pixel_series = row_data[c, :]
        val = 0.0
        if not np.any(np.isnan(pixel_series)) and len(pixel_series) > 1:
            try:
                if method == 'theil_sen':
                    val, _, _, _ = mstats.theilslopes(pixel_series, t, 0.95)
                elif method == 'kendall_tau':
                    val, _ = kendalltau(t, pixel_series)
            except (ValueError, IndexError):
                pass
            row_values[c] = val if np.isfinite(val) else 0.0
    return row_values

def generate_activity_map(frames, method, env_para):
    H, W, T = frames.shape
    t = np.arange(T)
    if T < 2:
        return np.zeros((H, W), dtype=np.float64)
    print(f"\n  - Calculating activity map using '{method}'...\n")
    results = Parallel(n_jobs=-1)(delayed(_calculate_slope_for_row)(frames[r, :, :], t, method) for r in tqdm(range(H), desc=f"    Processing Rows", leave=False, ncols=100))
    raw_activity_map = np.vstack(results)
    if env_para == 1:
        activity_map = raw_activity_map.copy()
        activity_map[activity_map < 0] = 0
    elif env_para == -1:
        activity_map = -raw_activity_map
        activity_map[activity_map < 0] = 0
    else:
        activity_map = np.abs(raw_activity_map)
    return activity_map

def create_border_roi_mask(frame_shape, border_percent):
    print(f"  - Creating {border_percent*100:.0f}% border ROI mask...")
    H, W = frame_shape
    border_h, border_w = int(H * border_percent), int(W * border_percent)
    roi_mask = np.zeros(frame_shape, dtype=np.uint8)
    roi_mask[border_h: H - border_h, border_w: W - border_w] = 1
    return roi_mask

def create_panel_roi_mask(median_frame, erosion_iterations):
    print("\n  - Creating Panel ROI Mask via Otsu's Thresholding...\n")
    frame_norm = cv2.normalize(median_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, thresh = cv2.threshold(frame_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    if num_labels <= 1:
        return np.ones_like(median_frame, dtype=np.uint8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    panel_mask = np.zeros_like(thresh)
    panel_mask[labels == largest_label] = 1
    if erosion_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        panel_mask = cv2.erode(panel_mask, kernel, iterations=erosion_iterations)
    return panel_mask

def find_prompts_by_quantile(activity_map, num_prompts, quantile_thresh):
    print(f"\n  - Finding prompts via global quantile (threshold: {quantile_thresh:.3f})...\n")
    if activity_map is None or not np.any(activity_map > 1e-9): return [], []
    active_pixels = activity_map[activity_map > 1e-9]
    if active_pixels.size == 0: return [], []
    activity_threshold = np.quantile(active_pixels, quantile_thresh)
    binary_mask = (activity_map >= activity_threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
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

def find_prompts_by_local_peaks(activity_map, num_prompts, min_distance, abs_threshold, prompt_box_size):
    print(f"\n  - Finding prompts via local peaks (min_dist: {min_distance}, threshold: {abs_threshold})...")
    coordinates = peak_local_max(activity_map, min_distance=min_distance, threshold_abs=abs_threshold)
    if coordinates.size == 0: return [], []
    all_candidates = []
    for r, c in coordinates:
        all_candidates.append({'centroid': np.array([c, r]), 'peak_activity': activity_map[r, c]})
    sorted_candidates = sorted(all_candidates, key=lambda x: x['peak_activity'], reverse=True)
    top_candidates = sorted_candidates[:num_prompts]
    H, W = activity_map.shape
    for cand in top_candidates:
        center_x, center_y = cand['centroid']; box_half = prompt_box_size // 2
        x1 = int(np.clip(center_x - box_half, 0, W-1)); y1 = int(np.clip(center_y - box_half, 0, H-1))
        w = int(np.clip(center_x + box_half, 0, W-1)) - x1; h = int(np.clip(center_y + box_half, 0, H-1)) - y1
        cand['stat'] = [x1, y1, w, h]
    return top_candidates, sorted_candidates

def find_prompts_manually(manual_prompts_db, dataset_key, frame_shape):
    print(f"\n  - Finding prompts via manual coordinates for key '{dataset_key}'...")
    leaks_to_use = None
    for key in manual_prompts_db:
        if key in dataset_key:
            leaks_to_use = manual_prompts_db[key]
            break
    if not leaks_to_use:
        print(f"  Warning: No manual coordinates found for '{dataset_key}'. Returning no prompts.")
        return [], []
    print(f"  Found {len(leaks_to_use)} manual leak definitions.")
    top_candidates = []
    H, W = frame_shape
    for leak_def in leaks_to_use:
        r, c = leak_def["coord"]
        box_size = leak_def["box_size"]
        cand = {'centroid': np.array([c, r]), 'peak_activity': 999, 'name': leak_def.get("name")}
        box_half = box_size // 2
        x1 = int(np.clip(c - box_half, 0, W-1)); y1 = int(np.clip(r - box_half, 0, H-1))
        w = int(np.clip(c + box_half, 0, W-1)) - x1; h = int(np.clip(r + box_half, 0, H-1)) - y1
        cand['stat'] = [x1, y1, w, h]
        top_candidates.append(cand)
    return top_candidates, []

def find_all_local_peaks(activity_map, min_distance, abs_threshold):
    coordinates = peak_local_max(activity_map, min_distance=min_distance, threshold_abs=abs_threshold)
    if coordinates.size == 0: return []
    all_candidates = [{'centroid': np.array([c, r]), 'peak_activity': activity_map[r, c]} for r, c in coordinates]
    return sorted(all_candidates, key=lambda x: x['peak_activity'], reverse=True)

def filter_and_assign_prompts(all_candidates, slit_coords, target_hole_coords, exclusion_radius=30):
    filtered_prompts = []
    slit_locations = np.array([coord[::-1] for coord in slit_coords.values()]) # Convert to (x, y)
    for cand in all_candidates:
        if np.all(cdist(cand['centroid'].reshape(1, 2), slit_locations) > exclusion_radius):
            filtered_prompts.append(cand)
    
    assigned_prompts = {}
    remaining_prompts = list(filtered_prompts)
    for hole_id, target_coord in target_hole_coords.items():
        target_xy = target_coord[::-1]
        if not remaining_prompts: break
        distances = [np.linalg.norm(p['centroid'] - target_xy) for p in remaining_prompts]
        closest_prompt_idx = np.argmin(distances)
        closest_prompt = remaining_prompts.pop(closest_prompt_idx)
        closest_prompt['hole_id'] = hole_id
        assigned_prompts[hole_id] = closest_prompt
        
    return [assigned_prompts[key] for key in sorted(assigned_prompts.keys())]

def visualize_prompts(activity_map, all_candidates, final_prompts, save_path, method):
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(activity_map, cmap='hot', vmin=0)
    label = 'Activity (Theil-Sen Slope)' if method == 'theil_sen' else 'Activity (Kendall Tau Corr.)'
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title('Debug Map: Prompts from Most Intense Blobs')
    if all_candidates:
        ax.scatter([c['centroid'][0] for c in all_candidates], [c['centroid'][1] for c in all_candidates], s=100, facecolors='none', edgecolors='cyan', lw=1.5, label='All Found Blobs')
    if final_prompts:
        ax.scatter(np.array([p['centroid'] for p in final_prompts])[:, 0], np.array([p['centroid'] for p in final_prompts])[:, 1], s=600, c='yellow', marker='*', edgecolor='black', label='Final Selected Prompts', zorder=5)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)

def visualize_assigned_prompts(activity_map, all_candidates, final_prompts, save_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(activity_map, cmap='hot', vmin=0)
    fig.colorbar(im, ax=ax, label='Activity (Kendall Tau Corr.)')
    ax.set_title('Debug Map: Prompts Assigned & Labeled')
    if all_candidates:
        ax.scatter([c['centroid'][0] for c in all_candidates], [c['centroid'][1] for c in all_candidates], s=150, facecolors='none', edgecolors='cyan', lw=1.5, label='All Found Peaks')
    if final_prompts:
        ax.scatter([p['centroid'][0] for p in final_prompts], [p['centroid'][1] for p in final_prompts], s=600, c='yellow', marker='*', edgecolor='black', zorder=5, label='Assigned Prompts')
        for p in final_prompts:
            ax.text(p['centroid'][0] + 15, p['centroid'][1] + 15, str(p['hole_id']), color='white', fontsize=16, fontweight='bold')
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)

def run_sam_with_box_and_point_prompts(frame_rgb, top_candidates, predictor, prompt_box_size):
    all_final_masks = []
    predictor.set_image(frame_rgb)
    H, W, _ = frame_rgb.shape
    for cand in top_candidates:
        center_x, center_y = cand['centroid']; box_half = prompt_box_size // 2
        x1 = int(np.clip(center_x - box_half, 0, W-1)); y1 = int(np.clip(center_y - box_half, 0, H-1))
        x2 = int(np.clip(center_x + box_half, 0, W-1)); y2 = int(np.clip(center_y + box_half, 0, H-1))
        input_box = np.array([x1, y1, x2, y2])
        point_coords = cand['centroid'].reshape(1, 2); point_labels = np.array([1])
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=input_box[None, :], multimask_output=False)
        if len(masks) > 0:
            all_final_masks.append(masks[0])
    return all_final_masks

def main(args):

    print("--- Starting Automated Hotspot Segmentation Batch Process ---")
    os.makedirs(args.base_output_dir, exist_ok=True)
    print("Loading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for root, _, files in os.walk(args.dataset_dir):
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
                frames = scipy.io.loadmat(video_path)["TempFrames"].astype(np.float64)
                H, W, T = frames.shape
                if T < 2: raise ValueError("Not enough frames.")
            except Exception as e:
                print(f"  Error loading {video_filename}: {e}. Skipping.", file=sys.stderr)
                continue

            target_frame = np.median(frames, axis=2)
            
            spatially_filtered_frames = apply_spatial_filter(frames, args.spatial_filter)
            temporally_smoothed_frames = apply_temporal_smoothing(spatially_filtered_frames, args.temporal_smooth_window)
            activity_map = generate_activity_map(temporally_smoothed_frames, args.activity_method, args.env_para)
            
            if args.roi_method == 'border':
                roi_mask = create_border_roi_mask((H, W), args.roi_border_percent)
            else:
                roi_mask = create_panel_roi_mask(target_frame, args.roi_erosion)
            masked_activity_map = activity_map * roi_mask

            final_prompts, all_candidates = [], []

            if args.processing_mode == 'multi_hole_filter':
                print("\n  - Running in 'multi_hole_filter' mode.")
                all_candidates = find_all_local_peaks(masked_activity_map, args.peak_min_distance, args.peak_abs_threshold)
                final_prompts = filter_and_assign_prompts(all_candidates, SLIT_COORDS, TARGET_HOLE_COORDS)
                debug_plot_path = os.path.join(video_output_dir, f"{base_filename}_prompt_assignment.png")
                visualize_assigned_prompts(masked_activity_map, all_candidates, final_prompts, debug_plot_path)
            else: # Standard and Manual modes
                if args.prompt_method == 'manual':
                    print("\n  - Running in 'manual' mode.")
                    dataset_key = os.path.basename(os.path.normpath(args.dataset_dir))
                    final_prompts, _ = find_prompts_manually(MANUAL_PROMPTS, dataset_key, (H, W))
                else:
                    print("\n  - Running in 'standard' dynamic mode.")
                    if args.prompt_method == 'quantile':
                        final_prompts, all_candidates = find_prompts_by_quantile(masked_activity_map, args.num_leaks, args.activity_quantile)
                    else: # local_peak
                        final_prompts, all_candidates = find_prompts_by_local_peaks(masked_activity_map, args.num_leaks, args.peak_min_distance, args.peak_abs_threshold, args.prompt_box_size)
                    
                    debug_plot_path = os.path.join(video_output_dir, f"{base_filename}_prompt_verification.png")
                    visualize_prompts(masked_activity_map, all_candidates, final_prompts, debug_plot_path, args.activity_method)

            if not final_prompts:
                print(f"  No prompts found for {video_filename}. Skipping.", file=sys.stderr)
                continue

            print("\nRunning SAM segmentation...")
            frame_normalized_8bit = cv2.normalize(target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_rgb = cv2.cvtColor(frame_normalized_8bit, cv2.COLOR_GRAY2BGR)
            all_final_masks = run_sam_with_box_and_point_prompts(frame_rgb, final_prompts, predictor, args.prompt_box_size)

            if not all_final_masks:
                print(f"  SAM did not generate masks for {video_filename}. Skipping save.", file=sys.stderr)
                continue

            print("\nSaving final outputs...")
            final_visualization_image = frame_rgb.copy()
            colors = [[255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 0, 128], [0, 128, 128]]
            
            for i, mask_info in enumerate(zip(final_prompts, all_final_masks)):
                cand, mask = mask_info
                hole_id = cand.get('hole_id', cand.get('name', f"mask_{i}"))

                color = colors[i % len(colors)]
                color_overlay = np.zeros_like(final_visualization_image)
                color_overlay[mask] = color
                final_visualization_image = cv2.addWeighted(final_visualization_image, 1, color_overlay, 0.6, 0)

                mask_save_path = os.path.join(video_output_dir, f"{base_filename}_mask_{hole_id}.npy")
                np.save(mask_save_path, mask)

            plot_save_path = os.path.join(video_output_dir, f"{base_filename}_sam_segmentation.png")
            cv2.imwrite(plot_save_path, final_visualization_image)
            print(f"  Saved {len(all_final_masks)} individual masks to {video_output_dir}")

    print("\n--- Batch Process Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process IR videos to segment thermal leaks.")
    
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--base_output_dir", required=True, type=str)
    default_checkpoint = os.path.join('SAM', 'sam_checkpoints', 'sam_vit_b_01ec64.pth')
    parser.add_argument("--checkpoint_path", type=str, default=default_checkpoint)
    parser.add_argument("--model_type", type=str, default="vit_b")

    parser.add_argument("--processing_mode", type=str, default="standard", choices=['standard', 'multi_hole_filter'],
                        help="Processing mode: 'standard' for 1-2 holes, 'multi_hole_filter' for 10-hole dataset.")

    param_group = parser.add_argument_group('Pipeline Control Parameters')
    param_group.add_argument("--num_leaks", type=int, default=1, help="Number of leaks to find in 'standard' mode.")
    param_group.add_argument("--env_para", type=int, default=1, choices=[1, -1, 0])
    param_group.add_argument("--activity_method", type=str, default="kendall_tau", choices=['theil_sen', 'kendall_tau'])
    param_group.add_argument("--spatial_filter", type=str, default="bilateral", choices=['none', 'bilateral'])
    param_group.add_argument("--temporal_smooth_window", type=int, default=3)
    param_group.add_argument("--roi_method", type=str, default="border", choices=['otsu', 'border'])
    param_group.add_argument("--roi_border_percent", type=float, default=0.1)
    param_group.add_argument("--roi_erosion", type=int, default=3)

    prompt_group = parser.add_argument_group('Prompt Finding Method')
    prompt_group.add_argument("--prompt_method", type=str, default="local_peak", choices=['quantile', 'local_peak', 'manual'],
                              help="Method to find prompts: 'quantile', 'local_peak' (adaptive), or 'manual' (hardcoded).")
    prompt_group.add_argument("--activity_quantile", type=float, default=0.995,
                              help="Quantile threshold for 'quantile' method.")
    prompt_group.add_argument("--peak_min_distance", type=int, default=50,
                              help="Min distance between peaks for 'local_peak' method.")
    prompt_group.add_argument("--peak_abs_threshold", type=float, default=0.2,
                              help="Min activity value for a peak in 'local_peak' method.")
    prompt_group.add_argument("--prompt_box_size", type=int, default=30,
                              help="Box size for 'local_peak' method.")

    args = parser.parse_args()
    main(args)
"""
python -m scripts.run_SAM --dataset_dir datasets/Fluke_Gypsum_07292025_noshutter \
    --base_output_dir output_SAM/datasets/local-peak/Fluke_Gypsum_07292025_noshutter \
    --num_leaks 1 \
    --roi_method border 
    
python -m scripts.run_SAM --dataset_dir datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter \
    --base_output_dir output_SAM/datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter \
    --num_leaks 2 

python -m scripts.run_SAM --dataset_dir /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0616_2025_noshutter \
    --base_output_dir output_SAM/datasets/Fluke_BrickCladding_2holes_0616_2025_noshutter \
    --num_leaks 2 \
    --roi_method border  

python -m scripts.run_SAM --dataset_dir /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0616_2025_noshutter \
    --base_output_dir output_SAM/datasets/Fluke_BrickCladding_2holes_0616_2025_noshutter \
    --num_leaks 2 \
    --roi_method border  

python -m scripts.run_SAM --dataset_dir /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_HardyBoard_08132025_2holes_noshutter \
    --base_output_dir output_SAM/datasets/Fluke_HardyBoard_08132025_2holes_noshutter \
    --num_leaks 2 \
    --roi_method border  \
    --prompt_box_size 25

python -m scripts.run_SAM --dataset_dir /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_Gypsum_09032025_10holes_noshutter \
    --base_output_dir output_SAM/datasets/Fluke_Gypsum_09032025_10holes_noshutter \
    --num_leaks 10 \
    --roi_method border \
    --processing_mode multi_hole_filter
    
"""
"""
Manual hotspot mask:

Fluke_BrickCladding_2holes_0808_2025_noshutter
python -m scripts.run_SAM \
    --dataset_dir /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter \
    --base_output_dir output_SAM/datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter-manual \
    --prompt_method manual

Fluke_BrickCladding_2holes_0805_2025_noshutter
python -m scripts.run_SAM \
    --dataset_dir /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0805_2025_noshutter \
    --base_output_dir output_SAM/datasets/Fluke_BrickCladding_2holes_0805_2025_noshutter-manual \
    --prompt_method manual

"""


"""
OLD GYPSUM:
python -m scripts.run_SAM --dataset_dir /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_Gypsum_03072025/FanPower_2.4V \
    --base_output_dir output_SAM/datasets/Fluke_Gypsum_03072025/FanPower_2.4V \
    --num_leaks 1 \
    --roi_method border 

OLD HARDYBOARD:
python -m scripts.run_SAM --dataset_dir /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_HardyBoard_03132025/2.4V \
    --base_output_dir output_SAM/datasets/Fluke_HardyBoard_03132025/2.4V \
    --num_leaks 1 \
    --roi_method border 


"""
