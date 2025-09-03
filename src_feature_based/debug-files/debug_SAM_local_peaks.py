# scripts/debug_simple_SAM.py
"""
UNIVERSAL & FLEXIBLE DEBUG SCRIPT (v5 - with Presentation Visuals)
- Adds a '--create_presentation_visual' flag to generate a 2x3 storyboard
  of the entire segmentation pipeline for a single video.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, mstats
from tqdm import tqdm
from joblib import Parallel, delayed

try:
    from segment_anything import sam_model_registry, SamPredictor
    from skimage.feature import peak_local_max
except ImportError:
    print("Error: Required libraries not found. Please run:", file=sys.stderr)
    print("pip install scikit-image segment-anything-py imageio", file=sys.stderr)
    sys.exit(1)

# ========================================================================================
# --- HELPER & CORE FUNCTIONS ---
# ========================================================================================

def save_parameters(args, output_dir):
    # ... (unchanged)
    params_path = os.path.join(output_dir, "parameters.txt")
    with open(params_path, 'w') as f:
        f.write("--- Run Parameters ---\n")
        for arg, value in sorted(vars(args).items()): f.write(f"{arg}: {value}\n")
    print(f"Saved run parameters to: {params_path}")

def apply_spatial_filter(frames, filter_type='none', d=9, sigmaColor=75, sigmaSpace=75):
    # ... (unchanged)
    if filter_type.lower() == 'none': return frames
    print(f"  - Step 1: Applying '{filter_type}' spatial filter...")
    H, W, T = frames.shape
    filtered_frames = np.zeros_like(frames, dtype=np.float64)
    for i in tqdm(range(T), desc="    Filtering Spatially", leave=False, ncols=100):
        frame_8bit = cv2.normalize(frames[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if filter_type.lower() == 'bilateral':
            filtered_frame_8bit = cv2.bilateralFilter(frame_8bit, d, sigmaColor, sigmaSpace)
        else: filtered_frame_8bit = frame_8bit
        min_val, max_val = np.min(frames[:,:,i]), np.max(frames[:,:,i])
        filtered_frames[:, :, i] = cv2.normalize(filtered_frame_8bit.astype(np.float64), None, min_val, max_val, cv2.NORM_MINMAX)
    return filtered_frames

def apply_temporal_smoothing(frames, window_size):
    # ... (unchanged)
    if window_size <= 1: return frames
    print(f"  - Step 2: Applying temporal smoothing (window: {window_size})...")
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
    # ... (unchanged)
    W = row_data.shape[0]; row_values = np.zeros(W, dtype=np.float64)
    for c in range(W):
        pixel_series = row_data[c, :]; val = 0.0
        if not np.any(np.isnan(pixel_series)) and len(pixel_series) > 1:
            try:
                if method == 'theil_sen': val, _, _, _ = mstats.theilslopes(pixel_series, t, 0.95)
                elif method == 'kendall_tau': val, _ = kendalltau(t, pixel_series)
            except (ValueError, IndexError): pass
            row_values[c] = val if np.isfinite(val) else 0.0
    return row_values

def generate_activity_map(frames, method, env_para):
    # ... (unchanged)
    H, W, T = frames.shape; t = np.arange(T)
    if T < 2: return np.zeros((H, W), dtype=np.float64)
    print(f"  - Step 3: Calculating activity map using '{method}'...")
    results = Parallel(n_jobs=-1)(delayed(_calculate_slope_for_row)(frames[r, :, :], t, method) for r in tqdm(range(H), desc=f"    Processing Rows", leave=False, ncols=100))
    raw_activity_map = np.vstack(results)
    if env_para == 1: activity_map = raw_activity_map.copy(); activity_map[activity_map < 0] = 0
    elif env_para == -1: activity_map = -raw_activity_map; activity_map[activity_map < 0] = 0
    else: activity_map = np.abs(raw_activity_map)
    return activity_map

def create_border_roi_mask(frame_shape, border_percent):
    # ... (unchanged)
    print(f"  - Step 4: Creating {border_percent*100:.0f}% border ROI mask...")
    H, W = frame_shape
    border_h, border_w = int(H * border_percent), int(W * border_percent)
    roi_mask = np.zeros(frame_shape, dtype=np.uint8)
    roi_mask[border_h : H - border_h, border_w : W - border_w] = 1
    return roi_mask

def create_panel_roi_mask(median_frame, erosion_iterations):
    # ... (unchanged)
    print("  - Step 4: Creating Panel ROI Mask via Otsu's Thresholding...")
    frame_norm = cv2.normalize(median_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, thresh = cv2.threshold(frame_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    if num_labels <= 1: return np.ones_like(median_frame, dtype=np.uint8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    panel_mask = np.zeros_like(thresh); panel_mask[labels == largest_label] = 1
    if erosion_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        panel_mask = cv2.erode(panel_mask, kernel, iterations=erosion_iterations)
    return panel_mask

def find_prompts_by_quantile(activity_map, num_prompts, quantile_thresh):
    # ... (unchanged)
    print(f"  - Step 5: Finding prompts via global quantile (threshold: {quantile_thresh:.3f})...")
    if activity_map is None or not np.any(activity_map > 1e-9): return [], []
    active_pixels = activity_map[activity_map > 1e-9]
    if active_pixels.size == 0: return [], []
    activity_threshold = np.quantile(active_pixels, quantile_thresh)
    binary_mask = (activity_map >= activity_threshold).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8); binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
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
    # ... (unchanged)
    print(f"  - Step 5: Finding prompts via local peaks (min_dist: {min_distance}, threshold: {abs_threshold})...")
    coordinates = peak_local_max(activity_map, min_distance=min_distance, threshold_abs=abs_threshold)
    if coordinates.size == 0: return [], []
    all_candidates = []
    for r, c in coordinates: all_candidates.append({'centroid': np.array([c, r]), 'peak_activity': activity_map[r, c]})
    sorted_candidates = sorted(all_candidates, key=lambda x: x['peak_activity'], reverse=True)
    top_candidates = sorted_candidates[:num_prompts]
    H, W = activity_map.shape
    for cand in top_candidates:
        center_x, center_y = cand['centroid']; box_half = prompt_box_size // 2
        x1 = int(np.clip(center_x - box_half, 0, W-1)); y1 = int(np.clip(center_y - box_half, 0, H-1))
        w = int(np.clip(center_x + box_half, 0, W-1)) - x1; h = int(np.clip(center_y + box_half, 0, H-1)) - y1
        cand['stat'] = [x1, y1, w, h]
    return top_candidates, all_candidates

def run_sam_with_box_and_point_prompts(frame_rgb, top_candidates, predictor):
    # ... (unchanged)
    all_final_masks = []; predictor.set_image(frame_rgb)
    for cand in top_candidates:
        stat = cand.get('stat'); input_box = None
        if stat is not None: x1, y1, w, h = stat; input_box = np.array([x1, y1, x1 + w, y1 + h])
        point_coords = cand['centroid'].reshape(1, 2); point_labels = np.array([1])
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=input_box[None, :] if input_box is not None else None, multimask_output=False)
        if len(masks) > 0: all_final_masks.append(masks[0])
    return all_final_masks

# --- NEW: Function to enhance frames for visualization ---
def enhance_frame(frame):
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(frame_normalized)

# --- NEW: Function to generate the 2x3 presentation visual ---
def create_presentation_visual(artifacts, save_path):
    print("\n--- Creating Presentation Visual Storyboard ---")
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle("Automated Segmentation Pipeline", fontsize=24, y=0.97)
    
    titles = [
        "1. Raw Input (Enhanced)", "2. Pre-processed Data (Smoothed)", "3. Activity Map (Kendall Tau)",
        "4. ROI Mask Applied", "5. Prompts Generated", "6. Final Segmentation"
    ]
    
    # Panel 1: Raw Input
    axes[0, 0].imshow(artifacts['enhanced_frame'], cmap='gray')
    
    # Panel 2: Pre-processed Data
    axes[0, 1].imshow(artifacts['smoothed_frame'], cmap='gray')

    # Panel 3: Activity Map
    axes[0, 2].imshow(artifacts['activity_map'], cmap='hot')
    
    # Panel 4: ROI Mask Applied
    axes[1, 0].imshow(artifacts['masked_activity_map'], cmap='hot')
    
    # Panel 5: Prompts Generated
    ax = axes[1, 1]
    ax.imshow(artifacts['masked_activity_map'], cmap='hot')
    if artifacts['all_candidates']:
        ax.scatter([c['centroid'][0] for c in artifacts['all_candidates']], [c['centroid'][1] for c in artifacts['all_candidates']], s=100, facecolors='none', edgecolors='cyan', lw=1.5)
    if artifacts['final_prompts']:
        ax.scatter(np.array([p['centroid'] for p in artifacts['final_prompts']])[:, 0], np.array([p['centroid'] for p in artifacts['final_prompts']])[:, 1], s=600, c='yellow', marker='*', edgecolor='black', zorder=5)

    # Panel 6: Final Segmentation
    axes[1, 2].imshow(cv2.cvtColor(artifacts['segmentation_image'], cv2.COLOR_BGR2RGB))
    
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i], fontsize=16)
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"  - Saved presentation visual to: {save_path}")

# ========================================================================================
# --- MAIN EXECUTION SCRIPT ---
# ========================================================================================
def main(args):
    # ... (Setup and model loading) ...
    print("--- Universal Hotspot Segmentation ---"); os.makedirs(args.output_dir, exist_ok=True)
    save_parameters(args, args.output_dir); print("\nLoading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint); device = "cuda" if torch.cuda.is_available() else "cpu"; sam.to(device=device)
    predictor = SamPredictor(sam)
    frames = scipy.io.loadmat(args.input)['TempFrames'].astype(np.float64); H, W, T = frames.shape
    
    # --- Execute Pipeline & Store Artifacts ---
    artifacts = {}
    artifacts['enhanced_frame'] = enhance_frame(frames[:, :, 0])
    
    spatially_filtered_frames = apply_spatial_filter(frames, args.spatial_filter)
    temporally_smoothed_frames = apply_temporal_smoothing(spatially_filtered_frames, args.temporal_smooth_window)
    artifacts['smoothed_frame'] = enhance_frame(temporally_smoothed_frames[:, :, 0])
    
    activity_map = generate_activity_map(temporally_smoothed_frames, args.activity_method, args.env_para)
    artifacts['activity_map'] = activity_map
    
    target_frame = np.median(frames, axis=2)
    if args.roi_method == 'border': roi_mask = create_border_roi_mask((H, W), args.roi_border_percent)
    else: roi_mask = create_panel_roi_mask(target_frame, args.roi_erosion)
    
    masked_activity_map = activity_map * roi_mask
    artifacts['masked_activity_map'] = masked_activity_map
    
    if args.prompt_method == 'quantile':
        top_candidates, all_candidates = find_prompts_by_quantile(masked_activity_map, args.num_leaks, args.activity_quantile)
    else:
        top_candidates, all_candidates = find_prompts_by_local_peaks(masked_activity_map, args.num_leaks, args.peak_min_distance, args.peak_abs_threshold, args.prompt_box_size)
    
    artifacts['all_candidates'] = all_candidates
    artifacts['final_prompts'] = top_candidates

    if not top_candidates: print("Could not find any significant prompts. Exiting.", file=sys.stderr); return

    print("\nRunning SAM segmentation...")
    frame_normalized_8bit = cv2.normalize(target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    frame_rgb = cv2.cvtColor(frame_normalized_8bit, cv2.COLOR_GRAY2BGR)
    all_final_masks = run_sam_with_box_and_point_prompts(frame_rgb, top_candidates, predictor)

    if not all_final_masks: print("SAM did not generate masks. Exiting.", file=sys.stderr); return

    print("\nSaving final outputs...")
    segmentation_image = frame_rgb.copy(); colors = [[255, 255, 0], [0, 255, 255]]
    for i, mask in enumerate(all_final_masks):
        color = colors[i % len(colors)]; color_overlay = np.zeros_like(segmentation_image); color_overlay[mask] = color
        segmentation_image = cv2.addWeighted(segmentation_image, 1, color_overlay, 0.6, 0)
        mask_save_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.input))[0]}_mask_{i}.npy")
        np.save(mask_save_path, mask)
    
    artifacts['segmentation_image'] = segmentation_image

    # --- Generate Visuals if Requested ---
    if args.create_presentation_visual:
        save_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.input))[0]}_pipeline_visual.png")
        create_presentation_visual(artifacts, save_path)
    
    print("\n--- Script finished. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal script for hotspot segmentation.")
    
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--output_dir", required=True, type=str)
    
    param_group = parser.add_argument_group('Pipeline Control Parameters')
    param_group.add_argument("--num_leaks", type=int, default=1)
    param_group.add_argument("--env_para", type=int, default=1, choices=[1, -1, 0])
    param_group.add_argument("--activity_method", type=str, default="kendall_tau", choices=['theil_sen', 'kendall_tau'])
    param_group.add_argument("--temporal_smooth_window", type=int, default=3)
    param_group.add_argument("--spatial_filter", type=str, default="bilateral", choices=['none', 'bilateral'])
    param_group.add_argument("--roi_method", type=str, default="otsu", choices=['otsu', 'border'])
    param_group.add_argument("--roi_border_percent", type=float, default=0.1)
    param_group.add_argument("--roi_erosion", type=int, default=3)

    prompt_group = parser.add_argument_group('Prompt Finding Method')
    prompt_group.add_argument("--prompt_method", type=str, default="quantile", choices=['quantile', 'local_peak'])
    prompt_group.add_argument("--activity_quantile", type=float, default=0.995)
    prompt_group.add_argument("--peak_min_distance", type=int, default=50)
    prompt_group.add_argument("--peak_abs_threshold", type=float, default=0.2)
    prompt_group.add_argument("--prompt_box_size", type=int, default=30)
    
    # --- NEW FLAG FOR VISUALIZATION ---
    parser.add_argument("--create_presentation_visual", action='store_true', 
                        help="If set, generates a 2x3 composite image visualizing the entire pipeline.")

    args = parser.parse_args()
    main(args)

"""
python src_feature_based/debug-files/debug_SAM_local_peaks.py \
  --input /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_HardyBoard_08132025_2holes_noshutter/T1.4V_2025-08-15-18-45-57_21_32_11_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --output_dir Output_SAM/failing_masks/Fluke_HardyBoard_08132025_2holes_noshutter/T1.4V_2025-08-15-18-45-57_21_32_11/iter-2-local-peak \
  --num_leaks 2 \
  --roi_method border \
  --roi_border_percent 0.10 \
  --prompt_method local_peak \
  --peak_min_distance 50 \
  --create_presentation_visual \
  --peak_abs_threshold 0.2  
  
"""