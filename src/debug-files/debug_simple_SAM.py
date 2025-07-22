# universal_SAM_debug.py
"""
vid-2:
python src/debug-files/debug_simple_SAM.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-34-30_20_34_14_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple/vid-2/iter-5-tophat_before \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --activity_method kendall_tau \
  --spatial_filter bilateral \
  --tophat_filter_size 5

vid-4:
python src/debug-files/debug_simple_SAM.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-44-18_20_30_10_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple/vid-4/iter-4-tophat0 \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --activity_method kendall_tau \
  --spatial_filter bilateral \
  --tophat_filter_size 0

vid-5:
python src/debug-files/debug_simple_SAM.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-34-30_20_34_14_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple/vid-5/iter-1-tophat \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --activity_method kendall_tau \
  --spatial_filter bilateral \
  --tophat_filter_size 5

  
vid-6:
python src/debug-files/debug_simple_SAM.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-17-0-30_20_26_6_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple/vid-4/iter-2-tophat0 \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --activity_method kendall_tau \
  --spatial_filter bilateral \
  --tophat_filter_size 0

vid-8:
python src/debug-files/debug_simple_SAM.py \
  --input datasets/dataset_two_holes/T2.0V_6.3Pa_2025-6-16-17-16-39_20_34_14_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple/vid-8/iter-2-ROI \
  --temporal_smooth_window 3 \
  --roi_erosion 3

vid-9:
python src/debug-files/debug_simple_SAM.py \
  --input datasets/dataset_two_holes/T2.0V_6.3Pa_2025-6-16-17-25-13_20_30_10_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple/vid-9/iter-1-tophat0 \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --activity_method kendall_tau \
  --spatial_filter bilateral \
  --tophat_filter_size 0

vid-12
  python src/debug-files/debug_simple_SAM.py \
  --input datasets/dataset_two_holes/T2.0V_6.3Pa_2025-6-16-17-41-25_20_26_6_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple/vid-12/iter-1-KT-BT \
  --temporal_smooth_window 3 \
  --roi_erosion 3
  --activity_method kendall_tau \
  --spatial_filter bilateral \
  --tophat_filter_size 0

"""
"""
single-hole-dataset:
1.4/vid-1
  python src/debug-files/debug_simple_SAM.py \
  --input datasets/dataset_brickcladding/FanPower_1.8V/temp_2025-3-11-18-16-37_21.4_35_13.6_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple-single-hole/1.8V/vid-1/iter-1-ROI \
  --temporal_smooth_window 3 \
  --roi_erosion 3

2.4/vid-3:
  python src/debug-files/debug_simple_SAM.py \
  --input datasets/dataset_brickcladding/FanPower_2.4V/temp_2025-3-10-18-42-20_21.4_30_8.6_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple-single-hole/2.4V/vid-3/iter-1-ROI \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --num_leaks 1

1.8V/vid-2:
  python src/debug-files/debug_simple_SAM.py \
  --input datasets/dataset_brickcladding/FanPower_1.8V/temp_2025-3-11-18-24-29_21.4_32.5_11.1_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple-single-hole/1.8V/vid-2/iter-1-tophat0 \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --activity_method kendall_tau \
  --spatial_filter bilateral \
  --tophat_filter_size 0 \
  --num_leaks 1

"""


# debug_simple_SAM.py
"""
FINAL MODULAR & FLEXIBLE DEBUG SCRIPT
- Contains all processing steps as distinct functions.
- Includes selectable activity methods (Theil-Sen & Kendall's Tau).
- Includes optional Spatial Filtering (Bilateral) as a pre-processing step.
- Includes optional Top-Hat Filtering as a post-processing step on the activity map.
- Saves all run parameters to a 'parameters.txt' file for reproducibility.
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

# --- Ensure project modules are importable ---
try:
    import config
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    try: from segment_anything import sam_model_registry, SamPredictor
    except ImportError: print(f"Error: segment_anything library not found.", file=sys.stderr); sys.exit(1)
    class MockConfig: MAT_FRAMES_KEY = 'TempFrames'
    config = MockConfig()

# ========================================================================================
# --- HELPER & CORE FUNCTIONS ---
# ========================================================================================

def save_parameters(args, output_dir):
    """Saves the command-line arguments to a text file."""
    params_path = os.path.join(output_dir, "parameters.txt")
    with open(params_path, 'w') as f:
        f.write("--- Run Parameters ---\n")
        for arg, value in sorted(vars(args).items()): f.write(f"{arg}: {value}\n")
    print(f"Saved run parameters to: {params_path}")

def apply_spatial_filter(frames, filter_type='none', d=9, sigmaColor=75, sigmaSpace=75):
    """(Optional) Applies a spatial filter to each frame to reduce noise."""
    if filter_type.lower() == 'none':
        print("Step 1: Skipping spatial filter.")
        return frames
    
    print(f"Step 1: Applying '{filter_type}' spatial filter to each frame...")
    H, W, T = frames.shape
    filtered_frames = np.zeros_like(frames, dtype=np.float64)
    
    for i in tqdm(range(T), desc="  Filtering Frames Spatially", leave=False, ncols=100):
        frame_8bit = cv2.normalize(frames[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if filter_type.lower() == 'bilateral':
            filtered_frame_8bit = cv2.bilateralFilter(frame_8bit, d, sigmaColor, sigmaSpace)
        else: # If other filters are added, they would go here
            filtered_frame_8bit = frame_8bit
        
        # Convert back to original float64 range to maintain data integrity
        min_val, max_val = np.min(frames[:,:,i]), np.max(frames[:,:,i])
        filtered_frames[:, :, i] = cv2.normalize(filtered_frame_8bit.astype(np.float64), None, min_val, max_val, cv2.NORM_MINMAX)
        
    return filtered_frames

def apply_temporal_smoothing(frames, window_size):
    """Applies a moving average filter along the time axis."""
    if window_size <= 1: return frames
    print(f"Step 2: Applying temporal smoothing (window: {window_size})...")
    H, W, T = frames.shape
    smoothed_frames = np.zeros_like(frames, dtype=np.float64)
    kernel = np.ones(window_size) / window_size
    for r in tqdm(range(H), desc="  Smoothing Frames Temporally", leave=False, ncols=100):
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
    print(f"\nStep 3: Calculating activity map using '{method}'...")
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_slope_for_row)(frames[r, :, :], t, method) for r in tqdm(range(H), desc=f"  Processing Rows ({method})", leave=False, ncols=100)
    )
    raw_activity_map = np.vstack(results)
    if env_para == 1: activity_map = raw_activity_map.copy(); activity_map[activity_map < 0] = 0
    elif env_para == -1: activity_map = -raw_activity_map; activity_map[activity_map < 0] = 0
    else: activity_map = np.abs(raw_activity_map)
    return activity_map

def apply_tophat_filter(activity_map, kernel_size):
    """(Optional) Applies a White Top-Hat filter to the activity map."""
    if kernel_size <= 0:
        print("Step 4: Skipping Top-Hat filter.")
        return activity_map
    print(f"Step 4: Applying White Top-Hat filter (kernel size: {kernel_size})...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    map_max = np.max(activity_map)
    if map_max == 0: return activity_map
    map_8bit = (activity_map / map_max * 255).astype(np.uint8)
    tophat_map_8bit = cv2.morphologyEx(map_8bit, cv2.MORPH_TOPHAT, kernel)
    return tophat_map_8bit.astype(np.float64) / 255.0 * map_max

# old roi_mask_code
# def create_panel_roi_mask(median_frame, erosion_iterations):
#     """Creates a binary mask for the panel to ignore edges."""
#     print("Step 5: Creating Panel ROI Mask...")
#     frame_norm = cv2.normalize(median_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     _, thresh = cv2.threshold(frame_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
#     if num_labels <= 1: return np.ones_like(median_frame, dtype=np.uint8)
#     largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#     panel_mask = np.zeros_like(thresh); panel_mask[labels == largest_label] = 1
#     if erosion_iterations > 0:
#         kernel = np.ones((3, 3), np.uint8)
#         panel_mask = cv2.erode(panel_mask, kernel, iterations=erosion_iterations)
#     return panel_mask

def create_panel_roi_mask(median_frame, erosion_iterations=2):
    """
    Creates a more robust binary mask for the panel by specifically finding the
    largest dark object in the frame.
    """
    print("  - Step 5: Creating Panel ROI Mask via Otsu's Thresholding...")
    # 1. Normalize to 8-bit for OpenCV functions
    frame_norm = cv2.normalize(median_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 2. Apply Otsu's thresholding. The panel is dark, so we want to keep the dark parts.
    # THRESH_BINARY_INV makes dark areas white (255) and bright areas black (0).
    _, thresh = cv2.threshold(frame_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Find the largest connected component (which should be the panel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    
    if num_labels <= 1:
        print("    Warning: Could not find any distinct objects for ROI mask. Using full frame.", file=sys.stderr)
        return np.ones_like(median_frame, dtype=np.uint8) # Fallback to no mask

    # The first label (0) is the background. Find the largest component among the others.
    # This correctly assumes the panel is the largest contiguous dark region.
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create a mask that only contains the largest component
    panel_mask = np.zeros_like(thresh)
    panel_mask[labels == largest_label] = 1
    
    # 4. Erode the mask slightly to pull its edges inward
    if erosion_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        panel_mask = cv2.erode(panel_mask, kernel, iterations=erosion_iterations)
        
    return panel_mask

def find_prompts_by_peak_activity(activity_map, num_prompts, quantile_thresh):
    """Finds prompts by selecting the N blobs with the highest peak activity."""
    print(f"\nStep 6: Finding the {num_prompts} most intense blobs (quantile: {quantile_thresh:.3f})...")
    if activity_map is None or not np.any(activity_map > 1e-9): return []
    active_pixels = activity_map[activity_map > 1e-9]
    if active_pixels.size == 0: return []
    activity_threshold = np.quantile(active_pixels, quantile_thresh)
    binary_mask = (activity_map >= activity_threshold).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1: return []
    all_candidates = []
    for i in range(1, num_labels):
        component_mask = (labels == i)
        peak_activity_value = np.max(activity_map[component_mask])
        all_candidates.append({'centroid': centroids[i], 'stat': stats[i], 'peak_activity': peak_activity_value})
    sorted_by_intensity = sorted(all_candidates, key=lambda x: x['peak_activity'], reverse=True)
    top_candidates = sorted_by_intensity[:num_prompts]
    return top_candidates, all_candidates

def visualize_prompts(activity_map, all_candidates, final_prompts, save_path, method):
    # This visualization function remains the same
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(activity_map, cmap='hot', vmin=0)
    label = 'Activity (Theil-Sen Slope)' if method == 'theil_sen' else 'Activity (Kendall Tau Corr.)'
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title('Debug Map: Prompts from Most Intense Blobs')
    if all_candidates: ax.scatter([c['centroid'][0] for c in all_candidates], [c['centroid'][1] for c in all_candidates], s=100, facecolors='none', edgecolors='cyan', lw=1.5, label='All Found Blobs')
    if final_prompts: ax.scatter(np.array([p['centroid'] for p in final_prompts])[:, 0], np.array([p['centroid'] for p in final_prompts])[:, 1], s=600, c='yellow', marker='*', edgecolor='black', label='Final Selected Prompts', zorder=5)
    ax.legend(); fig.savefig(save_path); plt.close(fig)

def run_sam_with_box_prompts(frame_rgb, top_candidates, predictor):
    # This function remains the same
    all_final_masks = []; predictor.set_image(frame_rgb)
    for cand in top_candidates:
        stat = cand['stat']
        x1, y1 = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
        x2, y2 = x1 + stat[cv2.CC_STAT_WIDTH], y1 + stat[cv2.CC_STAT_HEIGHT]
        input_box = np.array([x1, y1, x2, y2])
        masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False)
        if len(masks) > 0: all_final_masks.append(masks[0])
    return all_final_masks

# ========================================================================================
# --- MAIN EXECUTION SCRIPT ---
# ========================================================================================
def main(args):
    """Main execution: Pre-process -> Activity Map -> ROI -> Post-process -> Find Prompts -> SAM."""
    print("--- Modular Hotspot Segmentation (Debug Mode) ---")
    os.makedirs(args.output_dir, exist_ok=True)
    save_parameters(args, args.output_dir)

    print("\nLoading SAM model...")
    try:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"; sam.to(device=device)
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"Error loading SAM model: {e}", file=sys.stderr); sys.exit(1)

    try:
        mat_data = scipy.io.loadmat(args.input)
        frames = mat_data[config.MAT_FRAMES_KEY].astype(np.float64)
        H, W, T = frames.shape
    except Exception as e:
        print(f"  Error loading {args.input}: {e}", file=sys.stderr); sys.exit(1)
    
    # --- Execute Corrected Pipeline Step-by-Step ---
    spatially_filtered_frames = apply_spatial_filter(frames, args.spatial_filter)
    temporally_smoothed_frames = apply_temporal_smoothing(spatially_filtered_frames, args.temporal_smooth_window)
    activity_map = generate_activity_map(temporally_smoothed_frames, args.activity_method, args.env_para)
    
    # Apply Top-Hat filter to the activity map
    post_processed_map = apply_tophat_filter(activity_map, args.tophat_filter_size)
    
    target_frame = np.median(frames, axis=2)
    roi_mask = create_panel_roi_mask(target_frame, args.roi_erosion)
    
    # Use the post-processed and ROI-masked map for finding prompts
    masked_activity_map = post_processed_map * roi_mask
    
    top_candidates, all_candidates = find_prompts_by_peak_activity(
        masked_activity_map, args.num_leaks, args.activity_quantile
    )

    if not top_candidates:
        print("Could not find any significant prompts after all filtering. Exiting.", file=sys.stderr); sys.exit(1)

    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    debug_plot_path = os.path.join(args.output_dir, f"{base_filename}_prompt_verification.png")
    # Visualize the map that was used for selection (the final analysis map)
    visualize_prompts(masked_activity_map, all_candidates, top_candidates, debug_plot_path, args.activity_method)

    print("\nRunning SAM segmentation...")
    frame_normalized_8bit = cv2.normalize(target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    frame_rgb = cv2.cvtColor(frame_normalized_8bit, cv2.COLOR_GRAY2BGR)
    all_final_masks = run_sam_with_box_prompts(frame_rgb, top_candidates, predictor)

    if not all_final_masks: print("SAM did not generate masks. Exiting.", file=sys.stderr); sys.exit(1)

    print("\nSaving final outputs...")
    # (Saving logic is unchanged)
    final_combined_mask = np.zeros((H, W), dtype=bool)
    segmentation_image = frame_rgb.copy()
    colors = [[255, 255, 0], [0, 255, 255]]
    for i, mask in enumerate(all_final_masks):
        color = colors[i % len(colors)]
        color_overlay = np.zeros_like(segmentation_image); color_overlay[mask] = color
        segmentation_image = cv2.addWeighted(segmentation_image, 1, color_overlay, 0.6, 0)
        final_combined_mask = np.logical_or(final_combined_mask, mask)
    plot_save_path = os.path.join(args.output_dir, f"{base_filename}_sam_segmentation.png")
    cv2.imwrite(plot_save_path, segmentation_image)
    mask_save_path = os.path.join(args.output_dir, f"{base_filename}_sam_mask.npy")
    np.save(mask_save_path, final_combined_mask)
    print(f"  Saved outputs to {args.output_dir}")
    print("\n--- Modular debug script finished. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular script for debugging hotspot segmentation on a SINGLE file.")
    
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--output_dir", required=True, type=str)
    
    param_group = parser.add_argument_group('Pipeline Control Parameters')
    param_group.add_argument("--num_leaks", type=int, default=2)
    param_group.add_argument("--env_para", type=int, default=0, choices=[1, -1, 0])
    param_group.add_argument("--activity_method", type=str, default="theil_sen", choices=['theil_sen', 'kendall_tau'])
    # --- Correctly re-added spatial_filter ---
    param_group.add_argument("--spatial_filter", type=str, default="none", choices=['none', 'bilateral'])
    param_group.add_argument("--temporal_smooth_window", type=int, default=3)
    param_group.add_argument("--tophat_filter_size", type=int, default=0, help="Kernel size for White Top-Hat filter on activity map. 0 to disable.")
    param_group.add_argument("--activity_quantile", type=float, default=0.995)
    param_group.add_argument("--roi_erosion", type=int, default=3)

    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found at '{args.input}'", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"Error: SAM checkpoint not found at '{args.checkpoint}'", file=sys.stderr); sys.exit(1)
        
    main(args)