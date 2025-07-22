# universal_SAM_debug.py
"""
UNIVERSAL & FLEXIBLE DEBUG SCRIPT (v4 - Shape Filtering)
- Core Logic: ... -> ROI MASKING -> **SHAPE FILTERING** -> Quantile Threshold.
- SELECTION LOGIC: Finds all active blobs, filters them by area and circularity to
  remove edge artifacts, then selects the N blobs with the highest peak activity.
- **NEW**: Intelligently distinguishes between compact "leak-like" blobs and
  long, thin "edge-like" artifacts.
"""
"""
vid-4:
python src/debug-files/debug_SAM_blob.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-44-18_20_30_10_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple_blob/vid-4/iter-2-Edgesuppresed \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --canny_low 50 \
  --canny_high 150 \
  --suppress_dilate_size 7

vid-5:
python src/debug-files/debug_SAM_blob.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-59-16_20_26_6_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple_blob/vid-5/iter-1-Edgesuppresed \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --canny_low 50 \
  --canny_high 150 \
  --suppress_dilate_size 7

vid-6:
python src/debug-files/debug_SAM_blob.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-17-0-30_20_26_6_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug_simple_blob/vid-6/iter-2-ROI-SHAPE \
  --temporal_smooth_window 3 \
  --roi_erosion 3 \
  --canny_low 50 \
  --canny_high 150 \
  --suppress_dilate_size 7


"""

import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import mstats

try:
    import config
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print(f"Error: segment_anything library not found.", file=sys.stderr)
        sys.exit(1)

    class MockConfig:
        MAT_FRAMES_KEY = 'TempFrames'
    config = MockConfig()


def apply_temporal_smoothing(frames, window_size):
    if window_size <= 1:
        return frames
    print(f"Step 1: Applying temporal smoothing (window: {window_size})...")
    H, W, T = frames.shape
    smoothed_frames = np.zeros_like(frames, dtype=np.float64)
    kernel = np.ones(window_size) / window_size
    for r in tqdm(range(H), desc="  Smoothing Frames", leave=False, ncols=100):
        for c in range(W):
            pixel_series = frames[r, c, :]
            smoothed_series = np.convolve(pixel_series, kernel, mode='valid')
            pad_before = (T - len(smoothed_series)) // 2
            pad_after = T - len(smoothed_series) - pad_before
            smoothed_frames[r, c, :] = np.pad(
                smoothed_series, (pad_before, pad_after), mode='edge')
    return smoothed_frames


def _calculate_slope_for_row(row_data, t):
    W = row_data.shape[0]
    row_slopes = np.zeros(W, dtype=np.float64)
    for c in range(W):
        pixel_series = row_data[c, :]
        if not np.any(np.isnan(pixel_series)) and len(pixel_series) > 1:
            try:
                slope, _, _, _ = mstats.theilslopes(pixel_series, t, 0.95)
            except (ValueError, IndexError):
                slope = 0.0
            row_slopes[c] = slope
    return row_slopes


def generate_activity_map(frames, env_para):
    H, W, T = frames.shape
    t = np.arange(T)
    if T < 2:
        return np.zeros((H, W), dtype=np.float64)
    print("\nStep 2: Calculating Theil-Sen slope map in parallel...")
    results = Parallel(n_jobs=-1)(delayed(_calculate_slope_for_row)(
        frames[r, :, :], t) for r in tqdm(range(H), desc="  Processing Rows", leave=False, ncols=100))
    slope_map = np.vstack(results)
    if env_para == 1:
        print("  - Mode: Heating Only")
        activity_map = slope_map.copy()
        activity_map[activity_map < 0] = 0
    elif env_para == -1:
        print("  - Mode: Cooling Only")
        activity_map = -slope_map
        activity_map[activity_map < 0] = 0
    else:
        print("  - Mode: Absolute Value")
        activity_map = np.abs(slope_map)
    return activity_map


def create_panel_roi_mask(median_frame, erosion_iterations=2):
    print("  - Creating Panel ROI Mask via Otsu's Thresholding...")
    frame_norm = cv2.normalize(
        median_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, thresh = cv2.threshold(
        frame_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thresh, 4, cv2.CV_32S)
    if num_labels <= 1:
        print("    Warning: Could not find a distinct object for ROI mask. Using full frame.", file=sys.stderr)
        return np.ones_like(median_frame, dtype=np.uint8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    panel_mask = np.zeros_like(thresh)
    panel_mask[labels == largest_label] = 1
    if erosion_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        panel_mask = cv2.erode(
            panel_mask, kernel, iterations=erosion_iterations)
    print("    ROI Mask created successfully.")
    return panel_mask


def create_artifact_suppression_mask(median_frame, roi_mask, canny_low, canny_high, dilation_size):
    """
    Identifies major structural edges on the panel and creates a mask to suppress them.
    This helps distinguish unpredictable leaks from predictable edge artifacts.
    """
    print("  - Creating Artifact Suppression Mask to ignore structural edges...")

    # 1. Normalize to 8-bit for OpenCV functions
    frame_norm = cv2.normalize(
        median_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 2. Apply a slight blur to reduce noise before edge detection
    blurred_frame = cv2.GaussianBlur(frame_norm, (5, 5), 0)

    # 3. Use Canny edge detection to find strong structural lines
    edges = cv2.Canny(blurred_frame, canny_low, canny_high)

    # 4. Ensure we only consider edges *inside* the panel ROI
    edges_on_panel = edges * roi_mask

    # 5. Thicken the detected edges to create a "suppression zone"
    if dilation_size > 0:
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        suppression_mask = cv2.dilate(edges_on_panel, kernel, iterations=1)
    else:
        suppression_mask = edges_on_panel

    print("    Artifact Suppression Mask created successfully.")
    return suppression_mask

# --- We revert to the simpler prompt finding function (no shape filters) ---


def find_prompts_by_peak_activity(activity_map, num_prompts, quantile_thresh):
    """
    Finds prompts by selecting the N blobs with the highest peak activity value inside them.
    (This is the version from step 2, without shape filters).
    """
    print(
        f"\nStep 3: Finding the {num_prompts} most intense active blobs (quantile: {quantile_thresh:.3f})...")
    # ... (This function is identical to the one in the v2/v3 code, so I'll omit its body for brevity) ...
    # ... (Make sure to use the version WITHOUT min_area, max_area, min_circularity) ...
    if activity_map is None or not np.any(activity_map > 1e-9):
        return [], []
    active_pixels = activity_map[activity_map > 1e-9]
    if active_pixels.size == 0:
        return [], []
    activity_threshold = np.quantile(active_pixels, quantile_thresh)
    binary_mask = (activity_map >= activity_threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8)
    if num_labels <= 1:
        return [], []
    all_candidates = []
    for i in range(1, num_labels):
        component_mask = (labels == i)
        peak_activity_value = np.max(activity_map[component_mask])
        all_candidates.append(
            {'centroid': centroids[i], 'stat': stats[i], 'peak_activity': peak_activity_value})
    sorted_by_intensity = sorted(
        all_candidates, key=lambda x: x['peak_activity'], reverse=True)
    top_candidates = sorted_by_intensity[:num_prompts]
    return top_candidates, all_candidates


# --- (No changes to visualize_prompts or run_sam_with_box_prompts) ---
def visualize_prompts(activity_map, all_candidates, final_prompts, save_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(activity_map, cmap='hot', vmin=0)
    fig.colorbar(im, ax=ax, label='Activity (Magnitude of Slope)')
    ax.set_title('Debug Map: Prompts from Most Intense Blobs (on Masked Area)')
    if all_candidates:
        cand_x = [c['centroid'][0] for c in all_candidates]
        cand_y = [c['centroid'][1] for c in all_candidates]
        ax.scatter(cand_x, cand_y, s=100, facecolors='none',
                   edgecolors='cyan', lw=1.5, label='All Found Blobs (Post-Filter)')
    if final_prompts:
        prompt_coords = np.array([p['centroid'] for p in final_prompts])
        ax.scatter(prompt_coords[:, 0], prompt_coords[:, 1], s=600, c='yellow', marker='*',
                   edgecolor='black', label='Final Selected Prompts (Most Intense)', zorder=5)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)


def run_sam_with_box_prompts(frame_rgb, top_candidates, predictor):
    all_final_masks = []
    predictor.set_image(frame_rgb)
    for cand in top_candidates:
        stat = cand['stat']
        x1, y1 = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
        x2, y2 = x1 + stat[cv2.CC_STAT_WIDTH], y1 + stat[cv2.CC_STAT_HEIGHT]
        input_box = np.array([x1, y1, x2, y2])
        masks, _, _ = predictor.predict(
            point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False)
        if len(masks) > 0:
            all_final_masks.append(masks[0])
    return all_final_masks


def main(args):
    # --- (No changes to initial setup) ---
    print("--- Universal Hotspot Segmentation (Debug Mode) ---")
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading SAM model...")
    try:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device=device)
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"Error loading SAM model: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        mat_data = scipy.io.loadmat(args.input)
        frames = mat_data[config.MAT_FRAMES_KEY].astype(np.float64)
        H, W, T = frames.shape
    except Exception as e:
        print(f"  Error loading {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    smoothed_frames = apply_temporal_smoothing(
        frames, args.temporal_smooth_window)
    activity_map = generate_activity_map(smoothed_frames, args.env_para)

    print("\nStep 2.5: Creating and applying a spatial ROI mask to ignore edges...")
    target_frame = np.median(frames, axis=2)
    roi_mask = create_panel_roi_mask(
        target_frame, erosion_iterations=args.roi_erosion)
    masked_activity_map = activity_map * roi_mask

    # Step 2.6: Identify and suppress predictable structural artifacts
    suppression_mask = create_artifact_suppression_mask(
        target_frame, 
        roi_mask, 
        args.canny_low, 
        args.canny_high, 
        args.suppress_dilate_size
    )

    final_activity_map = activity_map * roi_mask * (1 - suppression_mask / 255.0)

    # --- MODIFIED CALL with new shape filter arguments ---
    top_candidates, all_candidates = find_prompts_by_peak_activity(
        final_activity_map, 
        num_prompts=args.num_leaks, 
        quantile_thresh=args.activity_quantile
    )

    if not top_candidates:
        print("Could not find any suitable prompts after shape filtering. Exiting.", file=sys.stderr)
        sys.exit(1)

    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    debug_plot_path = os.path.join(
        args.output_dir, f"{base_filename}_prompt_verification.png")
    visualize_prompts(final_activity_map, all_candidates, top_candidates, debug_plot_path)

    # --- (No changes to SAM prediction or saving) ---
    print("\nStep 4: Running SAM segmentation...")
    frame_normalized_8bit = cv2.normalize(
        target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    frame_rgb = cv2.cvtColor(frame_normalized_8bit, cv2.COLOR_GRAY2BGR)
    all_final_masks = run_sam_with_box_prompts(
        frame_rgb, top_candidates, predictor)
    if not all_final_masks:
        print("SAM did not generate masks. Exiting.", file=sys.stderr)
        sys.exit(1)
    print("\nStep 5: Saving final outputs...")
    final_combined_mask = np.zeros((H, W), dtype=bool)
    segmentation_image = frame_rgb.copy()
    # Yellow, Cyan, Magenta in BGR
    colors = [[255, 255, 0], [0, 255, 255], [255, 0, 255]]
    for i, mask in enumerate(all_final_masks):
        color = colors[i % len(colors)]
        color_overlay = np.zeros_like(segmentation_image)
        color_overlay[mask] = color
        segmentation_image = cv2.addWeighted(
            segmentation_image, 1, color_overlay, 0.6, 0)
        final_combined_mask = np.logical_or(final_combined_mask, mask)
    plot_save_path = os.path.join(
        args.output_dir, f"{base_filename}_sam_segmentation.png")
    cv2.imwrite(plot_save_path, segmentation_image)
    mask_save_path = os.path.join(
        args.output_dir, f"{base_filename}_sam_mask.npy")
    np.save(mask_save_path, final_combined_mask)
    print(f"  Saved outputs to {args.output_dir}")
    print("\n--- Universal debug script finished. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Universal script for debugging hotspot segmentation on a SINGLE file.")

    parser.add_argument("--input", required=True, type=str,
                        help="Path to the input .mat video file.")
    parser.add_argument("--checkpoint", required=True,
                        type=str, help="Path to the SAM model checkpoint.")
    parser.add_argument("--model_type", type=str, default="vit_b",
                        help="Type of SAM model (e.g., 'vit_b', 'vit_l').")
    parser.add_argument("--output_dir", required=True,
                        type=str, help="Directory to save all output files.")

    param_group = parser.add_argument_group('Pipeline Control Parameters')
    param_group.add_argument("--num_leaks", type=int, default=2,
                             help="Number of top leak candidates to find.")
    param_group.add_argument("--env_para", type=int, default=0, choices=[
                             1, -1, 0], help="Activity mode: 1=Heating, -1=Cooling, 0=Absolute.")
    param_group.add_argument("--temporal_smooth_window", type=int, default=3,
                             help="Size of the moving average window for temporal smoothing.")
    param_group.add_argument("--activity_quantile", type=float, default=0.99,
                             help="Quantile to determine the threshold for active pixels.")
    param_group.add_argument("--roi_erosion", type=int, default=3,
                             help="Number of erosion iterations for the ROI mask to avoid panel edges.")

    suppress_group = parser.add_argument_group('Artifact Suppression Parameters (Generalizable)')
    suppress_group.add_argument("--canny_low", type=int, default=50, help="Lower threshold for Canny edge detection.")
    suppress_group.add_argument("--canny_high", type=int, default=150, help="Higher threshold for Canny edge detection.")
    suppress_group.add_argument("--suppress_dilate_size", type=int, default=5, help="Size of the dilation kernel to thicken suppressed edges.")
    
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(
            f"Error: Input file not found at '{args.input}'", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(
            f"Error: SAM checkpoint not found at '{args.checkpoint}'", file=sys.stderr)
        sys.exit(1)

    main(args)
