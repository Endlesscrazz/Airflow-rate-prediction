# src/run_sam_segmentation.py
"""
vid-1:
python src/debug-files/run_SAM_debug.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-33-25_20_34_14_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/run-sam-debug/vid-1/iter-2 \
  --temporal_smooth_window 3

vid-2:
python src/debug-files/run_SAM_debug.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-34-30_20_34_14_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/run-sam-debug/vid-2/iter-1-ts-focus10 \
  --temporal_smooth_window 3

vid-6:
python src/debug-files/run_SAM_debug.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-17-0-30_20_26_6_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/run-sam-debug/vid-6/iter-1-ts-focus10 \
  --temporal_smooth_window 3

vid-4:
python src/debug-files/run_SAM_debug.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-44-18_20_30_10_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/run-sam-debug/vid-4/iter-1-HC \
  --temporal_smooth_window 3

vid-8:
python src/debug-files/run_SAM_debug.py \
  --input datasets/dataset_two_holes/T2.0V_6.3Pa_2025-6-16-17-16-39_20_34_14_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/run-sam-debug/vid-8/iter-3-ts-focus10 \
  --temporal_smooth_window 3

vid-12
  python src/debug-files/run_SAM_debug.py \
  --input datasets/dataset_two_holes/T2.0V_6.3Pa_2025-6-16-17-41-25_20_26_6_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/run-sam-debug/vid-12/iter-1- \
  --temporal_smooth_window 3
  
"""

"""
UNIVERSAL ROBUST DEBUG SCRIPT
- Handles BOTH heating and cooling leaks by using the absolute value of the slope.
- This is the definitive version based on the discovery that the dataset contains mixed event types.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.stats import linregress, mstats
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

# --- Core Data Processing Functions ---
def apply_temporal_smoothing(frames, window_size):
    if window_size <= 1: return frames
    print(f"  - Applying temporal smoothing (window: {window_size})...")
    H, W, T = frames.shape
    smoothed_frames = np.zeros_like(frames, dtype=np.float64)
    kernel = np.ones(window_size) / window_size
    for r in tqdm(range(H), desc="  Smoothing Frames", leave=False, ncols=100):
        for c in range(W):
            pixel_series = frames[r, c, :]
            smoothed_series = np.convolve(pixel_series, kernel, mode='valid')
            pad_before = (T - len(smoothed_series)) // 2
            pad_after = T - len(smoothed_series) - pad_before
            smoothed_frames[r, c, :] = np.pad(smoothed_series, (pad_before, pad_after), mode='edge')
    return smoothed_frames

def _calculate_slope_for_row(row_data, t):
    W = row_data.shape[0]
    row_slopes = np.zeros(W, dtype=np.float64)
    for c in range(W):
        pixel_series = row_data[c, :]
        if not np.any(np.isnan(pixel_series)) and len(pixel_series) > 1:
            try: slope, _, _, _ = mstats.theilslopes(pixel_series, t, 0.95)
            except (ValueError, IndexError): slope = 0.0
            row_slopes[c] = slope
    return row_slopes

def calculate_universal_activity_map(frames):
    """Calculates Theil-Sen slope and uses its ABSOLUTE value for the activity map."""
    H, W, T = frames.shape
    if T < 2: return np.zeros((H, W), dtype=np.float64)
    t = np.arange(T)
    print(f"  - Calculating Theil-Sen slope map in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_slope_for_row)(frames[r, :, :], t) for r in tqdm(range(H), desc="  Processing Rows", leave=False, ncols=100)
    )
    slope_map = np.vstack(results)
    
    # --- KEY CHANGE: Use the absolute value to detect both heating and cooling ---
    return np.abs(slope_map)

# ... (The rest of the functions: find_prompts_with_tunable_score, visualize_hybrid_prompts,
#      run_sam_with_box_prompts, and main can remain almost identical, as they operate
#      on the activity_map, which will now correctly represent all thermal events) ...

def find_prompts_with_tunable_score(activity_map, num_prompts, quantile_thresh, min_area, aspect_ratio_limit, border_margin, area_weight):
    """Finds prompts using a tunable hybrid score: Peak Activity * (Area ^ area_weight)."""
    if activity_map is None or not np.any(activity_map > 1e-9): return [], []
    active_pixels = activity_map[activity_map > 1e-9]
    if active_pixels.size == 0: return [], []
    activity_threshold = np.quantile(active_pixels, quantile_thresh)
    binary_mask = (activity_map >= activity_threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1: return [], []
    H, W = activity_map.shape
    valid_candidates = []; all_candidates_for_viz = []
    for i in range(1, num_labels):
        stat = stats[i]
        area = stat[cv2.CC_STAT_AREA]
        w, h = stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_HEIGHT]
        left, top = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
        right, bottom = left + w, top + h
        component_mask = (labels == i)
        peak_activity = np.max(activity_map[component_mask])
        candidate_info = {'centroid': centroids[i], 'stat': stat, 'area': area, 'peak_activity': peak_activity, 'is_valid': True, 'score': peak_activity * (area ** area_weight)}
        all_candidates_for_viz.append(candidate_info)
        if area < min_area: candidate_info['is_valid'] = False; continue
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
        if aspect > aspect_ratio_limit: candidate_info['is_valid'] = False; continue
        if (left < border_margin or right > (W - border_margin) or top < border_margin or bottom > (H - border_margin)): candidate_info['is_valid'] = False; continue
        valid_candidates.append(candidate_info)
    sorted_by_score = sorted(valid_candidates, key=lambda x: x['score'], reverse=True)
    top_candidates = sorted_by_score[:num_prompts]
    return top_candidates, all_candidates_for_viz

def visualize_hybrid_prompts(activity_map, all_candidates, final_prompts, save_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(activity_map, cmap='hot')
    fig.colorbar(ax.images[0], ax=ax, label='Activity (Absolute Slope)') # Label updated
    ax.set_title('Hybrid Score Debug Map (Intensity * Area^weight)')
    if all_candidates:
        for cand in all_candidates:
            color = 'cyan' if cand['is_valid'] else 'red'
            ax.scatter(cand['centroid'][0], cand['centroid'][1], s=100, facecolors='none', edgecolors=color, lw=1.0)
            ax.text(cand['centroid'][0] + 6, cand['centroid'][1] + 6, f"{cand.get('score', 0):.4f}", color=color, fontsize=8, ha='left', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
    if final_prompts:
        prompt_coords = np.array([p['centroid'] for p in final_prompts])
        ax.scatter(prompt_coords[:, 0], prompt_coords[:, 1], s=600, c='yellow', marker='*', edgecolor='black', zorder=5)
    legend_elements = [ plt.Line2D([0], [0], marker='o', color='w', label='Valid Candidates', markerfacecolor='none', markeredgecolor='cyan'), plt.Line2D([0], [0], marker='o', color='w', label='Filtered Out', markerfacecolor='none', markeredgecolor='red'), plt.Line2D([0], [0], marker='*', color='w', label='Final Prompts', markerfacecolor='yellow', markeredgecolor='black', markersize=15)]
    ax.legend(handles=legend_elements)
    fig.savefig(save_path)
    plt.close(fig)

def run_sam_with_box_prompts(frame_rgb, top_candidates, predictor):
    all_final_masks = []; predictor.set_image(frame_rgb)
    for cand in top_candidates:
        stat = cand['stat']
        x1, y1 = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
        x2, y2 = x1 + stat[cv2.CC_STAT_WIDTH], y1 + stat[cv2.CC_STAT_HEIGHT]
        input_box = np.array([x1, y1, x2, y2])
        masks, _, _ = predictor.predict( point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False)
        if len(masks) > 0: all_final_masks.append(masks[0])
    return all_final_masks

def main(args):
    """Main execution function for the universal debug pipeline."""
    print("--- Universal Hotspot Segmentation (Debug Mode) ---")
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading SAM model...")
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

    print("\nStep 1: Pre-processing...")
    smoothed_frames = apply_temporal_smoothing(frames, args.temporal_smooth_window)
    print("\nStep 2: Generating Universal Activity Map...")
    activity_map = calculate_universal_activity_map(smoothed_frames)
    print("\nStep 3: Finding Prompts...")
    top_candidates, all_candidates_for_viz = find_prompts_with_tunable_score(
        activity_map, num_prompts=args.num_leaks, quantile_thresh=args.activity_quantile,
        min_area=args.min_area, aspect_ratio_limit=args.aspect_ratio_limit,
        border_margin=args.border_margin, area_weight=args.area_weight
    )
    if not top_candidates: print("Could not find any valid prompts. Exiting.", file=sys.stderr); sys.exit(1)
    print(f"  Selected {len(top_candidates)} candidates with the highest hybrid scores.")
    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    debug_plot_path = os.path.join(args.output_dir, f"{base_filename}_prompt_verification.png")
    visualize_hybrid_prompts(activity_map, all_candidates_for_viz, top_candidates, debug_plot_path)

    print("\nStep 4: Running SAM segmentation...")
    target_frame = np.median(frames, axis=2)
    frame_normalized_8bit = cv2.normalize(target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    frame_rgb = cv2.cvtColor(frame_normalized_8bit, cv2.COLOR_GRAY2BGR)
    all_final_masks = run_sam_with_box_prompts(frame_rgb, top_candidates, predictor)
    if not all_final_masks: print("SAM did not generate masks. Exiting.", file=sys.stderr); sys.exit(1)

    print("\nStep 5: Saving final outputs...")
    final_combined_mask = np.zeros((H, W), dtype=bool)
    plt.figure(figsize=(12, 9)); plt.imshow(frame_rgb); plt.axis('off')
    colors = [[1, 1, 0], [0, 1, 1]]
    for i, mask in enumerate(all_final_masks):
        color = colors[i % len(colors)]
        mask_image = mask.reshape(H, W, 1) * np.array([*color, 0.6]).reshape(1, 1, -1)
        plt.imshow(mask_image)
    prompt_coords = np.array([p['centroid'] for p in top_candidates])
    plt.scatter(prompt_coords[:, 0], prompt_coords[:, 1], c='lime', marker='*', s=250, edgecolor='black')
    plt.title(f"Universal Segmentation for {base_filename}")
    plot_save_path = os.path.join(args.output_dir, f"{base_filename}_sam_segmentation.png")
    plt.savefig(plot_save_path); plt.close()
    mask_save_path = os.path.join(args.output_dir, f"{base_filename}_sam_mask.npy")
    np.save(mask_save_path, final_combined_mask)
    print(f"  Saved outputs to {args.output_dir}")
    print("\n--- Universal debug script finished. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal robust script for debugging hotspot segmentation on a SINGLE file.")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--output_dir", required=True, type=str)
    param_group = parser.add_argument_group('Pipeline Control Parameters')
    param_group.add_argument("--num_leaks", type=int, default=2)
    param_group.add_argument("--temporal_smooth_window", type=int, default=5)
    param_group.add_argument("--activity_quantile", type=float, default=0.98)
    param_group.add_argument("--min_area", type=int, default=5)
    param_group.add_argument("--aspect_ratio_limit", type=float, default=4.0)
    param_group.add_argument("--border_margin", type=int, default=20)
    param_group.add_argument("--area_weight", type=float, default=0.5)
    args = parser.parse_args()
    if not os.path.isfile(args.input): print(f"Error: Input file not found at '{args.input}'", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(args.checkpoint): print(f"Error: SAM checkpoint not found at '{args.checkpoint}'", file=sys.stderr); sys.exit(1)
    main(args)