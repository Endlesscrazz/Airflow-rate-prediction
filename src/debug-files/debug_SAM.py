# src/run_sam_segmentation.py
"""
Script to automatically segment thermal hotspots from a SINGLE IR video file
using a robust, multi-pass filtering approach.

INSTRUCTIONS:
This script now runs on a single file. Use the --input_file argument.
Example:
python run_SAM.py --input_file path/to/your/video.mat --output_dir ./single_run_output
python src/debug-files/debug_SAM.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-44-18_20_30_10_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/debug/vid-4/iter-1

"""

import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.spatial import distance
from tqdm import tqdm
import fnmatch

# --- Ensure project modules are importable ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path: sys.path.append(src_path)
except NameError:
    project_root = os.getcwd(); src_path = os.path.join(project_root, "src")
    if src_path not in sys.path: sys.path.append(src_path)

# --- Import SAM and Config ---
#import config
from segment_anything import sam_model_registry, SamPredictor

def calculate_simple_slope_map(frames):
    """Calculates a linear regression slope for each pixel over the entire video duration."""
    H, W, T = frames.shape
    if T < 2: return np.zeros((H, W), dtype=np.float64)
    t = np.arange(T)
    slope_map = np.zeros((H, W), dtype=np.float64)
    print("Step 1: Calculating raw activity (slope) map...")
    for r in range(H):
        for c in range(W):
            pixel_series = frames[r, c, :]
            if not np.any(np.isnan(pixel_series)):
                try: slope, _, _, _, _ = linregress(t, pixel_series)
                except ValueError: slope = 0.0
                slope_map[r, c] = slope
    return np.abs(slope_map)

def score_all_candidates(activity_map, quantile, w_activity_boost, w_isolation, w_solidity):
    """Finds and scores all potential candidates to establish a stable global ranking."""
    active_pixels = activity_map[activity_map > 1e-9]
    if active_pixels.size == 0: return [], []
    
    activity_threshold = np.quantile(active_pixels, quantile)
    binary_mask = (activity_map >= activity_threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)

    if num_labels <= 1: return [], []
    print(f"  Found {num_labels - 1} initial components using quantile {quantile:.3f}.")

    all_components = []
    for i in range(1, num_labels):
        component_mask = (labels == i)
        peak_activity = np.max(activity_map[component_mask])
        area = stats[i, cv2.CC_STAT_AREA]
        contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 0.0
        if contours:
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            if hull_area > 0: solidity = float(area) / hull_area
        all_components.append({
            'label': i, 'centroid': centroids[i], 'stat': stats[i],
            'raw_activity': peak_activity, 'raw_solidity': solidity, 'area': area,
            'aspect_ratio': (stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT] if stats[i, cv2.CC_STAT_HEIGHT] > 0 else float('inf'))
        })

    if not all_components: return [], []

    if len(all_components) > 1:
        coords = np.array([c['centroid'] for c in all_components])
        dist_matrix = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        min_distances = np.min(dist_matrix, axis=1)
        for i, c in enumerate(all_components): c['raw_isolation'] = min_distances[i]
    else:
        all_components[0]['raw_isolation'] = np.max(activity_map.shape)

    def normalize(arr):
        min_val, max_val = np.min(arr), np.max(arr)
        return (arr - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(arr)

    norm_isolations = normalize(np.array([c['raw_isolation'] for c in all_components]))
    norm_solidities = normalize(np.array([c['raw_solidity'] for c in all_components]))

    for i, c in enumerate(all_components):
        isolation_booster = 1 + (w_isolation * norm_isolations[i])
        solidity_booster = 1 + (w_solidity * norm_solidities[i])
        c['final_score'] = (c['raw_activity'] * w_activity_boost) * isolation_booster * solidity_booster

    return sorted(all_components, key=lambda x: x['final_score'], reverse=True), all_components

def apply_filters_to_candidates(sorted_candidates, img_shape, border_margin, min_area, aspect_limit, min_solidity):
    """Applies a series of hard filters to a pre-scored list of candidates."""
    print("\nStep 3: Applying Post-Scoring Filters...")
    print(f"  (Margin:{border_margin}px, Min Area:{min_area}, Aspect Limit:{aspect_limit:.1f}, Min Solidity:{min_solidity:.2f})")
    
    clean_candidates = []
    for cand in sorted_candidates:
        x, y = cand['centroid']
        if cand['area'] < min_area: continue
        if cand['aspect_ratio'] > aspect_limit and (1/cand['aspect_ratio']) > aspect_limit: continue
        if (x < border_margin or x > (img_shape[1] - border_margin) or y < border_margin or y > (img_shape[0] - border_margin)):
            continue
        if cand['raw_solidity'] < min_solidity:
            continue
        clean_candidates.append(cand)
    return clean_candidates

def visualize_prompts_with_scores(activity_map, all_candidates, final_prompts, save_path):
    """Visualizes candidate and final prompts, annotating candidates with their scores."""
    plt.figure(figsize=(14, 10)); plt.imshow(activity_map, cmap='hot')
    plt.colorbar(label='Activity (abs_slope)'); plt.title('Debug: Activity Map with Scored Candidates and Final Prompts')
    if all_candidates:
        for cand in all_candidates:
            x, y = cand['centroid']; score = cand.get('final_score', 0)
            plt.scatter(x, y, s=100, facecolors='none', edgecolors='cyan', lw=1.0)
            plt.text(x + 6, y + 6, f'{score:.4f}', color='cyan', fontsize=8, ha='left', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
    if final_prompts is not None and final_prompts.size > 0:
        plt.scatter(final_prompts[:, 0], final_prompts[:, 1], s=600, c='lime', marker='*', edgecolor='black', label='Final Selected Prompts', zorder=5)
    plt.scatter([], [], s=100, facecolors='none', edgecolors='cyan', label='All Candidates (Score Ann.)')
    plt.legend(); plt.savefig(save_path); plt.close()
    print(f"  Saved prompt visualization with scores to: {save_path}")

def run_sam_with_prompts(frame_rgb, top_candidates, predictor):
    """Runs SAM using bounding box negative prompts for better constraint."""
    all_final_masks = []
    predictor.set_image(frame_rgb)
    for cand in top_candidates:
        pos_point = cand['centroid']; stat = cand['stat']
        x, y, w, h = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP], stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_HEIGHT]
        offset = 5
        negative_points = [[x - offset, y - offset], [x + w + offset, y - offset], [x + w + offset, y + h + offset], [x - offset, y + h + offset]]
        points = [pos_point]; points.extend(negative_points)
        labels = [1, 0, 0, 0, 0]
        masks, _, _ = predictor.predict(point_coords=np.array(points), point_labels=np.array(labels), multimask_output=False)
        if len(masks) > 0: all_final_masks.append(masks[0])
    return all_final_masks

def main(args):
    """Main execution function to process a single video file."""
    print("--- Starting Automated Hotspot Segmentation for a Single File ---")
    print(f"Input File: {args.input_file}")
    print(f"Output Directory: {args.output_dir}")
    print("-" * 30)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load SAM model once
    print("Loading SAM model...")
    try:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"; sam.to(device=device)
        predictor = SamPredictor(sam)
        print(f"SAM model loaded to {device}.")
    except Exception as e:
        print(f"Error loading SAM model: {e}", file=sys.stderr); sys.exit(1)

    # A. Load video data
    try:
        mat_data = scipy.io.loadmat(args.input_file)
        frames = mat_data["TempFrames"].astype(np.float64)
        H, W, T = frames.shape
        if T < 2: raise ValueError("Not enough frames.")
    except Exception as e:
        print(f"  Error loading {args.input_file}: {e}", file=sys.stderr); sys.exit(1)

    # B. Generate Activity Map
    activity_map = calculate_simple_slope_map(frames)

    # C. Score all candidates
    print("\nStep 2: Scoring all potential candidates...")
    sorted_candidates, all_candidates_for_viz = score_all_candidates(
        activity_map, quantile=args.activity_quantile,
        w_activity_boost=1.0, w_isolation=args.w_isolation, w_solidity=args.w_solidity
    )
    if not sorted_candidates:
        print("No candidates found after scoring. Exiting.", file=sys.stderr); sys.exit(1)

    # D. Apply hard filters to the scored list
    clean_candidates = apply_filters_to_candidates(
        sorted_candidates, img_shape=activity_map.shape,
        border_margin=args.border_margin, min_area=args.min_area,
        aspect_limit=args.aspect_ratio_limit, min_solidity=args.min_solidity
    )

    # E. Select the top N from the clean list
    top_candidates = clean_candidates[:args.num_leaks]
    if not top_candidates:
        print("No candidates survived the filtering process. Exiting.", file=sys.stderr); sys.exit(1)

    final_prompts = np.array([c['centroid'] for c in top_candidates])
    print(f"\nFinal {len(final_prompts)} prompts selected after filtering.")

    # F. Visualize the prompt finding process
    base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    debug_plot_path = os.path.join(args.output_dir, f"{base_filename}_prompt_verification.png")
    visualize_prompts_with_scores(activity_map, all_candidates_for_viz, final_prompts, debug_plot_path)

    # G. Prepare image and run SAM
    print("\nStep 4: Running SAM segmentation...")
    target_frame = np.median(frames, axis=2)
    frame_normalized_8bit = cv2.normalize(target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    frame_rgb = cv2.cvtColor(frame_normalized_8bit, cv2.COLOR_GRAY2BGR)
    
    all_final_masks = run_sam_with_prompts(frame_rgb, top_candidates, predictor)
    
    if not all_final_masks:
        print("SAM did not generate any masks from the prompts. Skipping save.", file=sys.stderr); sys.exit(1)

    # H. Visualize and Save Final Results
    print("\nStep 5: Visualizing and saving final results...")
    final_combined_mask = np.zeros((H, W), dtype=bool)
    plt.figure(figsize=(12, 9)); plt.imshow(frame_rgb)
    colors = [[0, 1, 1], [1, 1, 0], [1, 0, 1]] # Cyan, Yellow, Magenta
    for i, mask in enumerate(all_final_masks):
        color = colors[i % len(colors)]
        mask_image = mask.reshape(H, W, 1) * np.array([*color, 0.6]).reshape(1, 1, -1)
        plt.imshow(mask_image)
        final_combined_mask = np.logical_or(final_combined_mask, mask)
    plt.scatter(final_prompts[:, 0], final_prompts[:, 1], color='lime', marker='*', s=250, edgecolor='black', lw=1.5)
    plt.title(f"Automated Segmentation for {base_filename}"); plt.axis('off')
    plot_save_path = os.path.join(args.output_dir, f"{base_filename}_sam_segmentation.png")
    plt.savefig(plot_save_path); plt.close()
    print(f"  Final visualization saved to: {plot_save_path}")
    
    mask_save_path = os.path.join(args.output_dir, f"{base_filename}_sam_mask.npy")
    np.save(mask_save_path, final_combined_mask)
    print(f"  Final mask saved to: {mask_save_path}")

    print("\n--- Process Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run robust hotspot segmentation on a SINGLE IR video file.")
    
    # --- MODIFIED: Single file input ---
    parser.add_argument("--input_file", required=True, type=str, 
                        help="Path to the specific .mat file to process.")
    parser.add_argument("--output_dir", required=True, type=str, 
                        help="Directory to save all outputs for this run.")
    
    parser.add_argument("--num_leaks", type=int, default=2, 
                        help="The number of leaks to detect.")
    
    # --- ADDED: All tuning parameters ---
    score_group = parser.add_argument_group('Scoring Parameters')
    score_group.add_argument("--activity_quantile", type=float, default=0.98, help="Quantile to find initial candidate blobs.")
    score_group.add_argument("--w_isolation", type=float, default=0.2, help="Weight for the isolation booster.")
    score_group.add_argument("--w_solidity", type=float, default=2.0, help="Weight for the shape solidity booster.")
    
    filter_group = parser.add_argument_group('Post-Score Filtering Parameters')
    filter_group.add_argument("--min_area", type=int, default=5, help="Minimum pixel area for a candidate blob.")
    filter_group.add_argument("--aspect_ratio_limit", type=float, default=4.0, help="Maximum aspect ratio for a candidate blob.")
    filter_group.add_argument("--border_margin", type=int, default=10, help="Pixel margin from the edge to filter candidates.")
    filter_group.add_argument("--min_solidity", type=float, default=0.80, help="Hard filter: minimum solidity required for a candidate to be considered.")

    # --- SAM Parameters ---
    default_checkpoint = os.path.join('SAM', 'sam_checkpoints', 'sam_vit_b_01ec64.pth')
    parser.add_argument("--checkpoint_path", type=str, default=default_checkpoint, help="Path to the SAM model checkpoint file.")
    parser.add_argument("--model_type", type=str, default="vit_b", help="The type of SAM model to use.")

    args = parser.parse_args()
    
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file not found at '{args.input_file}'", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(args.checkpoint_path):
        print(f"Error: SAM checkpoint not found at '{args.checkpoint_path}'", file=sys.stderr); sys.exit(1)
        
    main(args)