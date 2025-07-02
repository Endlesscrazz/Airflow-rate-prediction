#!/usr/bin/env python3
"""
debug_SAM.py

Standalone debug script to visualize each step of the robust, multi-stage
hotspot mask generation pipeline for a SINGLE thermal .mat file.
"""
import os
import argparse
import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.spatial import distance
import torch
from segment_anything import sam_model_registry, SamPredictor

# --- Core Functions ---

def calculate_simple_slope_map(frames):
    H, W, T = frames.shape
    if T < 2: return np.zeros((H, W), dtype=np.float64)
    t = np.arange(T)
    slope_map = np.zeros((H, W), dtype=np.float64)
    print("1. Calculating raw slope map...")
    for r in range(H):
        for c in range(W):
            s = frames[r, c, :]
            if not np.any(np.isnan(s)):
                slope_map[r, c] = linregress(t, s)[0]
    return np.abs(slope_map)

# This is the robust prompt finding function
def find_prompt_points_robust(activity_map, num_points, quantile, min_area, aspect_limit, w_solidity, w_isolation):
    """
    Finds prompts by finding ALL potential blobs, then filtering and scoring them
    based on a combination of their peak activity, solidity, and spatial isolation.
    """
    if activity_map is None or not np.any(activity_map > 0):
        print("Warning: Activity map is empty."); return None, None

    active_pixels = activity_map[activity_map > 1e-9]
    if active_pixels.size == 0:
        print("Warning: No active pixels found in activity map."); return None, None
        
    activity_threshold = np.quantile(active_pixels, quantile)
    binary_mask = (activity_map >= activity_threshold).astype(np.uint8)
    
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)

    if num_labels <= 1:
        print("Warning: No connected components found after thresholding."); return None, None

    candidate_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        if area < min_area: continue
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
        if aspect_ratio > aspect_limit: continue
        
        component_mask = (labels == i)
        contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 0.0
        if contours:
            hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
            if hull_area > 0: solidity = area / hull_area
        
        peak_activity = np.max(activity_map[component_mask])
        
        candidate_components.append({
            'centroid': centroids[i], 'peak_activity': peak_activity,
            'solidity': solidity, 'label_id': i
        })

    if not candidate_components:
        print("Warning: No candidates passed shape filtering."); return None, None

    if len(candidate_components) > 1:
        coords = np.array([c['centroid'] for c in candidate_components])
        dist_matrix = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        min_distances = np.min(dist_matrix, axis=1)
        for i, candidate in enumerate(candidate_components):
            candidate['final_score'] = candidate['peak_activity'] * (1 + w_isolation * np.log1p(min_distances[i])) * (1 + w_solidity * candidate['solidity'])
    else:
        for candidate in candidate_components: candidate['final_score'] = candidate['peak_activity']
    
    sorted_candidates = sorted(candidate_components, key=lambda x: x['final_score'], reverse=True)
    
    print("\n--- Top 5 Candidates by Final Score ---")
    for cand in sorted_candidates[:5]:
        print(f"  ID:{cand.get('label_id', 'N/A')} @({int(cand['centroid'][0])},{int(cand['centroid'][1])}) -> Score: {cand.get('final_score', 0):.4f} (Act: {cand.get('peak_activity', 0):.4f}, Solid: {cand.get('solidity', 0):.2f})")
    
    top_candidates = sorted_candidates[:num_points]
    prompt_points = np.array([c['centroid'] for c in top_candidates]) if top_candidates else None
    return prompt_points, candidate_components

def run_sam_with_prompts(frame_rgb, prompt_points, checkpoint, model_type, device):
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    predictor = SamPredictor(sam); predictor.set_image(frame_rgb)
    all_final_masks = []
    negative_prompt_offset = 15
    for pos_point in prompt_points:
        points = [pos_point]; labels = [1]
        x,y = pos_point
        points.extend([[x,y-negative_prompt_offset],[x,y+negative_prompt_offset],[x-negative_prompt_offset,y],[x+negative_prompt_offset,y]])
        labels.extend([0,0,0,0])
        masks, _, _ = predictor.predict(point_coords=np.array(points), point_labels=np.array(labels), multimask_output=False)
        if len(masks)>0: all_final_masks.append(masks[0])
    return all_final_masks

# ---- Main debug pipeline ----
def main():
    parser = argparse.ArgumentParser(description="Debug hotspot mask generation for a SINGLE .mat file.")
    p = parser # For brevity
    p.add_argument('--input', required=True, help="Path to the specific .mat file.")
    p.add_argument('--checkpoint', required=True, help="Path to the SAM model checkpoint.")
    p.add_argument('--output_dir', default='./debug_outputs', help="Directory to save all debug plots and final mask.")
    p.add_argument('--model_type', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    p.add_argument('--top_k', type=int, default=2, help='Number of leaks to find.')
    
    prompt_group = p.add_argument_group('Prompt Finding Parameters')
    prompt_group.add_argument("--activity_quantile", type=float, default=0.97, help="Quantile to find initial candidate blobs.")
    prompt_group.add_argument("--min_area", type=int, default=3, help="Minimum pixel area for a candidate blob.")
    prompt_group.add_argument("--aspect_ratio_limit", type=float, default=3.5, help="Maximum aspect ratio for a candidate blob.")
    prompt_group.add_argument("--w_solidity", type=float, default=2.0, help="Weight for the solidity score.")
    prompt_group.add_argument("--w_isolation", type=float, default=0.5, help="Weight for the isolation score.")
    args=p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load frames and prepare for visualization
    mat = scipy.io.loadmat(args.input)
    frames = mat['TempFrames'].astype(np.float32)
    bg = np.median(frames, axis=2)
    norm = (bg - bg.min()) / (np.ptp(bg) + 1e-9)
    frame_rgb = cv2.cvtColor((norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # 1) Slope map
    activity_map = calculate_simple_slope_map(frames)
    plt.figure(figsize=(12,9)); plt.imshow(activity_map, cmap='hot'); plt.title('1) Raw Activity Map'); plt.colorbar();
    plt.savefig(os.path.join(args.output_dir, '1_raw_activity_map.png')); plt.close()
    print("1. Saved raw activity map.")
    
    # 2) Find prompt points using the robust method
    print("\n2. Finding and scoring prompt candidates...")
    
    # --- FIX #1: Correctly call the function and pass all arguments ---
    final_prompts, all_candidates = find_prompt_points_robust(
        activity_map, 
        num_points=args.top_k, 
        quantile=args.activity_quantile, 
        min_area=args.min_area, 
        aspect_limit=args.aspect_ratio_limit,
        w_solidity=args.w_solidity, 
        w_isolation=args.w_isolation
    )

    # 3) Visualize candidates AND final selected prompts
    print("3. Visualizing prompt selection process...")
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(activity_map, cmap='hot')
    ax.set_title('2) Candidates (Cyan Circles) & Final Prompts (Lime Star)')
    if all_candidates:
        candidate_coords = np.array([c['centroid'] for c in all_candidates])
        ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1], s=100, facecolors='none', edgecolors='cyan', lw=1.5, label='Candidate Prompts')
    
    # --- FIX #2: Correctly plot final prompts ---
    if final_prompts is not None and final_prompts.size > 0:
        # Centroids from cv2 are (x,y) which matches scatter's expectation
        ax.scatter(final_prompts[:, 0], final_prompts[:, 1], s=600, c='lime', marker='*', edgecolor='black', label='Final Selected Prompts')
    
    ax.legend()
    plt.savefig(os.path.join(args.output_dir, '2_prompt_selection.png')); plt.close()
    print("   Saved prompt selection visualization.")

    if final_prompts is None:
        print("No valid prompts found after filtering. Exiting."); return

    # 4) SAM segmentation
    print("\n4. Running SAM segmentation...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    masks = run_sam_with_prompts(frame_rgb, final_prompts, args.checkpoint, args.model_type, device)

    # 5) Visualize final SAM masks
    print("5. Visualizing final SAM masks...")
    fig, ax = plt.subplots(figsize=(12,9))
    ax.imshow(frame_rgb)
    ax.set_title(f'3) Final SAM Masks ({len(masks)} found)')
    ax.axis('off')

    MASK_COLORS = [np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0])] 

    for i, m in enumerate(masks):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        h, w = m.shape
        mask_image = m.reshape(h, w, 1) * np.array([*color, 0.6]).reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    if final_prompts is not None and final_prompts.size > 0:
        ax.scatter(final_prompts[:,0], final_prompts[:,1], c='lime', marker='*', s=250, edgecolor='black')

    plt.savefig(os.path.join(args.output_dir, f'3_final_segmentation_combined.png')); plt.close()
    print("   Saved final combined segmentation.")
    
    # Save the combined mask as .npy
    final_combined_mask = np.zeros(frame_rgb.shape[:2], dtype=bool)
    for m in masks:
        final_combined_mask = np.logical_or(final_combined_mask, m)
    np.save(os.path.join(args.output_dir, 'final_mask.npy'), final_combined_mask)
    print("   Saved final_mask.npy")

if __name__ == '__main__':
    main()