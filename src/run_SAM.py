# src/run_sam_segmentation.py
"""
Script to automatically segment thermal hotspots from all IR videos in a dataset
using a hybrid approach:
1. Simple Slope Analysis: Calculates an activity map to find regions of change.
2. Robust Prompt Point Identification: Filters active regions by shape and isolation.
3. SAM Segmentation: Uses the identified points as prompts to generate precise masks.
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
import config
from segment_anything import sam_model_registry, SamPredictor

def calculate_simple_slope_map(frames):
    """Calculates a linear regression slope for each pixel over the entire video duration."""
    H, W, T = frames.shape
    if T < 2: return np.zeros((H, W), dtype=np.float64)
    t = np.arange(T)
    slope_map = np.zeros((H, W), dtype=np.float64)
    for r in range(H): # tqdm can be added here if desired for long processes
        for c in range(W):
            pixel_series = frames[r, c, :]
            if not np.any(np.isnan(pixel_series)):
                slope, _, _, _, _ = linregress(t, pixel_series)
                slope_map[r, c] = slope
    return slope_map

def find_prompt_points_by_shape_and_isolation(activity_map, num_points=2, quantile_thresh=0.98,
                                             min_area=5, aspect_ratio_limit=4.0):
    """Finds prompt points by filtering blobs by shape and then re-scoring by isolation."""
    if activity_map is None or not np.any(activity_map > 0): return None, None
    active_pixels = activity_map[activity_map > 1e-9]
    if active_pixels.size == 0: return None, None
        
    activity_threshold = np.quantile(active_pixels, quantile_thresh)
    binary_mask = (activity_map >= activity_threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)

    if num_labels <= 1: return None, None

    candidate_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_area: continue
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
        if aspect_ratio > aspect_ratio_limit: continue
        
        component_mask = (labels == i)
        peak_activity = np.max(activity_map[component_mask])
        candidate_components.append({'centroid': centroids[i], 'score': peak_activity, 'label': i})
    
    if not candidate_components: return None, None

    if len(candidate_components) > 1:
        coords = np.array([c['centroid'] for c in candidate_components])
        dist_matrix = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        min_distances = np.min(dist_matrix, axis=1)
        for i, candidate in enumerate(candidate_components):
            candidate['score'] = candidate['score'] * np.log1p(min_distances[i])
    
    sorted_candidates = sorted(candidate_components, key=lambda x: x['score'], reverse=True)
    top_candidates = sorted_candidates[:num_points]
    if len(top_candidates) < num_points: print(f"  Warning: Found only {len(top_candidates)} valid candidates.")
    
    prompt_points = np.array([c['centroid'] for c in top_candidates]) if top_candidates else None
    return prompt_points, candidate_components

def visualize_prompts_on_activity_map(activity_map, all_candidates, final_prompts, save_path):
    """Visualizes candidate and final prompts on the activity map."""
    plt.figure(figsize=(12, 9)); plt.imshow(activity_map, cmap='hot'); plt.colorbar(label='Activity (abs_slope)')
    plt.title('Debug: Activity Map with Candidate and Final Prompts')
    if all_candidates:
        candidate_coords = np.array([c['centroid'] for c in all_candidates])
        plt.scatter(candidate_coords[:, 0], candidate_coords[:, 1], s=80, facecolors='none', edgecolors='cyan', lw=1.5, label='Candidate Prompts')
    if final_prompts is not None and final_prompts.size > 0:
        plt.scatter(final_prompts[:, 0], final_prompts[:, 1], s=400, c='lime', marker='*', edgecolor='black', label='Final Selected Prompts')
    plt.legend(); plt.savefig(save_path); plt.close()
    print(f"  Saved prompt-finding debug plot to: {save_path}")

def main(args):
    """Main execution function to process all videos in a dataset."""
    print("--- Starting Automated Hotspot Segmentation Batch Process ---")
    print(f"Dataset Directory: {args.dataset_dir}")
    print(f"Base Output Directory: {args.base_output_dir}")
    print("-" * 30)

    # 1. Load SAM model once to be reused for all videos
    print("Loading SAM model...")
    try:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"; sam.to(device=device)
        predictor = SamPredictor(sam)
        print(f"SAM model loaded to {device}.")
    except Exception as e:
        print(f"Error loading SAM model: {e}"); sys.exit(1)

    # 2. Walk through the dataset directory and find all .mat files
    for root, _, files in os.walk(args.dataset_dir):
        for video_filename in tqdm(fnmatch.filter(files, '*.mat'), desc="Processing Videos"):
            video_path = os.path.join(root, video_filename)
            base_filename = os.path.splitext(video_filename)[0]
            
            # --- DYNAMIC OUTPUT FOLDER CREATION ---
            # Create a subfolder for each video's output
            video_output_dir = os.path.join(args.base_output_dir, base_filename)
            os.makedirs(video_output_dir, exist_ok=True)
            
            print(f"\n--- Processing: {video_filename} ---")
            print(f"--- Outputs will be saved to: {video_output_dir} ---")

            # A. Load video data
            try:
                mat_data = scipy.io.loadmat(video_path)
                frames = mat_data[config.MAT_FRAMES_KEY].astype(np.float64)
                H, W, T = frames.shape
                if T < 2: raise ValueError("Not enough frames.")
            except Exception as e:
                print(f"  Error loading {video_filename}: {e}. Skipping."); continue

            # B. Generate Activity Map
            # print("  Step 1: Generating activity map...") # Can be verbose
            slope_map = calculate_simple_slope_map(frames)
            activity_map = np.abs(slope_map)

            # C. Find Prompt Points
            # print("  Step 2: Finding prompt points...") # Can be verbose
            prompt_points, all_candidates = find_prompt_points_by_shape_and_isolation(
                activity_map, num_points=args.num_leaks,
                min_area=args.min_area, aspect_ratio_limit=args.aspect_ratio_limit,
                quantile_thresh=args.activity_quantile
            )

            # Visualize the prompts for this specific video
            debug_plot_path = os.path.join(video_output_dir, f"{base_filename}_prompt_verification.png")
            visualize_prompts_on_activity_map(activity_map, all_candidates, prompt_points, debug_plot_path)

            if prompt_points is None:
                print("  Could not automatically determine prompt points. Skipping SAM segmentation for this file.")
                continue
            
            print(f"  Found {len(prompt_points)} final prompt points.")

            # D. Prepare image and run SAM
            # print("  Step 3: Running SAM prediction...") # Can be verbose
            target_frame = np.median(frames, axis=2)
            frame_normalized_8bit = cv2.normalize(target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_rgb = cv2.cvtColor(frame_normalized_8bit, cv2.COLOR_GRAY2BGR)
            predictor.set_image(frame_rgb)
            
            all_final_masks = []
            negative_prompt_offset = 15
            for pos_point in prompt_points:
                current_points = [pos_point]; current_labels = [1]
                x, y = pos_point
                negative_points = [[x,y-negative_prompt_offset], [x,y+negative_prompt_offset], [x-negative_prompt_offset,y], [x+negative_prompt_offset,y]]
                current_points.extend(negative_points); current_labels.extend([0,0,0,0])
                masks, scores, _ = predictor.predict(point_coords=np.array(current_points), point_labels=np.array(current_labels), multimask_output=False)
                if len(masks) > 0: all_final_masks.append(masks[0])
            
            if not all_final_masks:
                print("  SAM did not generate any masks from the prompts. Skipping save."); continue

            # E. Visualize and Save Final Results
            final_combined_mask = np.zeros((H, W), dtype=bool)
            plt.figure(figsize=(12, 9)); plt.imshow(frame_rgb)
            for mask in all_final_masks:
                color = np.random.random(3); h, w = mask.shape
                mask_image = mask.reshape(h,w,1) * np.array([*color, 0.5]).reshape(1,1,-1)
                plt.imshow(mask_image); final_combined_mask = np.logical_or(final_combined_mask, mask)
            plt.scatter(prompt_points[:, 0], prompt_points[:, 1], color='lime', marker='*', s=250, edgecolor='black', lw=1.5)
            plt.title(f"Automated Segmentation for {base_filename}"); plt.axis('off')
            plot_save_path = os.path.join(video_output_dir, f"{base_filename}_sam_segmentation.png")
            plt.savefig(plot_save_path); plt.close()
            print(f"  Final visualization saved to: {plot_save_path}")
            
            mask_save_path = os.path.join(video_output_dir, f"{base_filename}_sam_mask.npy")
            np.save(mask_save_path, final_combined_mask)
            print(f"  Final mask saved to: {mask_save_path}")

    print("\n--- Batch Process Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process a dataset of IR videos to automatically segment thermal leaks.")
    
    # Changed from video_path to dataset_dir
    parser.add_argument("dataset_dir", type=str, 
                        help="Path to the root directory of the dataset (e.g., 'datasets/dataset_two_holes').")
    # Changed from output_dir to base_output_dir
    parser.add_argument("base_output_dir", type=str, 
                        help="Base directory to save all outputs. Subfolders will be created for each video.")
    
    parser.add_argument("--num_leaks", type=int, default=2, 
                        help="The number of leaks to detect per video.")
    
    prompt_group = parser.add_argument_group('Prompt Finding Parameters')
    prompt_group.add_argument("--min_area", type=int, default=5, help="Minimum pixel area for a candidate hotspot.")
    prompt_group.add_argument("--aspect_ratio_limit", type=float, default=4.0, help="Maximum aspect ratio to filter edge artifacts.")
    prompt_group.add_argument("--activity_quantile", type=float, default=0.98, help="Quantile to threshold activity map for candidates.")

    default_checkpoint = os.path.join('SAM', 'sam_checkpoints', 'sam_vit_b_01ec64.pth')
    parser.add_argument("--checkpoint_path", type=str, default=default_checkpoint, help="Path to the SAM model checkpoint file.")
    parser.add_argument("--model_type", type=str, default="vit_b", help="The type of SAM model to use.")

    args = parser.parse_args()
    
    if not os.path.isdir(args.dataset_dir):
        print(f"Error: Dataset directory not found at '{args.dataset_dir}'"); sys.exit(1)
    if not os.path.isfile(args.checkpoint_path):
        print(f"Error: SAM checkpoint not found at '{args.checkpoint_path}'"); sys.exit(1)
        
    main(args)