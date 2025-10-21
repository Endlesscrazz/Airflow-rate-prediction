# scripts/debug_SAM_local_peaks.py
"""
UNIVERSAL & FLEXIBLE DEBUG SCRIPT (v10 - With Coordinate-based Assignment)
- Includes a 'two_hole_assign' mode for robustly assigning hole IDs based on proximity to known coordinates.
- Adds a dedicated verification plot to visually confirm the assignment of detected peaks to target locations.
- Retains all previous functionality including 'standard' mode and the 6-panel presentation visual.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import kendalltau, mstats
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

try:
    from segment_anything import sam_model_registry, SamPredictor
    from skimage.feature import peak_local_max
except ImportError:
    print("Error: Required libraries not found. Please run:", file=sys.stderr)
    print("pip install scikit-image segment-anything-py imageio matplotlib", file=sys.stderr)
    sys.exit(1)

# ========================================================================================
# --- COORDINATES FOR HOLE ASSIGNMENT ---
# ========================================================================================
# Coordinates are (row, col) which corresponds to (y, x)

# For 10-hole Gypsum dataset
SLIT_COORDS = {
    'Hole_1': np.array([233, 553]),
    'Hole_10': np.array([414, 353]),
}
TARGET_HOLE_COORDS = {
    2: np.array([79, 507]), 3: np.array([53, 344]), 4: np.array([78, 181]),
    5: np.array([213, 219]), 6: np.array([332, 149]), 7: np.array([326, 307]),
    8: np.array([325, 453]), 9: np.array([201, 355]),
}

# --- For 2-hole datasets ---
# ACTION: Verify and update these approximate (y, x) coordinates for your datasets.
TWO_HOLE_COORDS = {
    'hardyboard': { # 08132025
        1: np.array([322, 328]),  # Approx. center of "center hole"
        2: np.array([131, 497]),  # Approx. center of "right corner hole"
    },
    # 'brickcladding': { # Note: removed space for easier matching
    #     1: np.array([272, 328]),  # Approx. center of "center hole"
    #     2: np.array([360, 140]),  # Approx. center of "corner hole"
    # }
}

# ========================================================================================
# --- HELPER & CORE FUNCTIONS ---
# ========================================================================================

def save_parameters(args, output_dir):
    params_path = os.path.join(output_dir, "parameters.txt")
    with open(params_path, 'w') as f:
        f.write("--- Run Parameters ---\n")
        for arg, value in sorted(vars(args).items()): f.write(f"{arg}: {value}\n")
    print(f"Saved run parameters to: {params_path}")

def apply_spatial_filter(frames, filter_type='none', d=9, sigmaColor=75, sigmaSpace=75):
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
    print(f"  - Step 4: Creating {border_percent*100:.0f}% border ROI mask...")
    H, W = frame_shape; border_h, border_w = int(H * border_percent), int(W * border_percent)
    roi_mask = np.zeros(frame_shape, dtype=np.uint8)
    roi_mask[border_h : H - border_h, border_w : W - border_w] = 1
    return roi_mask

def create_panel_roi_mask(median_frame, erosion_iterations):
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

def find_prompts_by_local_peaks(activity_map, num_prompts, min_distance, abs_threshold, prompt_box_size, return_all=False):
    print(f"  - Step 5: Finding local peaks (min_dist: {min_distance}, threshold: {abs_threshold})...")
    coordinates = peak_local_max(activity_map, min_distance=min_distance, threshold_abs=abs_threshold)
    if coordinates.size == 0: 
        return [], []

    all_candidates = [{'centroid': np.array([c, r]), 'peak_activity': activity_map[r, c]} for r, c in coordinates]
    sorted_candidates = sorted(all_candidates, key=lambda x: x['peak_activity'], reverse=True)

    if return_all:
        print(f"    - Found {len(sorted_candidates)} total peaks (return_all=True).")
        return sorted_candidates, sorted_candidates

    top_candidates = sorted_candidates[:num_prompts]
    print(f"    - Found {len(sorted_candidates)} total peaks, selecting top {len(top_candidates)}.")
    return top_candidates, sorted_candidates

def filter_and_assign_prompts(all_candidates, slit_coords_to_filter, target_hole_coords, exclusion_radius=30):
    print(f"  - Step 5b: Filtering and assigning prompts...")
    filtered_prompts = []
    if slit_coords_to_filter:
        slit_locations = np.array([coord[::-1] for coord in slit_coords_to_filter.values()])
        for cand in all_candidates:
            if np.all(cdist(cand['centroid'].reshape(1, 2), slit_locations) > exclusion_radius):
                filtered_prompts.append(cand)
    else:
        filtered_prompts = all_candidates
            
    print(f"    - Found {len(all_candidates)} total peaks, {len(filtered_prompts)} remaining after filtering.")
    if not filtered_prompts: return []

    assigned_prompts = {}
    remaining_prompts = list(filtered_prompts)
    for hole_id, target_coord in target_hole_coords.items():
        if not remaining_prompts: break
        target_xy = target_coord[::-1]
        distances_to_target = [np.linalg.norm(p['centroid'] - target_xy) for p in remaining_prompts]
        closest_prompt_idx = np.argmin(distances_to_target)
        closest_prompt = remaining_prompts.pop(closest_prompt_idx)
        closest_prompt['hole_id'] = hole_id
        assigned_prompts[hole_id] = closest_prompt

    final_prompts = [assigned_prompts[key] for key in sorted(assigned_prompts.keys())]
    print(f"    - Successfully assigned {len(final_prompts)} prompts to target holes.")
    return final_prompts

def create_assignment_visual(activity_map, all_candidates, final_prompts, target_coords, save_path):
    print("\n--- Creating Assignment Verification Visual ---")
    plt.figure(figsize=(12, 12))
    plt.imshow(activity_map, cmap='hot')
    if all_candidates:
        all_coords = np.array([p['centroid'] for p in all_candidates])
        plt.scatter(all_coords[:, 0], all_coords[:, 1], s=100, facecolors='none', edgecolors='cyan', lw=1.5, label='All Found Peaks')
    target_x = [c[1] for c in target_coords.values()]
    target_y = [c[0] for c in target_coords.values()]
    plt.scatter(target_x, target_y, s=300, c='red', marker='+', lw=2, label='Target Locations')
    if final_prompts:
        for prompt in final_prompts:
            hole_id, prompt_coord = prompt['hole_id'], prompt['centroid']
            target_coord_xy = target_coords[hole_id][::-1]
            plt.plot([target_coord_xy[0], prompt_coord[0]], [target_coord_xy[1], prompt_coord[1]], 'w--', lw=1.5)
            plt.scatter(prompt_coord[0], prompt_coord[1], s=600, c='yellow', marker='*', edgecolor='black', zorder=5)
            plt.text(prompt_coord[0] + 10, prompt_coord[1] - 10, f"Hole {hole_id}", color='white', fontsize=12, fontweight='bold')
    plt.title('Debug Map: Prompts Assigned & Labeled', fontsize=16)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"  - Saved assignment verification to: {save_path}")

def run_sam_with_box_and_point_prompts(frame_rgb, top_candidates, predictor, prompt_box_size):
    all_final_masks = []; predictor.set_image(frame_rgb)
    H, W, _ = frame_rgb.shape
    for cand in top_candidates:
        center_x, center_y = cand['centroid']; box_half = prompt_box_size // 2
        x1, y1 = int(np.clip(center_x - box_half, 0, W-1)), int(np.clip(center_y - box_half, 0, H-1))
        x2, y2 = int(np.clip(center_x + box_half, 0, W-1)), int(np.clip(center_y + box_half, 0, H-1))
        input_box = np.array([x1, y1, x2, y2])
        point_coords, point_labels = cand['centroid'].reshape(1, 2), np.array([1])
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=input_box[None, :], multimask_output=False)
        if len(masks) > 0: all_final_masks.append(masks[0])
    return all_final_masks

def enhance_frame(frame):
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(frame_normalized)

def create_presentation_visual(artifacts, save_path):
    print("\n--- Creating Presentation Visual Storyboard ---")
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle("Automated Segmentation Pipeline", fontsize=24, y=0.97)
    titles = ["1. Raw Input (Enhanced)", "2. Pre-processed Data (Smoothed)", "3. Activity Map", "4. ROI Mask Applied", "5. Prompts Assigned & Labeled", "6. Final Segmentation"]
    
    axes[0, 0].imshow(artifacts['enhanced_frame'], cmap='gray')
    axes[0, 1].imshow(artifacts['smoothed_frame'], cmap='gray')
    axes[0, 2].imshow(artifacts['activity_map'], cmap='hot')
    axes[1, 0].imshow(artifacts['masked_activity_map'], cmap='hot')
    
    ax = axes[1, 1]
    ax.imshow(artifacts['masked_activity_map'], cmap='hot')
    if artifacts.get('all_candidates'):
        ax.scatter([c['centroid'][0] for c in artifacts['all_candidates']], [c['centroid'][1] for c in artifacts['all_candidates']], s=150, facecolors='none', edgecolors='cyan', lw=1.5, label='All Found Peaks')
    if artifacts.get('final_prompts'):
        prompts = artifacts['final_prompts']
        ax.scatter([p['centroid'][0] for p in prompts], [p['centroid'][1] for p in prompts], s=600, c='yellow', marker='*', edgecolor='black', zorder=5, label='Assigned Prompts')
        for p in prompts:
            if 'hole_id' in p:
                ax.text(p['centroid'][0] + 15, p['centroid'][1] + 15, str(p['hole_id']), color='white', fontsize=16, fontweight='bold')
    ax.legend()

    axes[1, 2].imshow(cv2.cvtColor(artifacts['segmentation_image'], cv2.COLOR_BGR2RGB))
            
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i], fontsize=16); ax.axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(save_path); plt.close()
    print(f"  - Saved presentation visual to: {save_path}")

# ========================================================================================
# --- MAIN EXECUTION SCRIPT ---
# ========================================================================================
def main(args):
    print("--- Universal Hotspot Segmentation ---"); os.makedirs(args.output_dir, exist_ok=True)
    save_parameters(args, args.output_dir); print("\nLoading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint); device = "cuda" if torch.cuda.is_available() else "cpu"; sam.to(device=device)
    predictor = SamPredictor(sam)
    frames = scipy.io.loadmat(args.input)['TempFrames'].astype(np.float64); H, W, T = frames.shape
    
    artifacts = {}
    
    spatially_filtered_frames = apply_spatial_filter(frames, args.spatial_filter)
    temporally_smoothed_frames = apply_temporal_smoothing(spatially_filtered_frames, args.temporal_smooth_window)
    activity_map = generate_activity_map(temporally_smoothed_frames, args.activity_method, args.env_para)
    target_frame = np.median(frames, axis=2)
    if args.roi_method == 'border': roi_mask = create_border_roi_mask((H, W), args.roi_border_percent)
    else: roi_mask = create_panel_roi_mask(target_frame, args.roi_erosion)
    masked_activity_map = activity_map * roi_mask
    
    artifacts.update({
        'enhanced_frame': enhance_frame(frames[:, :, T // 2]),
        'smoothed_frame': enhance_frame(temporally_smoothed_frames[:, :, T // 2]),
        'activity_map': activity_map,
        'masked_activity_map': masked_activity_map
    })
    
    _, all_candidates = find_prompts_by_local_peaks(masked_activity_map, 20, args.peak_min_distance, args.peak_abs_threshold, args.prompt_box_size, return_all=True)
    artifacts['all_candidates'] = all_candidates
    
    final_prompts, target_coords_for_visual = [], None

    if args.processing_mode == 'multi_hole_filter':
        print("\nRunning in 'multi_hole_filter' mode.")
        final_prompts = filter_and_assign_prompts(all_candidates, SLIT_COORDS, TARGET_HOLE_COORDS)
        target_coords_for_visual = TARGET_HOLE_COORDS

    elif args.processing_mode == 'two_hole_assign':
        print("\nRunning in 'two_hole_assign' mode.")
        material_coords, input_path_lower = None, args.input.lower().replace("_", "").replace("-", "")
        for material, coords in TWO_HOLE_COORDS.items():
            if material in input_path_lower:
                material_coords, target_coords_for_visual = coords, coords
                print(f"  - Using '{material}' coordinates for assignment.")
                break
        if not material_coords:
            print("Could not determine material for 'two_hole_assign' mode. Exiting.", file=sys.stderr); return
        final_prompts = filter_and_assign_prompts(all_candidates, {}, material_coords)

    else: # standard mode
        print("\nRunning in 'standard' mode (intensity-based).")
        final_prompts = all_candidates[:args.num_leaks]
        for i, p in enumerate(final_prompts): p['hole_id'] = i + 1
    
    artifacts['final_prompts'] = final_prompts

    if target_coords_for_visual:
        verification_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.input))[0]}_prompt_assignment.png")
        create_assignment_visual(masked_activity_map, all_candidates, final_prompts, target_coords_for_visual, verification_path)

    if not final_prompts: print("Could not find/assign prompts. Exiting.", file=sys.stderr); return

    print("\nRunning SAM segmentation...")
    frame_rgb = cv2.cvtColor(cv2.normalize(target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_GRAY2BGR)
    all_final_masks = run_sam_with_box_and_point_prompts(frame_rgb, final_prompts, predictor, args.prompt_box_size)
    if not all_final_masks: print("SAM generated no masks. Exiting.", file=sys.stderr); return

    print("\nSaving final outputs...")
    segmentation_image = frame_rgb.copy(); colors = [[255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 0, 128], [0, 128, 128]]
    for i, mask in enumerate(all_final_masks):
        hole_id = final_prompts[i].get('hole_id')
        np.save(os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.input))[0]}_mask_{hole_id}.npy"), mask)
        color_overlay = np.zeros_like(segmentation_image); color_overlay[mask] = colors[i % len(colors)]
        segmentation_image = cv2.addWeighted(segmentation_image, 1, color_overlay, 0.6, 0)
    
    print(f"  - Saved {len(all_final_masks)} masks with assigned hole IDs.")
    artifacts['segmentation_image'] = segmentation_image

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
    
    parser.add_argument("--processing_mode", type=str, default="standard", choices=['standard', 'two_hole_assign', 'multi_hole_filter'])

    param_group = parser.add_argument_group('Pipeline Control Parameters')
    param_group.add_argument("--num_leaks", type=int, default=1, help="Number of leaks to find in 'standard' mode.")
    param_group.add_argument("--env_para", type=int, default=1, choices=[1, -1, 0])
    param_group.add_argument("--activity_method", type=str, default="kendall_tau", choices=['theil_sen', 'kendall_tau'])
    param_group.add_argument("--temporal_smooth_window", type=int, default=3)
    param_group.add_argument("--spatial_filter", type=str, default="bilateral", choices=['none', 'bilateral'])
    param_group.add_argument("--roi_method", type=str, default="otsu", choices=['otsu', 'border'])
    param_group.add_argument("--roi_border_percent", type=float, default=0.1)
    param_group.add_argument("--roi_erosion", type=int, default=3)

    prompt_group = parser.add_argument_group('Prompt Finding Method')
    prompt_group.add_argument("--peak_min_distance", type=int, default=50)
    prompt_group.add_argument("--peak_abs_threshold", type=float, default=0.2)
    prompt_group.add_argument("--prompt_box_size", type=int, default=30)
    
    parser.add_argument("--create_presentation_visual", action='store_true', help="Generates a 2x3 composite image visualizing the pipeline.")

    args = parser.parse_args()
    main(args)
"""
python src_feature_based/debug-files/debug_SAM_local_peaks.py \
  --input /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_HardyBoard_08132025_2holes_noshutter/T1.4V_2025-08-14-15-47-12_21_34_13_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --output_dir debug_ouputs/Fluke_HardyBoard_08132025_2holes_noshutter/vid-1/iter-1-local-peak \
  --num_leaks 2 \
  --roi_method border \
  --roi_border_percent 0.10 \
  --peak_min_distance 50 \
  --create_presentation_visual \
  --peak_abs_threshold 0.2 \
  --processing_mode standard
  
"""