"""

vid-4:
python src/debug-files/run_SAM_hpf.py \
  --input datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-44-18_20_30_10_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/run-sam-debug/vid-4/iter-1-hpf 

vid-12:
  python src/debug-files/run_SAM_hpf.py \
  --input datasets/dataset_two_holes/T2.0V_6.3Pa_2025-6-16-17-41-25_20_26_6_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --output_dir output_SAM/run-sam-debug/vid-12/iter-1-hpf \
  --temporal_smooth_window 3
"""

# robust_SAM_hpf.py
"""
DEFINITIVE SCRIPT v2
- Uses a High-Pass Filter (Standard Deviation of Frame-to-Frame Differences)
  to create an activity map that captures thermal "flutter".
- This is robust to heating, cooling, and oscillating leaks.
- Returns to a simple, clean pipeline based on this superior metric.
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

def generate_hpf_activity_map(frames):
    """
    Generates an activity map by calculating the standard deviation of the
    frame-to-frame differences. This is a form of high-pass filtering that
    captures thermal volatility or "flutter".
    """
    print("Step 1: Generating High-Pass Filter (Volatility) Activity Map...")
    if frames.shape[2] < 2:
        return np.zeros(frames.shape[:2], dtype=np.float64)
    
    # Calculate the difference between consecutive frames along the time axis
    frame_diffs = np.diff(frames, axis=2)
    
    # Calculate the standard deviation of these differences for each pixel
    # This measures how much the pixel's temperature "flutters"
    activity_map = np.std(frame_diffs, axis=2)
    
    return activity_map

def find_prompts_by_largest_area(activity_map, num_prompts, quantile_thresh):
    """
    Finds prompts by simply taking the N largest blobs above a quantile threshold.
    """
    print(f"Step 2: Finding the {num_prompts} largest active blobs (quantile: {quantile_thresh:.3f})...")
    if activity_map is None or not np.any(activity_map > 1e-9): return [], []

    active_pixels = activity_map[activity_map > 1e-9]
    if active_pixels.size == 0: return [], []

    activity_threshold = np.quantile(active_pixels, quantile_thresh)
    binary_mask = (activity_map >= activity_threshold).astype(np.uint8)
    
    # Optional: A small amount of morphological closing can help connect noisy blobs
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1: return [], []

    all_candidates = []
    for i in range(1, num_labels):
        all_candidates.append({
            'centroid': centroids[i],
            'stat': stats[i],
            'area': stats[i, cv2.CC_STAT_AREA]
        })

    sorted_by_area = sorted(all_candidates, key=lambda x: x['area'], reverse=True)
    top_candidates = sorted_by_area[:num_prompts]
    
    return top_candidates, all_candidates

# --- Visualization and SAM Functions ---

def visualize_clean_prompts(activity_map, all_candidates, final_prompts, save_path):
    """Visualizes the blobs and the final selected prompts."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(activity_map, cmap='hot')
    fig.colorbar(ax.images[0], ax=ax, label='Activity (Std. Dev. of Frame Diffs)')
    ax.set_title('High-Pass Filter Debug Map: Prompts from Largest Volatile Blobs')

    if all_candidates:
        cand_x = [c['centroid'][0] for c in all_candidates]
        cand_y = [c['centroid'][1] for c in all_candidates]
        ax.scatter(cand_x, cand_y, s=100, facecolors='none', edgecolors='cyan', lw=1.5, label='All Found Blobs')

    if final_prompts:
        prompt_coords = np.array([p['centroid'] for p in final_prompts])
        ax.scatter(prompt_coords[:, 0], prompt_coords[:, 1], s=600, c='yellow', marker='*',
                   edgecolor='black', label='Final Selected Prompts (Largest)', zorder=5)

    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)

def run_sam_with_box_prompts(frame_rgb, top_candidates, predictor):
    """Runs SAM using a bounding box prompt for each candidate."""
    all_final_masks = []
    predictor.set_image(frame_rgb)
    for cand in top_candidates:
        stat = cand['stat']
        x1, y1 = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
        x2, y2 = x1 + stat[cv2.CC_STAT_WIDTH], y1 + stat[cv2.CC_STAT_HEIGHT]
        input_box = np.array([x1, y1, x2, y2])
        masks, _, _ = predictor.predict(
            point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False,
        )
        if len(masks) > 0: all_final_masks.append(masks[0])
    return all_final_masks

def main(args):
    """Main execution function for the HPF pipeline."""
    print("--- HPF-Based Hotspot Segmentation (Debug Mode) ---")
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

    # --- New Simplified Pipeline ---
    activity_map = generate_hpf_activity_map(frames)
    top_candidates, all_candidates = find_prompts_by_largest_area(
        activity_map, num_prompts=args.num_leaks, quantile_thresh=args.activity_quantile
    )

    if not top_candidates:
        print("Could not find any prompts. Exiting.", file=sys.stderr); sys.exit(1)
    
    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    debug_plot_path = os.path.join(args.output_dir, f"{base_filename}_prompt_verification.png")
    visualize_clean_prompts(activity_map, all_candidates, top_candidates, debug_plot_path)

    print("\nStep 3: Running SAM segmentation...")
    target_frame = np.median(frames, axis=2)
    frame_normalized_8bit = cv2.normalize(target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    frame_rgb = cv2.cvtColor(frame_normalized_8bit, cv2.COLOR_GRAY2BGR)
    all_final_masks = run_sam_with_box_prompts(frame_rgb, top_candidates, predictor)

    if not all_final_masks:
        print("SAM did not generate masks. Exiting.", file=sys.stderr); sys.exit(1)

    # ... (Saving logic remains the same) ...
    print("\nStep 4: Saving final outputs...")
    final_combined_mask = np.zeros((H, W), dtype=bool)
    segmentation_image = frame_rgb.copy()
    colors = [[1, 1, 0], [0, 1, 1]] # Yellow, Cyan
    for i, mask in enumerate(all_final_masks):
        color = colors[i % len(colors)]
        color_overlay = np.zeros_like(segmentation_image)
        color_overlay[mask] = (np.array(color) * 255).astype(np.uint8) # Use BGR for OpenCV
        segmentation_image = cv2.addWeighted(segmentation_image, 1, color_overlay, 0.6, 0)
        final_combined_mask = np.logical_or(final_combined_mask, mask)
    plt.figure(figsize=(12, 9))
    plt.imshow(cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Final Segmentation for {base_filename}")
    plot_save_path = os.path.join(args.output_dir, f"{base_filename}_sam_segmentation.png")
    plt.savefig(plot_save_path); plt.close()
    mask_save_path = os.path.join(args.output_dir, f"{base_filename}_sam_mask.npy")
    np.save(mask_save_path, final_combined_mask)
    print(f"  Saved outputs to {args.output_dir}")
    print("\n--- HPF script finished. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simplified and robust script for debugging hotspot segmentation using a High-Pass Filter (Std Dev of Diffs).")
    
    parser.add_argument("--input", required=True, type=str, help="Path to the specific .mat file to process.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the SAM model checkpoint file.")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save all outputs for this run.")
    
    param_group = parser.add_argument_group('Pipeline Control Parameters')
    param_group.add_argument("--num_leaks", type=int, default=2, help="The number of leaks (largest blobs) to select.")
    param_group.add_argument("--activity_quantile", type=float, default=0.99, help="Quantile to threshold activity map. Higher value (e.g., 0.995) is more selective.")

    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found at '{args.input}'", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"Error: SAM checkpoint not found at '{args.checkpoint}'", file=sys.stderr); sys.exit(1)
        
    main(args)