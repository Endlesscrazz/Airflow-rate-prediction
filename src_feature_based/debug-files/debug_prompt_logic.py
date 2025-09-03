# scripts/debug_prompting_logic.py
"""
Generates a detailed 2x2 visualization of the SAM prompting logic for a single video.
This script is designed to clearly explain how the algorithm selects correct leak
locations while ignoring large thermal artifacts.
"""
import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.colors

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Calculation Functions (Identical) ---
def _calculate_kendall_for_row(row_data, t):
    W = row_data.shape[0]; row_values = np.zeros(W, dtype=np.float64)
    for c in range(W):
        pixel_series = row_data[c, :]; val = 0.0
        if not np.any(np.isnan(pixel_series)) and len(pixel_series) > 1:
            try: val, _ = kendalltau(t, pixel_series)
            except (ValueError, IndexError): pass
            row_values[c] = val if np.isfinite(val) else 0.0
    return row_values

def generate_activity_map(frames):
    H, W, T = frames.shape; t = np.arange(T)
    print("Step 1: Calculating Kendall Tau Activity Map...")
    results = Parallel(n_jobs=-1)(delayed(_calculate_kendall_for_row)(frames[r, :, :], t) for r in tqdm(range(H), ncols=100))
    raw_map = np.vstack(results); raw_map[raw_map < 0] = 0
    return raw_map

def create_border_roi_mask(frame_shape, border_percent):
    H, W = frame_shape
    border_h, border_w = int(H * border_percent), int(W * border_percent)
    roi_mask = np.zeros(frame_shape, dtype=np.uint8)
    roi_mask[border_h : H - border_h, border_w : W - border_w] = 1
    return roi_mask

def find_prompts_and_artifacts(activity_map, num_prompts, quantile_thresh):
    print("Step 2: Finding prompts and generating artifacts...")
    active_pixels = activity_map[activity_map > 1e-9]
    if active_pixels.size == 0: return None
    activity_threshold = np.quantile(active_pixels, quantile_thresh)
    binary_mask = (activity_map >= activity_threshold).astype(np.uint8)
    
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1: return None

    all_candidates = []
    for i in range(1, num_labels):
        component_mask = (labels_img == i)
        peak_activity = np.max(activity_map[component_mask])
        all_candidates.append({'label_id': i, 'centroid': centroids[i], 'peak_activity': peak_activity})
        
    sorted_candidates = sorted(all_candidates, key=lambda x: x['peak_activity'], reverse=True)
    final_prompts = sorted_candidates[:num_prompts]
    
    return binary_mask, labels_img, sorted_candidates, final_prompts

# --- MODIFIED: The 2x2 Plotting Function for Clarity ---
def plot_prompting_pipeline(masked_activity_map, binary_mask, labels_img, sorted_candidates, final_prompts, save_path):
    """Generates a single, clearer 2x2 plot visualizing the prompting pipeline."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Visualizing the Prompt Selection Pipeline", fontsize=20)
    ((ax1, ax2), (ax3, ax4)) = axes

    # Panel 1: Masked Activity Map (Input) - Unchanged
    ax1.imshow(masked_activity_map, cmap='hot')
    ax1.set_title("Step 1: Activity Map within ROI", fontsize=14)
    ax1.axis('off')

    # Panel 2: Thresholded Hotspots - Unchanged
    ax2.imshow(binary_mask, cmap='gray')
    ax2.set_title("Step 2: Thresholded 'Hottest' Pixels", fontsize=14)
    ax2.axis('off')

    # --- NEW: Panel 3: Highlight Top Blobs ---
    # Create a black background image to draw on
    highlight_img = np.zeros(labels_img.shape, dtype=np.uint8)
    # Define a set of distinct, bright colors for the top blobs
    highlight_colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)] # Red, Green, Cyan, Magenta, Yellow
    
    # Create an RGB image to draw colored blobs
    color_labels_img = cv2.cvtColor(highlight_img, cv2.COLOR_GRAY2RGB)

    # Only highlight the top N blobs for clarity
    num_to_highlight = min(len(sorted_candidates), 5)
    for i in range(num_to_highlight):
        candidate = sorted_candidates[i]
        label_id = candidate['label_id']
        color_labels_img[labels_img == label_id] = highlight_colors[i % len(highlight_colors)]

    ax3.imshow(color_labels_img)
    ax3.set_title(f"Step 3: Top {num_to_highlight} Blobs Highlighted", fontsize=14)
    ax3.axis('off')

    # --- NEW: Panel 4: De-cluttered Ranked Prompts ---
    ax4.imshow(masked_activity_map, cmap='hot')
    ax4.set_title("Step 4: Blobs Ranked & Top 2 Selected", fontsize=14)
    ax4.axis('off')
    
    num_to_label = min(len(sorted_candidates), 5) # Only label the top 5
    final_prompt_centroids = [p['centroid'] for p in final_prompts]
    
    for i in range(num_to_label):
        cand = sorted_candidates[i]
        centroid = cand['centroid']
        rank = i + 1
        is_selected = any(np.array_equal(centroid, fc) for fc in final_prompt_centroids)
        
        if is_selected:
            ax4.scatter(centroid[0], centroid[1], s=1000, c='yellow', marker='*', edgecolor='black', zorder=5)
            ax4.text(centroid[0] + 15, centroid[1], f"#{rank}", color='yellow', fontsize=16, fontweight='bold')
        else:
            # For non-selected top blobs, just show the rank number
            ax4.text(centroid[0] + 15, centroid[1], f"#{rank}", color='cyan', fontsize=14, alpha=0.9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"\nSaved detailed pipeline visualization to: {save_path}")

def main(args):
    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    frames = scipy.io.loadmat(args.input)['TempFrames'].astype(np.float64)
    H, W, _ = frames.shape
    
    activity_map = generate_activity_map(frames)
    roi_mask = create_border_roi_mask((H, W), args.roi_border_percent)
    masked_activity_map = activity_map * roi_mask

    artifacts = find_prompts_and_artifacts(masked_activity_map, args.num_leaks, args.activity_quantile)
    
    if artifacts:
        binary_mask, labels_img, sorted_candidates, final_prompts = artifacts
        save_path = os.path.join(args.output_dir, f"{base_filename}_prompting_pipeline.png")
        plot_prompting_pipeline(masked_activity_map, binary_mask, labels_img, sorted_candidates, final_prompts, save_path)
    else:
        print("Could not generate artifacts, no prompts were found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a detailed 2x2 visualization of the SAM prompting logic.")
    parser.add_argument("--input", required=True, type=str, help="Path to the single .mat video file.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save the visualization.")
    parser.add_argument("--num_leaks", type=int, default=2, help="Number of leaks to select.")
    parser.add_argument("--activity_quantile", type=float, default=0.995)
    parser.add_argument("--roi_border_percent", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
  
"""
python src_feature_based/debug-files/debug_prompt_logic.py \
  --input datasets/Fluke_BrickCladding_2holes_0616_2025_noshutter/T1.4V_2.2Pa_2025-6-16-16-33-25_20_34_14_.mat \
  --output_dir presentation_assets \
  --num_leaks 2
"""