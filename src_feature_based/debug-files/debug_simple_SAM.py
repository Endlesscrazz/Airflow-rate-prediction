# debug_simple_SAM.py
"""
A targeted script for debugging SAM segmentation on a single video file
using manual point prompts with custom-sized bounding boxes.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import torch
import matplotlib.pyplot as plt

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Error: segment_anything library not found.", file=sys.stderr)
    sys.exit(1)

# ========================================================================================
# --- MANUAL PROMPTS DATABASE ---
# Edit the coordinates and box_size for the video you are testing.
# The key should be a unique part of the input filename.
# ========================================================================================
MANUAL_PROMPTS = {
    "T1.6V_2025-08-08-19-20-36": [
        {
            "name": "center_hole",
            "coord": (272, 328), # (row, col)
            "box_size": 30       # Standard box size for the clear leak
        },
        {
            "name": "bottom_left_hole",
            "coord": (360, 140), # (row, col)
            "box_size": 15       # A smaller, more precise box for the noisy leak
        }
    ],
    "T1.8V_2025-08-08-18-53-18_20_32_12":[
        {
            "name": "center_hole",
            "coord": (272, 328), # (row, col)
            "box_size": 30       # Standard box size for the clear leak
        },
        {
            "name": "bottom_left_hole",
            "coord": (360, 140), # (row, col)
            "box_size": 15       # A smaller, more precise box for the noisy leak
        }
    ]
}

# ========================================================================================
# --- HELPER FUNCTIONS ---
# ========================================================================================

def visualize_prompts_on_image(image, prompts, save_path, title):
    """Visualizes point prompts and their bounding boxes on an image."""
    plt.figure(figsize=(12, 9))
    plt.imshow(image, cmap='gray')
    
    for leak_def in prompts:
        r, c = leak_def["coord"]
        box_size = leak_def["box_size"]
        box_half = box_size // 2
        
        # Plot the center point
        plt.scatter(c, r, s=400, c='yellow', marker='*', edgecolor='black', label='Foreground Point')
        
        # Draw the bounding box
        rect = plt.Rectangle((c - box_half, r - box_half), box_size, box_size,
                             linewidth=1.5, edgecolor='cyan', facecolor='none', label=f'{box_size}x{box_size} Box')
        plt.gca().add_patch(rect)
            
    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def main(args):
    print("--- SAM Debug Script (with Precise Box Prompts) ---")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load Data and Model ---
    print(f"Loading video: {args.input}")
    mat_data = scipy.io.loadmat(args.input)
    frames = mat_data['TempFrames'].astype(np.float64)
    median_frame = np.median(frames, axis=2)
    H, W = median_frame.shape

    print("Loading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    predictor = SamPredictor(sam)

    # --- 2. Find and Visualize Manual Prompts ---
    input_filename = os.path.basename(args.input)
    prompts_to_use = None
    for key in MANUAL_PROMPTS:
        if key in input_filename:
            prompts_to_use = MANUAL_PROMPTS[key]
            break
    
    if not prompts_to_use:
        print(f"Error: No manual prompts found for '{input_filename}' in the database.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(prompts_to_use)} leak definitions to process.")
    prompt_vis_path = os.path.join(args.output_dir, "DEBUG_01_manual_prompts_with_boxes.png")
    visualize_prompts_on_image(median_frame, prompts_to_use, prompt_vis_path, "Manual Prompts on Median Frame")
    print(f"Saved prompt visualization to: {prompt_vis_path}")

    # --- 3. Run SAM to Generate Masks ---
    print("Running SAM with point and precise box prompts...")
    frame_norm_8bit = cv2.normalize(median_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    frame_rgb = cv2.cvtColor(frame_norm_8bit, cv2.COLOR_GRAY2BGR)
    predictor.set_image(frame_rgb)
    
    final_masks = []
    for leak_def in prompts_to_use:
        r, c = leak_def["coord"]
        box_size = leak_def["box_size"]
        box_half = box_size // 2

        # Create the point prompt
        point_coords = np.array([[c, r]])
        point_labels = np.array([1])

        # Create the bounding box prompt
        x1 = int(np.clip(c - box_half, 0, W-1))
        y1 = int(np.clip(r - box_half, 0, H-1))
        x2 = int(np.clip(c + box_half, 0, W-1))
        y2 = int(np.clip(r + box_half, 0, H-1))
        input_box = np.array([x1, y1, x2, y2])
        
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box[None, :],
            multimask_output=False
        )
        final_masks.append({"name": leak_def["name"], "mask": masks[0]})

    # --- 4. Visualize and Save Final Output ---
    print("Saving final output...")
    final_vis_image = frame_rgb.copy()
    colors = [[255, 255, 0], [0, 255, 255]] # Yellow, Cyan
    for i, mask_info in enumerate(final_masks):
        # Save the individual refined mask
        mask_save_path = os.path.join(args.output_dir, f"{mask_info['name']}_mask.npy")
        np.save(mask_save_path, mask_info["mask"])
        print(f"  Saved mask to: {mask_save_path}")

        # Add to visualization
        color = colors[i % len(colors)]
        color_overlay = np.zeros_like(final_vis_image); color_overlay[mask_info["mask"]] = color
        final_vis_image = cv2.addWeighted(final_vis_image, 1, color_overlay, 0.6, 0)
    
    final_vis_path = os.path.join(args.output_dir, "FINAL_segmentation.png")
    cv2.imwrite(final_vis_path, final_vis_image)
    print(f"Saved final segmentation to: {final_vis_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug SAM segmentation on a single file with precise box prompts.")
    parser.add_argument("--input", required=True, type=str, help="Path to the input .mat video file.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the SAM checkpoint file.")
    parser.add_argument("--model_type", type=str, default="vit_b", help="SAM model type.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save debug outputs.")
    args = parser.parse_args()
    main(args)

"""
python src_feature_based/debug-files/debug_simple_SAM.py \
  --input /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter/T1.6V_2025-08-08-19-20-36_20_32_12_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --output_dir debug_ouputs/Fluke_BrickCladding_2holes_0808_2025_noshutter/vid-3

vid-5:
python src_feature_based/debug-files/debug_simple_SAM.py \
  --input /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0808_2025_noshutter/T1.8V_2025-08-08-18-53-18_20_32_12_.mat \
  --checkpoint SAM/sam_checkpoints/sam_vit_b_01ec64.pth \
  --output_dir debug_ouputs/Fluke_BrickCladding_2holes_0808_2025_noshutter/vid-5

"""