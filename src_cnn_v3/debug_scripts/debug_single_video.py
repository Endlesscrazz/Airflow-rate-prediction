# src_cnn_v3/debug_files/debug_single_video.py
import argparse
import os
import sys
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from scipy.spatial import distance

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src_cnn_v3 import config_v3 as cfg
from src_cnn_v3.core.leak_detection import FusedSignalDetector
from src_cnn_v3.core.image_processing import crop_rotate_resize


# Template coordinates
GYPSUM_10HOLE_TEMPLATE = {
    1:  (145, 228), 2:  (204, 70), 3:  (367, 46), 4:  (526, 74),
    5:  (488, 204), 6:  (357, 198), 7:  (401, 319), 8:  (258, 315),
    9:  (553, 322), 10: (353, 414)
}

def get_dynamic_box_from_signal(score_map, center, padding=5):
    """
    Creates a prompt box based on the blob size in the Fused Signal Map.
    This prevents SAM from segmenting the large 'halo' around leaks.
    """
    cx, cy = int(center[0]), int(center[1])
    H, W = score_map.shape
    
    # 1. Extract local region to analyze (limit search radius)
    radius = 60
    x1, y1 = max(0, cx-radius), max(0, cy-radius)
    x2, y2 = min(W, cx+radius), min(H, cy+radius)
    local_patch = score_map[y1:y2, x1:x2]
    
    # 2. Threshold the local patch relative to the peak score
    # We take pixels that are at least 20% as intense as the peak
    peak_score = score_map[cy, cx]
    thresh_val = peak_score * 0.20
    binary_patch = (local_patch > thresh_val).astype(np.uint8)
    
    # 3. Find the connected component belonging to the center
    contours, _ = cv2.findContours(binary_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Default box if contour logic fails
    final_box = [cx-10, cy-10, cx+10, cy+10]
    
    # Convert local center to patch coordinates
    local_cx, local_cy = cx - x1, cy - y1
    
    for cnt in contours:
        # Check if this contour contains the center point
        if cv2.pointPolygonTest(cnt, (local_cx, local_cy), False) >= 0:
            lx, ly, lw, lh = cv2.boundingRect(cnt)
            
            # Convert back to global coordinates
            global_x1 = x1 + lx - padding
            global_y1 = y1 + ly - padding
            global_x2 = x1 + lx + lw + padding
            global_y2 = y1 + ly + lh + padding
            
            # Clip to image bounds
            final_box = [
                max(0, global_x1), max(0, global_y1),
                min(W, global_x2), min(H, global_y2)
            ]
            break
            
    return np.array(final_box)

def match_peaks_to_template(candidates, template):
    """Strict 1-to-1 matching."""
    matched = []
    available_cands = candidates.copy()
    
    for hole_id in sorted(template.keys()):
        if not available_cands: break
        ref_coord = template[hole_id]
        
        cand_coords = [c['centroid'] for c in available_cands]
        dists = distance.cdist([ref_coord], cand_coords)[0]
        best_idx = np.argmin(dists)
        
        candidate = available_cands.pop(best_idx)
        candidate['hole_id'] = hole_id
        matched.append(candidate)
        
    return matched

def main():
    parser = argparse.ArgumentParser(description="V3 Pipeline Debugger")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_leaks", type=int, default=10)
    parser.add_argument("--use_template", action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    # 1. Load Video
    try:
        frames = scipy.io.loadmat(args.video_path)['TempFrames'].astype(np.float32)
    except Exception as e:
        sys.exit(f"Error loading .mat file: {e}")

    # 2. Leak Detection
    print("Running Fused Signal Detection...")
    detector = FusedSignalDetector()
    score_map = detector.get_score_map(frames)
    
    # Get Peaks
    raw_candidates = detector.find_peaks(score_map, min_distance=20)
    raw_candidates.sort(key=lambda x: x['score'], reverse=True)
    top_n_candidates = raw_candidates[:args.num_leaks]

    # Assign IDs
    if args.use_template:
        print("Applying Template Matching...")
        final_candidates = match_peaks_to_template(top_n_candidates, GYPSUM_10HOLE_TEMPLATE)
        final_candidates.sort(key=lambda x: x['hole_id'])
    else:
        final_candidates = top_n_candidates
        for i, c in enumerate(final_candidates):
            c['hole_id'] = i + 1

    # 3. SAM Setup
    sam = sam_model_registry[cfg.SAM_MODEL_TYPE](checkpoint=cfg.SAM_CHECKPOINT_PATH)
    sam.to(device=cfg.DEVICE)
    predictor = SamPredictor(sam)
    
    # Prepare Input for SAM
    avg_frame = frames.mean(axis=2)
    norm_img = cv2.normalize(avg_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # V3 CHANGE: Use milder contrast enhancement to reduce halo visibility
    # CLAHE clip limit reduced from 2.0 to 1.0
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(norm_img)
    sam_input = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
    
    predictor.set_image(sam_input)

    vis_background = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
    standardized_patches = []
    patch_labels = []

    print("\n--- Processing Candidates (Signal-Guided Prompting) ---")
    for i, cand in enumerate(final_candidates):
        cx, cy = cand['centroid']
        
        # --- NEW LOGIC: Dynamic Box from Signal Map ---
        input_box = get_dynamic_box_from_signal(score_map, (cx, cy), padding=5)
        
        masks, _, _ = predictor.predict(
            point_coords=np.array([[cx, cy]]), 
            point_labels=np.array([1]), 
            box=input_box[None, :], 
            multimask_output=False 
        )
        mask = masks[0]

        # VISUALIZATION
        color = plt.cm.tab10(i % 10)[:3] 
        color = tuple(int(c*255) for c in color)
        
        mask_indices = mask > 0
        if np.any(mask_indices):
            roi = vis_background[mask_indices].astype(np.float32)
            color_layer = np.full_like(roi, color)
            blended = cv2.addWeighted(roi, 0.6, color_layer, 0.4, 0)
            vis_background[mask_indices] = blended.astype(np.uint8)

        # OBB
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            (ox, oy), (w, h), angle = rect
            
            box_pts = cv2.boxPoints(rect)
            box_pts = np.int32(box_pts)
            # Draw OBB in Green
            cv2.drawContours(vis_background, [box_pts], 0, (0, 255, 0), 2)
            # Draw Prompt Box in Blue (to debug the dynamic logic)
            cv2.rectangle(vis_background, (int(input_box[0]), int(input_box[1])), 
                          (int(input_box[2]), int(input_box[3])), (255, 0, 0), 1)
            
            cv2.putText(vis_background, f"ID{cand['hole_id']}", (int(cx), int(cy)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            patch = crop_rotate_resize(avg_frame, (ox, oy), (w, h), angle)
            standardized_patches.append(patch)
            patch_labels.append(f"ID {cand['hole_id']}\n{w:.0f}x{h:.0f}\n{angle:.0f}°")

    # --- SAVE IMAGES ---
    fig1, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(sam_input)
    ax[0].set_title("SAM Input (Mild CLAHE)")
    ax[1].imshow(score_map, cmap='hot')
    ax[1].set_title("Fused Signal Map")
    ax[2].imshow(vis_background)
    ax[2].set_title("Masks (Fill) + OBB (Green) + Prompt (Blue)")
    
    diag_path = os.path.join(args.output_dir, f"{video_name}_diagnostic_dynamic.png")
    plt.tight_layout()
    fig1.savefig(diag_path)
    plt.close(fig1)
    
    if standardized_patches:
        num_patches = len(standardized_patches)
        cols = 5
        rows = (num_patches + cols - 1) // cols
        fig2, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3.5))
        if rows == 1 and cols == 1: axes = [axes] # Handle single patch case
        else: axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < num_patches:
                ax.imshow(standardized_patches[i], cmap='inferno')
                ax.set_title(patch_labels[i])
                ax.axis('off')
            else:
                ax.axis('off')
        
        patch_path = os.path.join(args.output_dir, f"{video_name}_patches_dynamic.png")
        plt.tight_layout()
        fig2.savefig(patch_path)
        plt.close(fig2)
        print(f"\nSaved Images:\n1. {diag_path}\n2. {patch_path}")

if __name__ == "__main__":
    main()

"""
# GYPSUM-10-HOLE
python src_cnn_v3/debug_files/debug_single_video.py \
  --video_path "/scratch/general/vast/u1527145/datasets/Fluke_Gypsum_09032025_10holes_noshutter_Sameem/5P/temp_2025-10-24-10-50-36_23_31_8_.mat" \
  --output_dir "debug_output_v3/gypsum-10hole/5P/vid-2/iter-2" \
  --use_template \
  --num_leaks 10

15P
python src_cnn_v3/debug_scripts/debug_single_video.py \
  --video_path "/scratch/general/vast/u1527145/datasets/Fluke_Gypsum_09032025_10holes_noshutter_Sameem/15P/temp_2025-10-31-10-8-22_23_31_8_.mat" \
  --output_dir "debug_output_v3/gypsum-10hole/15P/vid-11/iter-1" \
  --use_template \
  --num_leaks 10

python src_cnn_v3/debug_scripts/debug_single_video.py \
  --video_path "/scratch/general/vast/u1527145/datasets/Fluke_Gypsum_09032025_10holes_noshutter_Sameem/5P/temp_2025-10-23-15-14-3_24_36_12_.mat" \
  --output_dir "debug_output_v3/gypsum-10hole/5P/vid-1-iter-1-notemplate" \
  --num_leaks 10
"""