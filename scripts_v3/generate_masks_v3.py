# scripts_V3/generate_masks_v3.py
import os
import sys
import glob
import json
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from scipy.spatial import distance
from segment_anything import sam_model_registry, SamPredictor

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v3 import config_v3 as cfg
from src_cnn_v3.core.leak_detection import FusedSignalDetector

# --- HELPER FUNCTIONS ---

def get_dynamic_box_from_signal(score_map, center, padding=5):
    """Creates a tight prompt box based on signal blob size."""
    cx, cy = int(center[0]), int(center[1])
    H, W = score_map.shape
    
    radius = 60
    x1, y1 = max(0, cx-radius), max(0, cy-radius)
    x2, y2 = min(W, cx+radius), min(H, cy+radius)
    local_patch = score_map[y1:y2, x1:x2]
    
    peak_score = score_map[cy, cx]
    thresh_val = peak_score * 0.20
    binary_patch = (local_patch > thresh_val).astype(np.uint8)
    
    contours, _ = cv2.findContours(binary_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_box = [cx-10, cy-10, cx+10, cy+10]
    local_cx, local_cy = cx - x1, cy - y1
    
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, (local_cx, local_cy), False) >= 0:
            lx, ly, lw, lh = cv2.boundingRect(cnt)
            global_x1 = x1 + lx - padding
            global_y1 = y1 + ly - padding
            global_x2 = x1 + lx + lw + padding
            global_y2 = y1 + ly + lh + padding
            final_box = [max(0, global_x1), max(0, global_y1), min(W, global_x2), min(H, global_y2)]
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
        
        if dists[best_idx] < 60:
            candidate = available_cands.pop(best_idx)
            candidate['hole_id'] = hole_id
            matched.append(candidate)
        
    return matched

def get_obb_features(mask, hole_id):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 5: return None
    rect = cv2.minAreaRect(cnt)
    (center_x, center_y), (w, h), angle = rect
    return {
        "hole_id": hole_id, "center_x": center_x, "center_y": center_y,
        "obb_width": w, "obb_height": h, "obb_angle": angle,
        "area_px": cv2.contourArea(cnt), 
        "aspect_ratio": max(w, h)/(min(w, h)+1e-6), 
        "extent": cv2.contourArea(cnt)/(w*h+1e-6)
    }

def main():
    print("--- V3 Batch Processing: Config-Driven Detection ---")
    
    sam = sam_model_registry[cfg.SAM_MODEL_TYPE](checkpoint=cfg.SAM_CHECKPOINT_PATH)
    sam.to(device=cfg.DEVICE)
    predictor = SamPredictor(sam)
    detector = FusedSignalDetector()

    for d_key, d_conf in cfg.DATASET_CONFIGS.items():
        dataset_dir = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf["dataset_subfolder"])

        target_num_leaks = d_conf.get("num_leaks", 2)
        template = d_conf.get("template", None)
        
        print(f"\nProcessing: {d_key} | Expecting {target_num_leaks} leaks")
        
        mat_files = glob.glob(os.path.join(dataset_dir, "**", "*.mat"), recursive=True)
        
        for mat_path in tqdm(mat_files):
            try:
                video_id = os.path.splitext(os.path.basename(mat_path))[0]
                rel_path = os.path.relpath(os.path.dirname(mat_path), dataset_dir)
                out_dir = os.path.join(cfg.INTERMEDIATE_DATA_DIR, d_conf["dataset_subfolder"], rel_path, video_id)
                os.makedirs(out_dir, exist_ok=True)
                
                json_path = os.path.join(out_dir, f"{video_id}_features.json")
                if os.path.exists(json_path): continue

                frames = scipy.io.loadmat(mat_path)['TempFrames'].astype(np.float32)
                
                # A. Detect Peaks (Find everything)
                score_map = detector.get_score_map(frames)
                candidates = detector.find_peaks(score_map, min_distance=20)
                
                # B. Assign IDs & Filter
                if template:
                    final_candidates = match_peaks_to_template(candidates, template)
                    final_candidates.sort(key=lambda x: x['hole_id'])
                else:
                    # Fallback to score-based sorting for non-template datasets
                    final_candidates = candidates[:target_num_leaks]
                    for i, c in enumerate(final_candidates):
                        c['hole_id'] = i + 1
                
                if not final_candidates: continue

                # C. SAM Prep
                avg_frame = frames.mean(axis=2)
                norm_img = cv2.normalize(avg_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
                sam_input = cv2.cvtColor(clahe.apply(norm_img), cv2.COLOR_GRAY2RGB)
                predictor.set_image(sam_input)
                
                vis_img = sam_input.copy()
                feature_list = []

                # D. Generate Masks
                for i, cand in enumerate(final_candidates):
                    hole_id = cand['hole_id']
                    cx, cy = cand['centroid']
                    input_box = get_dynamic_box_from_signal(score_map, (cx, cy))
                    
                    masks, _, _ = predictor.predict(
                        point_coords=np.array([[cx, cy]]),
                        point_labels=np.array([1]),
                        box=input_box[None, :],
                        multimask_output=False
                    )
                    binary_mask = masks[0]
                    
                    mask_path = os.path.join(out_dir, f"{video_id}_mask_{hole_id}.npy")
                    np.save(mask_path, binary_mask)
                    
                    feats = get_obb_features(binary_mask, hole_id)
                    if feats:
                        feats["mask_path"] = mask_path
                        feature_list.append(feats)
                        
                        box = cv2.boxPoints(((feats['center_x'], feats['center_y']), 
                                             (feats['obb_width'], feats['obb_height']), 
                                             feats['obb_angle']))
                        cv2.drawContours(vis_img, [np.int32(box)], 0, (0, 255, 0), 2)
                        cv2.rectangle(vis_img, (int(input_box[0]), int(input_box[1])), 
                                      (int(input_box[2]), int(input_box[3])), (255, 0, 0), 1)
                        cv2.putText(vis_img, f"ID{hole_id}", (int(cx), int(cy)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                with open(json_path, 'w') as f:
                    json.dump(feature_list, f, indent=4)
                    
                fig, ax = plt.subplots(1, 2, figsize=(14, 7))
                ax[0].imshow(score_map, cmap='hot'); ax[0].set_title("Fused Signal Map")
                ax[1].imshow(vis_img); ax[1].set_title(f"Masks + OBB (Green) + Prompt (Blue)")
                plt.savefig(os.path.join(out_dir, f"{video_id}_verification.png"))
                plt.close()
                
            except Exception as e:
                print(f"Error processing {mat_path}: {e}")
                continue

if __name__ == "__main__":
    main()