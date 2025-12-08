# web_app/inference.py
import os
import sys
import torch
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry

# Ensure we can import from src_cnn_v3
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src_cnn_v3 import config_v3 as cfg
from src_cnn_v3.core.leak_detection import FusedSignalDetector
from src_cnn_v3.models_v3 import HybridCropRegressor
from src_cnn_v3.core.preprocessing import ThermalPreprocessor 
from web_app.utils_viz import visualize_patch

def get_dynamic_box_from_signal(score_map, center, padding=5):
    """
    Creates a tight prompt box based on signal blob size.
    """
    cx, cy = int(center[0]), int(center[1])
    H, W = score_map.shape
    
    radius = 60
    x1, y1 = max(0, cx-radius), max(0, cy-radius)
    x2, y2 = min(W, cx+radius), min(H, cy+radius)
    local_patch = score_map[y1:y2, x1:x2]
    
    peak_score = score_map[cy, cx]
    if peak_score == 0: peak_score = 1.0
        
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

def merge_close_candidates(candidates, min_dist=45):
    """
    Merges peaks that are too close to each other.
    Keeps the one with the higher score.
    Helps prevent splitting a single leak (like a slit) into two IDs.
    """
    if not candidates: return []
    
    # Sort by score descending (Keep hottest point)
    sorted_cands = sorted(candidates, key=lambda x: x['score'], reverse=True)
    kept = []
    
    for current in sorted_cands:
        cx, cy = current['centroid']
        is_close = False
        for k in kept:
            kx, ky = k['centroid']
            # Euclidean Distance
            dist = np.sqrt((cx-kx)**2 + (cy-ky)**2)
            if dist < min_dist:
                is_close = True
                break
        
        if not is_close:
            kept.append(current)
            
    return kept

class AirflowPredictor:
    def __init__(self, model_path, scaler_path, max_flow_rate, device=cfg.DEVICE):
        self.device = device
        self.max_flow_rate = max_flow_rate
        
        print(f"--- Loading Inference Engine on {self.device} ---")
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.scaler = checkpoint['scaler']
        
        num_features = self.scaler.mean_.shape[0]
        self.nn_model = HybridCropRegressor(
            num_tabular_features=num_features,
            lstm_hidden=checkpoint['config']['lstm_hidden_size'],
            lstm_layers=checkpoint['config']['lstm_layers']
        ).to(device)
        
        self.nn_model.load_state_dict(checkpoint['model_state_dict'])
        self.nn_model.eval()
        
        self.detector = FusedSignalDetector()
        
        sam = sam_model_registry[cfg.SAM_MODEL_TYPE](checkpoint=cfg.SAM_CHECKPOINT_PATH)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
        self.preprocessor = ThermalPreprocessor(
            resize_dim=cfg.RESIZE_DIM,
            blur_kernel=cfg.V3_PREPROCESS_PARAMS["BLUR_KERNEL_SIZE"],
            enable_temp_norm=cfg.V3_PREPROCESS_PARAMS["ENABLE_TEMPORAL_NORM"]
        )
        print("--- Inference Engine Ready ---")

    def process_video(self, frames, delta_t, sensitivity=0.35):
        print(">>> Starting Background Analysis...")
        
        # 1. Detection
        score_map = self.detector.get_score_map(frames)
        max_score = score_map.max()
        abs_threshold = max_score * sensitivity 
        
        candidates = self.detector.find_peaks(score_map, min_distance=20)
        valid_candidates = [c for c in candidates if c['score'] > abs_threshold]
        
        # --- MERGE STEP (Fixes Split Leaks) ---
        merged_candidates = merge_close_candidates(valid_candidates, min_dist=45)
        print(f"    - Merged {len(valid_candidates)} candidates down to {len(merged_candidates)}")
        
        merged_candidates.sort(key=lambda x: x['score'], reverse=True)

        # 2. SAM Prep
        avg_frame = frames.mean(axis=2)
        norm_img = cv2.normalize(avg_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        rgb_img = cv2.cvtColor(clahe.apply(norm_img), cv2.COLOR_GRAY2RGB)
        
        self.sam_predictor.set_image(rgb_img)
        
        # 3. Processing
        results = []
        frames_to_process = frames
        # Limit to 150 frames for AI Model consistency
        if frames.shape[2] > cfg.NUM_FRAMES_PER_SAMPLE:
            frames_to_process = frames[:, :, :cfg.NUM_FRAMES_PER_SAMPLE]
            
        for i, cand in enumerate(merged_candidates):
            cx, cy = cand['centroid']
            
            box_prompt = get_dynamic_box_from_signal(score_map, (cx, cy))
            
            masks, _, _ = self.sam_predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                box=box_prompt[None, :],
                multimask_output=False
            )
            mask = masks[0]
            
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 5:
                x1, y1, x2, y2 = box_prompt
                rect = ((cx, cy), (x2-x1, y2-y1), 0.0)
                cnt = None
            else:
                cnt = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(cnt)

            (ox, oy), (w, h), angle = rect
            
            area = cv2.contourArea(cnt) if cnt is not None else (w*h)
            aspect = max(w,h) / (min(w,h) + 1e-6)
            extent = area / (w*h + 1e-6)
            
            video_stack = self.preprocessor.process_sequence(
                frames_to_process, (ox, oy), (w, h), angle
            )
            
            vid_tensor = torch.tensor(video_stack, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(self.device)
            v_min, v_max = vid_tensor.min(), vid_tensor.max()
            if (v_max - v_min) > 1e-6:
                vid_tensor = (vid_tensor - v_min) / (v_max - v_min)
            else:
                vid_tensor = torch.zeros_like(vid_tensor)
            
            raw_feats = np.array([[delta_t, area, aspect, extent]])
            scaled_feats = self.scaler.transform(raw_feats)
            feat_tensor = torch.tensor(scaled_feats, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                pred_norm = self.nn_model(vid_tensor, feat_tensor)
                flow_rate = torch.clamp(pred_norm, 0.0, 1.0).item() * self.max_flow_rate
            
            vis_patch = video_stack[len(video_stack)//2]
            
            results.append({
                "temp_id": i,
                "centroid": (int(cx), int(cy)),
                "obb_box": cv2.boxPoints(rect).astype(int),
                "flow_rate": flow_rate,
                "mask": mask,
                "patch_vis": visualize_patch(vis_patch),
                "debug_features": {
                    "Area": int(area),
                    "Aspect": round(aspect, 2),
                    "Extent": round(extent, 2)
                }
            })
            
        # Final Sort by Flow Rate
        results.sort(key=lambda x: x['flow_rate'], reverse=True)
        for i, res in enumerate(results):
            res['id'] = i + 1
            
        return {
            "score_map": score_map,
            "vis_frame": rgb_img,
            "leaks": results
        }