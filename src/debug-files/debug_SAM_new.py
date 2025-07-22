import cv2
import numpy as np
import scipy.io
from scipy import stats
from segment_anything import sam_model_registry, SamPredictor
import torch
from joblib import Parallel, delayed
import time
import os

# --- 1. CONFIGURATION ---
# --- Adjust these parameters as needed ---

# Input/Output Configuration
MAT_FILE_PATH = 'datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-44-18_20_30_10_.mat' # <--- CHANGE THIS to your .mat file path
MAT_VARIABLE_NAME = 'TempFrames'   # <--- CHANGE THIS to the variable name inside your .mat file
OUTPUT_DIR = 'output_SAM/debug/debug_SAM_new/vid-4/iter-1' # <--- CHANGE THIS to your desired output folder name

# SAM Model Configuration
SAM_CHECKPOINT_PATH = "SAM/sam_checkpoints/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HOTSPOT_QUANTILE_THRESHOLD = 0.98

# Visualization Parameters
ALL_PROMPTS_COLOR = (255, 255, 0)  # Cyan (B, G, R)
TOP_PROMPTS_COLOR = (0, 255, 255)  # Yellow (B, G, R)
SAM_MASK_COLOR = (0, 255, 0)       # Green (B, G, R)
MASK_ALPHA = 0.5                   

# --- 2. CORE FUNCTIONS (Unchanged) ---
def calculate_slope_for_pixel(pixel_time_series):
    time_indices = np.arange(len(pixel_time_series))
    try:
        slope, _, _, _ = stats.theilslopes(pixel_time_series, time_indices, 0.90)
    except (ValueError, IndexError):
        slope = 0.0
    return slope if np.isfinite(slope) else 0.0

def generate_slope_map_parallel(video_data):
    height, width, num_frames = video_data.shape
    pixel_matrix = video_data.reshape(-1, num_frames)
    print(f"Calculating slopes for {pixel_matrix.shape[0]} pixels across {num_frames} frames.")
    print("Using Joblib for parallel processing...")
    start_time = time.time()
    slopes = Parallel(n_jobs=-1, verbose=10)(
        delayed(calculate_slope_for_pixel)(pixel_matrix[i]) for i in range(pixel_matrix.shape[0])
    )
    end_time = time.time()
    print(f"Slope calculation finished in {end_time - start_time:.2f} seconds.")
    slope_map = np.array(slopes).reshape(height, width)
    return slope_map

# --- *** UPDATED PROMPT FINDING FUNCTION with QUANTILE *** ---
def find_hotspot_regions_quantile(slope_map, num_regions=2, quantile=0.999):
    """
    Finds distinct hotspot regions using a robust quantile threshold.
    """
    # Filter out non-positive slopes to focus on heating
    positive_slopes = slope_map[slope_map > 0]
    if positive_slopes.size == 0:
        return []

    # 1. Calculate the threshold using the specified quantile of *positive* slopes
    threshold_value = np.quantile(positive_slopes, quantile)
    
    # 2. Create a binary map
    binary_map = np.where(slope_map >= threshold_value, 255, 0).astype(np.uint8)

    # 3. Find contours (the distinct blobs)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    hotspot_regions = []
    # 4. For each region, find its true peak
    for cnt in contours:
        mask = np.zeros(slope_map.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        _, max_val, _, max_loc = cv2.minMaxLoc(slope_map, mask=mask)
        hotspot_regions.append({
            'peak_value': max_val,
            'point': np.array([[max_loc[0], max_loc[1]]])
        })

    # 5. Rank regions by their peak value and return the top N
    hotspot_regions.sort(key=lambda r: r['peak_value'], reverse=True)
    return hotspot_regions[:num_regions]

# --- 3. VISUALIZATION AND SAM FUNCTIONS (Unchanged) ---
# ... (Functions get_sam_masks, draw_activity_map, draw_final_output)
def get_sam_masks(predictor, image_rgb, prompts):
    if not prompts: return []
    predictor.set_image(image_rgb)
    all_masks = []
    for prompt in prompts:
        masks, scores, _ = predictor.predict(
            point_coords=prompt['point'],
            point_labels=np.array([1]), 
            multimask_output=False
        )
        all_masks.append(masks[0])
    return all_masks

def draw_activity_map(base_image, all_prompts, top_prompts):
    activity_frame = base_image.copy()
    for prompt in all_prompts:
        center_point = (prompt['point'][0, 0], prompt['point'][0, 1])
        cv2.circle(activity_frame, center_point, 10, ALL_PROMPTS_COLOR, 2)
    for prompt in top_prompts:
        center_point = (prompt['point'][0, 0], prompt['point'][0, 1])
        cv2.drawMarker(activity_frame, center_point, TOP_PROMPTS_COLOR, 
                       markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
    return activity_frame

def draw_final_output(base_image, sam_masks):
    output_frame = base_image.copy()
    for mask in sam_masks:
        color_overlay = np.zeros_like(output_frame)
        color_overlay[mask] = SAM_MASK_COLOR
        output_frame = cv2.addWeighted(output_frame, 1, color_overlay, MASK_ALPHA, 0)
    return output_frame

# --- 4. MAIN EXECUTION SCRIPT (Updated) ---

if __name__ == "__main__":
    print("--- Dynamic Hotspot Detection via Theil-Sen Slope Analysis (Quantile Method) ---")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"All outputs will be saved to: '{OUTPUT_DIR}/'")

    # Step 1 & 2: Load Data and Generate Slope Map
    mat_data = scipy.io.loadmat(MAT_FILE_PATH)
    video_frames = mat_data[MAT_VARIABLE_NAME].astype(np.float32)
    slope_map = generate_slope_map_parallel(video_frames)

    # Step 3: Save slope map visualization
    slope_map_visual = cv2.normalize(slope_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    slope_map_colored = cv2.applyColorMap(slope_map_visual, cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "slope_map.png"), slope_map_colored)

    # Step 4: Find hotspot regions using the new quantile function
    top_2_prompts = find_hotspot_regions_quantile(
        slope_map, 
        num_regions=2, 
        quantile=HOTSPOT_QUANTILE_THRESHOLD
    )
    all_peak_prompts = find_hotspot_regions_quantile(
        slope_map,
        num_regions=15,
        quantile=HOTSPOT_QUANTILE_THRESHOLD
    )

    if top_2_prompts:
        print(f"Top 2 distinct hotspot regions identified at: {[p['point'] for p in top_2_prompts]}")
    else:
        print("Could not identify any significant hotspots.")

    # Step 5, 6, 7: Prepare frame, draw maps, and run SAM
    final_frame_raw = video_frames[:, :, -1]
    final_frame_visual = cv2.normalize(final_frame_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    final_frame_rgb = cv2.applyColorMap(final_frame_visual, cv2.COLORMAP_INFERNO)

    activity_map_image = draw_activity_map(final_frame_rgb, all_peak_prompts, top_2_prompts)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "activity_map_final_frame.png"), activity_map_image)

    print("Loading SAM model...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    
    sam_masks = get_sam_masks(predictor, final_frame_rgb, top_2_prompts)
    
    if sam_masks:
        final_masked_image = draw_final_output(final_frame_rgb, sam_masks)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "final_masked_output.png"), final_masked_image)
    
    print("\n--- Processing Complete ---")