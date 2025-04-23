# --- START OF FILE feature_engineering.py ---

# feature_engineering.py
"""Functions for extracting features based on the DYNAMIC hotspot region
identified from interval-based temporal gradient maps.
"""

import numpy as np
import cv2
import os
import config
# Import the updated visualization function
from visualization_util import save_dynamic_hotspot_visualizations

# --- Mean Frame Calculation (Still needed for visualization comparison) ---
def compute_mean_frame(frames):
    if frames is None or frames.ndim != 3 or frames.shape[2] == 0: return None
    try: return np.mean(frames, axis=2, dtype=np.float64)
    except Exception: return None

# --- NEW: Calculate Gradient Map for ONE Interval ---
def compute_interval_gradient_map(frames, start_frame, end_frame):
    """
    Calculates a 2D map of mean absolute temporal gradient per pixel
    ONLY for frames within the specified interval [start_frame, end_frame).
    """
    if frames is None or frames.ndim != 3: return None
    total_frames_available = frames.shape[2]
    actual_start = max(0, start_frame)
    actual_end = min(total_frames_available, end_frame)

    if actual_end - actual_start < 2: # Need at least 2 frames in interval for diff
        # print(f"Warning: Interval {start_frame}-{end_frame} has < 2 frames ({actual_end - actual_start}). Cannot calculate gradient map.")
        return None

    height, width = frames.shape[:2]
    gradient_sum = np.zeros((height, width), dtype=np.float64)
    valid_diff_count = 0

    # Iterate through frames WITHIN the interval
    for i in range(actual_start + 1, actual_end):
        try:
            diff = np.abs(frames[:, :, i].astype(np.float64) - frames[:, :, i-1].astype(np.float64))
            gradient_sum += diff
            valid_diff_count += 1
        except IndexError: # Should be prevented by loop bounds, but safety
             print(f"Index error accessing frame {i} or {i-1} in interval {start_frame}-{end_frame}")
             continue
        except Exception as e:
            print(f"Error calculating diff between frame {i}/{i-1} in interval: {e}")
            continue

    if valid_diff_count == 0:
        print(f"Warning: No valid frame differences in interval {start_frame}-{end_frame}.")
        return None

    mean_gradient_map = gradient_sum / valid_diff_count
    # print(f"  Interval {start_frame}-{end_frame} gradient map calculated.") # Optional Debug
    return mean_gradient_map

# --- RENAMED: Extract hotspot from a generic activity map ---
def extract_hotspot_from_map(activity_map, threshold_quantile=0.98):
    """
    Extracts the hotspot mask (region of highest activity) from a 2D map
    (e.g., a combined gradient map).
    """
    if activity_map is None or activity_map.size == 0: return None, np.nan
    proc_map = activity_map.copy(); nan_mask = np.isnan(proc_map)
    if np.all(nan_mask): return None, np.nan
    valid_pixels = proc_map[~nan_mask]
    if valid_pixels.size == 0: return np.zeros_like(activity_map, dtype=bool), 0.0

    map_std = np.std(valid_pixels); map_max = np.max(valid_pixels)
    if map_max < 1e-6 or map_std < 1e-6:
        print("Warning: Activity map has low variation. Hotspot detection may fail.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    try: threshold_value = np.percentile(valid_pixels, threshold_quantile * 100)
    except IndexError: threshold_value = map_max
    print(f"  Hotspot threshold ({threshold_quantile*100}th percentile on activity map): {threshold_value:.4f}")

    if threshold_value <= 1e-6 and map_max > 1e-6: threshold_value = 1e-6
    binary_mask = (np.nan_to_num(proc_map, nan=-np.inf) >= threshold_value).astype(np.uint8)
    if not np.any(binary_mask):
        print(f"Warning: Binary mask empty after thresholding activity map (quantile={threshold_quantile}).")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    kernel_size = max(min(3, activity_map.shape[0]//20, activity_map.shape[1]//20), 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    try: mask_clean = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    except cv2.error: mask_clean = binary_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    hotspot_mask = np.zeros_like(activity_map, dtype=bool); hotspot_area = 0.0
    if num_labels > 1:
        largest_label_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_label = largest_label_idx + 1
        hotspot_mask = (labels == largest_label)
        hotspot_area = np.sum(hotspot_mask)
        if hotspot_area == 0: print("Warning: Largest component mask (hotspot) empty.")
    # else: print("Warning: No connected components in thresholded activity map.") # Reduce noise

    return hotspot_mask, hotspot_area

# --- Temporal Feature Calculations (No change needed, just use correct mask) ---
def compute_temporal_gradient_in_interval(frames, hotspot_mask, start_frame, end_frame):
    # (Code remains the same as previous version)
    intensity_series = []
    if frames is None or frames.ndim != 3 or frames.shape[2] == 0: return np.nan
    if hotspot_mask is None or not np.any(hotspot_mask) or hotspot_mask.shape != frames.shape[:2]: return np.nan
    total_frames = frames.shape[2]
    actual_start = max(0, start_frame); actual_end = min(total_frames, end_frame)
    if actual_end - actual_start < 2: return np.nan
    for fidx in range(actual_start, actual_end):
        try:
            frame = frames[:, :, fidx]
            if frame is None or frame.shape != hotspot_mask.shape: continue
            masked_pixels = frame[hotspot_mask]
            if masked_pixels.size == 0: continue
            avg_intensity = np.mean(masked_pixels)
            if not np.isfinite(avg_intensity): continue
            intensity_series.append(avg_intensity)
        except Exception as e: print(f"Error processing frame {fidx} for hotspot gradient: {e}"); continue
    if len(intensity_series) < 2: return np.nan
    mean_abs_gradient = np.mean(np.abs(np.diff(np.array(intensity_series))))
    return mean_abs_gradient if np.isfinite(mean_abs_gradient) else np.nan

def compute_temporal_std_dev_in_interval(frames, hotspot_mask, start_frame, end_frame):
    # (Code remains the same as previous version)
    intensity_series = []
    if frames is None or frames.ndim != 3 or frames.shape[2] == 0: return np.nan
    if hotspot_mask is None or not np.any(hotspot_mask) or hotspot_mask.shape != frames.shape[:2]: return np.nan
    total_frames = frames.shape[2]
    actual_start = max(0, start_frame); actual_end = min(total_frames, end_frame)
    if actual_end - actual_start < 2: return np.nan
    for fidx in range(actual_start, actual_end):
        try:
            frame = frames[:, :, fidx]
            if frame is None or frame.shape != hotspot_mask.shape: continue
            masked_pixels = frame[hotspot_mask]
            if masked_pixels.size == 0: continue
            avg_intensity = np.mean(masked_pixels)
            if not np.isfinite(avg_intensity): continue
            intensity_series.append(avg_intensity)
        except Exception as e: print(f"Error processing frame {fidx} for hotspot std dev: {e}"); continue
    if len(intensity_series) < 2: return np.nan
    std_dev = np.std(np.array(intensity_series))
    return std_dev if np.isfinite(std_dev) else np.nan


# --- REVISED: Main extraction function following professor's workflow ---
def extract_bright_region_features(frames, source_folder_name, mat_filename_no_ext, visualize_index=-1):
    """
    Extracts features based on the DYNAMIC hotspot identified from interval-based
    temporal gradient maps.

    Workflow:
    1. Calculate gradient map for each time interval.
    2. Combine interval maps (e.g., by averaging) into a single activity map.
    3. Extract hotspot mask from the combined activity map.
    4. Calculate features (area, interval grad/std) using the hotspot mask on original frames.

    Args:
        frames (np.ndarray): Video frames (H, W, N).
        source_folder_name (str): Parent directory name of the mat file.
        mat_filename_no_ext (str): Mat file name without extension.
        visualize_index (int): Sample index for visualization control.

    Returns:
        dict: Dictionary of calculated features.
    """
    # --- Configuration ---
    interval_frames = getattr(config, 'FRAME_INTERVAL_SIZE', 50)
    max_intervals = getattr(config, 'MAX_FRAME_INTERVALS', 3)
    hotspot_quantile = getattr(config, 'GRADIENT_MAP_HOTSPOT_QUANTILE', 0.98) # Quantile for combined map
    calculate_std = getattr(config, 'CALCULATE_STD_FEATURES', False)
    save_vis = getattr(config, 'SAVE_FOCUS_AREA_VISUALIZATION', False)
    num_to_visualize = getattr(config, 'NUM_SAMPLES_TO_VISUALIZE', 0)
    base_vis_save_dir = getattr(config, 'FOCUS_AREA_VIS_SAVE_DIR', None)
    vis_orig_colormap = getattr(config, 'VISUALIZATION_COLORMAP', cv2.COLORMAP_HOT)
    vis_grad_colormap = getattr(config, 'VISUALIZATION_GRADIENT_MAP_COLORMAP', cv2.COLORMAP_INFERNO)


    print(f"Extracting DYNAMIC features for: Folder='{source_folder_name}', File='{mat_filename_no_ext}.mat'")
    features = {}; area_feat_name = "hotspot_area"; features[area_feat_name] = np.nan

    if frames is None or frames.ndim != 3 or frames.shape[2] < 2:
        print("Warning: Invalid/insufficient frames."); return features
    num_frames = frames.shape[2]

    # --- Step 1: Calculate Interval Gradient Maps ---
    interval_gradient_maps = []
    print("  Calculating interval gradient maps...")
    for i in range(max_intervals):
        start_frame = i * interval_frames
        # Ensure start frame is valid, break if we request intervals beyond available frames
        if start_frame >= num_frames -1: # Need at least 2 frames starting from start_frame
             print(f"  Skipping interval {i} ({start_frame}-...) as it starts too late.")
             break
        end_frame = start_frame + interval_frames
        # No need to cap end_frame here, compute_interval_gradient_map handles it
        interval_map = compute_interval_gradient_map(frames, start_frame, end_frame)
        if interval_map is not None:
            interval_gradient_maps.append(interval_map)
        else:
            print(f"  Warning: Failed to calculate gradient map for interval {i} ({start_frame}-{end_frame}).")
            # Optionally append None or just skip

    if not interval_gradient_maps:
        print("Error: Failed to calculate ANY interval gradient maps.")
        return features # Return dict with only area=NaN

    # --- Step 2: Combine Gradient Maps ---
    # Combine by averaging the valid interval maps
    print(f"  Combining {len(interval_gradient_maps)} interval gradient map(s)...")
    # Use np.stack then np.mean, handling potential NaNs if needed
    try:
         # Stack valid maps, mean along the new axis (axis=0)
         combined_activity_map = np.mean(np.stack(interval_gradient_maps, axis=0), axis=0)
    except ValueError as e: # Handles case where interval maps might have different shapes (shouldn't happen) or empty list
         print(f"Error combining interval maps: {e}. Cannot proceed.")
         return features

    # --- Step 3: Extract Hotspot from Combined Map ---
    print(f"  Extracting hotspot from combined activity map using quantile {hotspot_quantile}...")
    hotspot_mask, hotspot_area = extract_hotspot_from_map(combined_activity_map, threshold_quantile=hotspot_quantile)
    features[area_feat_name] = hotspot_area # Store the dynamic hotspot area

    if hotspot_mask is None or not np.any(hotspot_mask):
        print("Warning: Dynamic hotspot extraction failed or resulted in empty mask.")
        # Keep the area feature (might be 0), return before calculating temporal features
        return features

    # --- Optional Visualization ---
    if save_vis and visualize_index >= 0 and visualize_index < num_to_visualize and base_vis_save_dir:
        if frames.shape[2] > 0:
             mean_frame_for_vis = compute_mean_frame(frames)
             final_vis_dir = os.path.join(base_vis_save_dir, source_folder_name, mat_filename_no_ext)
             save_dynamic_hotspot_visualizations( # Call the updated vis function
                 original_frame=frames[:, :, 0],
                 mean_frame=mean_frame_for_vis,
                 interval_gradient_maps=interval_gradient_maps, # Pass the list
                 combined_gradient_map=combined_activity_map,   # Pass the combined map
                 hotspot_mask=hotspot_mask,                    # Pass the final mask
                 index=visualize_index,
                 save_dir=final_vis_dir,
                 orig_colormap=vis_orig_colormap,
                 grad_colormap=vis_grad_colormap
             )
        else: print("Warning: Cannot visualize, no frames available.")

    # --- Step 4: Extract Interval Features using Hotspot Mask ---
    print("  Extracting interval features within dynamic hotspot...")
    for i in range(max_intervals):
         start_frame = i * interval_frames
         if start_frame >= num_frames -1: break # Don't calculate for intervals that don't exist
         end_frame = start_frame + interval_frames
         # Use "hotspot_" prefix for clarity
         grad_feat_name = f"hotspot_grad_{start_frame}_{end_frame}"
         std_feat_name = f"hotspot_std_{start_frame}_{end_frame}"

         # Calculate gradient using the final hotspot mask
         mean_abs_grad = compute_temporal_gradient_in_interval(frames, hotspot_mask, start_frame, end_frame)
         features[grad_feat_name] = mean_abs_grad

         # Calculate std dev using the final hotspot mask (conditionally)
         if calculate_std:
             std_dev = compute_temporal_std_dev_in_interval(frames, hotspot_mask, start_frame, end_frame)
             features[std_feat_name] = std_dev

    print(f"  Finished extracting dynamic features. Keys: {list(features.keys())}")
    return features

# --- END OF FILE feature_engineering.py ---