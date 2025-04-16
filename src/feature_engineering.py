# feature_engineering.py
"""Functions for extracting features based on the brightest region.

Features Extracted:
1.  Temporal Gradients: Mean absolute change in average bright region intensity 
    between consecutive frames, calculated over successive FRAME-BASED intervals.
2.  Bright Region Area: Total area (pixel count) of the consistently bright 
    region identified from the mean frame across the entire video.
"""

import numpy as np
import cv2
import config # Needed for BRIGHT_REGION_THRESHOLD, FRAME_INTERVAL_SIZE

def compute_mean_frame(frames):
    """Computes the pixel-wise mean frame across all video frames.
    Assumes frames are stacked along the 3rd axis (axis=2).
    """
    if frames is None:
        print("Warning: frames is None in compute_mean_frame.")
        return None
    if not isinstance(frames, np.ndarray) or frames.size == 0:
        print("Warning: Invalid or empty frames array provided to compute_mean_frame.")
        return None
    if frames.ndim != 3:
        print(f"Warning: compute_mean_frame expects 3D array (H, W, N), got {frames.ndim}D. Shape: {frames.shape}")
        if frames.ndim == 2:
            # print("Info: Input was 2D, returning it as is (assuming single frame).")
            return frames
        else:
            return None

    num_frames = frames.shape[2]
    if num_frames == 0:
        print("Warning: frames array has zero frames along axis 2.")
        return None
    
    # print(f"compute_mean_frame: original frames shape: {frames.shape}") # DEBUG

    try:
        mean_frame = np.mean(frames, axis=2, dtype=np.float64)
        # print(f"compute_mean_frame: computed mean frame shape: {mean_frame.shape}") # DEBUG
        return mean_frame
    except Exception as e:
        print(f"Error computing mean frame: {e}") # Keep Error
        return None

def extract_consistent_region(mean_frame, threshold_quantile=0.95): # Use quantile parameter
    """
    Extracts the consistently bright region from the mean frame using PERCENTILE 
    thresholding and morphology.
    Returns the mask of the largest connected component AND its area in pixels.
    """
    if mean_frame is None or mean_frame.size == 0 or mean_frame.ndim != 2:
        print("Warning: Mean frame is invalid, None, or not 2D.")
        return None, np.nan

    proc_frame = mean_frame.copy()

    if np.isnan(proc_frame).any():
        print("Warning: NaNs detected in mean_frame. Replacing with min value.")
        min_val = np.nanmin(proc_frame)
        if np.isnan(min_val):
             print("Error: Mean frame consists entirely of NaNs.")
             return None, np.nan
        proc_frame = np.nan_to_num(proc_frame, nan=min_val)

    if proc_frame.size > 0:
         try:
             # Calculate threshold based on percentile of NON-NaN values
             threshold_value = np.percentile(proc_frame, threshold_quantile * 100) 
             print(f"  Calculated threshold ({threshold_quantile*100}th percentile): {threshold_value:.4f}")
         except IndexError: # Handles empty array case, though size check should prevent it
             print("Warning: Could not calculate percentile (empty array after NaN handling?). Setting threshold to max.")
             threshold_value = np.max(proc_frame) # Fallback, likely results in empty mask
    else:
        print("Warning: Frame has size 0 after NaN check.")
        return np.zeros_like(mean_frame, dtype=bool), 0.0

    # Handle case where threshold is calculated as non-positive
    if threshold_value <= 1e-6:
         print(f"Warning: Calculated percentile threshold ({threshold_value:.4f}) is near zero. Adjust quantile or check data. Using small positive epsilon.")
         # Use a very small positive value if threshold is zero or negative
         # Or consider falling back to Otsu's method below if percentile consistently fails
         threshold_value = 1e-6 


    binary_mask = (proc_frame >= threshold_value).astype(np.uint8)
    
    # Check if initial mask is empty - percentile might be too high
    if not np.any(binary_mask):
        print(f"Warning: Binary mask is empty after percentile thresholding ({threshold_quantile=}). Returning empty mask/zero area.")
        return np.zeros_like(mean_frame, dtype=bool), 0.0

    kernel_size = min(5, mean_frame.shape[0] // 10, mean_frame.shape[1] // 10)
    kernel_size = max(kernel_size, 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    try:
        mask_clean = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    except cv2.error as e:
         print(f"Warning: OpenCV morphology error: {e}. Using binary mask directly.")
         mask_clean = binary_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

    region_mask = np.zeros_like(mean_frame, dtype=bool)
    region_area = 0.0

    if num_labels <= 1:
        print("Warning: No connected components found after morphology.")
    else:
        largest_label_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_label = largest_label_idx + 1
        region_mask = (labels == largest_label)
        if np.any(region_mask):
             region_area = np.sum(region_mask)
             # print(f"extract_consistent_region: Found region mask with {region_area} pixels.") # DEBUG
        else:
             print("Warning: Largest component mask is empty.")
             region_area = 0.0

    return region_mask, region_area

def compute_temporal_gradient_in_interval(frames, region_mask, start_frame, end_frame):
    """
    Computes mean absolute temporal gradient of average intensity within the 
    specified region mask for frames in the [start_frame, end_frame) interval.
    
    Args:
        frames (np.ndarray): The video frames (H, W, N).
        region_mask (np.ndarray): Boolean mask (H, W) for the region of interest.
        start_frame (int): The starting frame index (inclusive).
        end_frame (int): The ending frame index (exclusive).

    Returns:
        float: mean_abs_gradient or np.nan if calculation fails.
    """
    if frames is None or frames.ndim != 3 or frames.shape[2] == 0:
        print(f"Warning: Invalid frames provided for gradient interval {start_frame}-{end_frame}.") # Keep Warning
        return np.nan
    if region_mask is None or not np.any(region_mask) or region_mask.shape != frames.shape[:2]:
        print(f"Warning: Invalid or mismatched region mask for gradient interval {start_frame}-{end_frame}.") # Keep Warning
        return np.nan

    total_frames_available = frames.shape[2]
    
    actual_start = max(0, start_frame)
    actual_end = min(total_frames_available, end_frame)

    # Need at least 2 frames to compute differences
    if actual_end - actual_start < 2:
        print(f"Warning: Not enough frames ({actual_end - actual_start}) in adjusted interval [{actual_start}, {actual_end}) "
              f"for requested interval {start_frame}-{end_frame} to compute gradient.") # Keep Warning
        return np.nan

    # print(f"compute_temporal_gradient: Processing frames {actual_start} to {actual_end-1} for interval {start_frame}-{end_frame}.") # DEBUG
    
    intensity_series = []
    for fidx in range(actual_start, actual_end):
        try:
            frame = frames[:, :, fidx]
            if frame is None or frame.shape != region_mask.shape:
                 print(f"Warning: Frame {fidx} has unexpected shape or is None. Skipping.") # Keep Warning
                 continue
                 
            masked_pixels = frame[region_mask]
            if masked_pixels.size == 0:
                print(f"Warning: No pixels selected by mask for frame {fidx}. Skipping.") # Keep Warning
                continue
            
            avg_intensity = np.mean(masked_pixels)
            if not np.isfinite(avg_intensity):
                 print(f"Warning: Non-finite average intensity calculated for frame {fidx}. Skipping.") # Keep Warning
                 continue

            intensity_series.append(avg_intensity)
            
        except IndexError:
             print(f"Error: IndexError accessing frame {fidx}. Total frames: {total_frames_available}") # Keep Error
             continue
        except Exception as e:
             print(f"Error processing frame {fidx}: {type(e).__name__} - {e}") # Keep Error
             continue

    if len(intensity_series) < 2:
        print(f"Warning: Not enough valid intensity values ({len(intensity_series)}) collected in interval {start_frame}-{end_frame} for gradient.") # Keep Warning
        return np.nan

    intensity_series = np.array(intensity_series)
    
    gradient = np.diff(intensity_series)
    mean_abs_gradient = np.mean(np.abs(gradient))
    
    mean_abs_gradient = mean_abs_gradient if np.isfinite(mean_abs_gradient) else np.nan

    # To caculate Standard deviation of avg intensites
    std_dev = np.std(intensity_series)
    std_dev = std_dev if np.isfinite(std_dev) else np.nan
    
    # print(f"  Gradient for {start_frame}-{end_frame}: {mean_abs_gradient:.4f}") # DEBUG

    return mean_abs_gradient

# --- MODIFIED: Extracts Area and Interval Gradients ---
def extract_bright_region_features(frames):
    """
    Extracts features from the consistently brightest region:
    1. Area: Total pixel count of the bright region identified from the mean frame.
    2. Interval Gradients: Mean absolute temporal gradient over successive FRAME-BASED intervals.
    
    Assumes frames are (Height, Width, NumFrames).

    Returns a dictionary of features named like:
      {
        "bright_region_area": value, 
        "bright_region_grad_0_50": value, 
        "bright_region_grad_50_100": value,
        ...
      }
    Returns an empty dictionary or dictionary with NaNs if critical steps fail.
    """
    # --- Configuration ---
    interval_frames = getattr(config, 'FRAME_INTERVAL_SIZE', 50)
    #threshold_factor = getattr(config, 'BRIGHT_REGION_THRESHOLD', 0.8)
    threshold_quantile = getattr(config, 'BRIGHT_REGION_QUANTILE', 0.95) # Add this to config or use default 0.95
    calculate_std = getattr(config, 'CALCULATE_STD_FEATURES', False)
    
    # --- Initialize features dictionary ---
    features = {} 
    # Define the area feature key explicitly
    area_feat_name = "bright_region_area"
    features[area_feat_name] = np.nan 

    # --- Input Validation ---
    if frames is None or not isinstance(frames, np.ndarray) or frames.ndim != 3 or frames.shape[2] == 0:
        print(f"Warning: Invalid or empty frames array passed to extract_bright_region_features. Shape: {getattr(frames, 'shape', 'None')}") # Keep Warning
        return features # Return dict with Area=NaN

    num_frames = frames.shape[2]
    # print(f"extract_bright_region_features: received frames with shape: {frames.shape} ({num_frames} frames). Interval size: {interval_frames} frames.") # DEBUG

    # --- Compute Mean and Mask & Area ---
    mean_frame = compute_mean_frame(frames)
    if mean_frame is None:
        print("Warning: Mean frame could not be computed.")
        return features 


    region_mask, region_area = extract_consistent_region(mean_frame, threshold_quantile=threshold_quantile) 
    
    features[area_feat_name] = region_area 

    if region_mask is None or not np.any(region_mask):
        print("Warning: Region extraction failed or resulted in empty mask.")
        return features

    # --- Iterate through Frame Intervals for Gradients ---
    start_frame = 0
    while start_frame < num_frames:
        end_frame = start_frame + interval_frames
        
        # Define feature name for this interval's gradient
        grad_feat_name = f"bright_region_grad_{start_frame}_{end_frame}"
        
        # Compute ONLY the gradient for the interval [start_frame, end_frame)
        mean_abs_grad = compute_temporal_gradient_in_interval(
            frames, region_mask, start_frame, end_frame
        )
        
        # Store the gradient result (even if it is NaN)
        features[grad_feat_name] = mean_abs_grad
        
        # Standard deviation caluclation (optional and configurable via flag)
        if calculate_std:
            std_feat_name = f"bright_region_std_{start_frame}_{end_frame}"
            std_dev = compute_temporal_gradient_in_interval(
                frames, region_mask, start_frame, end_frame
            )
            features[std_feat_name] = std_dev
        
        # Move to the next interval
        start_frame += interval_frames

    if len(features) == 1 and area_feat_name in features: # Only area was calculated
         print("Warning: Only area feature was calculated. No gradient intervals were processed (e.g., num_frames < 2).") # Keep Warning
         
    return features