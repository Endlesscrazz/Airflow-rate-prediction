# feature_engineering.py
"""Functions for extracting features based on the brightest region's temporal gradient."""

import numpy as np
import warnings
import cv2  # Import OpenCV
import config # Import config to potentially get parameters

# --- Helper Functions (from the provided script) ---

def compute_mean_frame(frames):
    """Computes the pixel-wise mean frame across all video frames."""
    if frames is None or len(frames) == 0:
        return None
    try:
        return np.mean(frames, axis=0, dtype=np.float32) # Use float32 for precision
    except Exception as e:
        print(f"Error computing mean frame: {e}")
        return None

def extract_consistent_region(mean_frame, threshold_factor=0.8):
    """
    Extracts the consistently bright region from the mean frame using thresholding and morphology.
    Returns the mask of the largest bright connected component.
    """
    if mean_frame is None or mean_frame.size == 0:
        print("Warning: Mean frame is empty or None in extract_consistent_region.")
        return None, None

    max_intensity = np.max(mean_frame)
    if max_intensity <= 0: # Handle cases where the frame is all dark or zero
        print("Warning: Max intensity of mean frame is <= 0. Cannot find bright region.")
        return None, None

    thresh_value = threshold_factor * max_intensity
    # Ensure frame is 8-bit for OpenCV binary operations if needed, but comparison works on float
    # Convert boolean mask to uint8 for morphology and connected components
    binary_mask_uint8 = (mean_frame >= thresh_value).astype(np.uint8)

    # Define kernel for morphological closing
    kernel = np.ones((5, 5), np.uint8) # 5x5 kernel, adjust if needed
    # Close operation: Dilate then Erode - fills small holes, connects nearby objects
    mask_clean = cv2.morphologyEx(binary_mask_uint8, cv2.MORPH_CLOSE, kernel)

    # Find connected components in the cleaned mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

    if num_labels <= 1: # Only background found
        print("Warning: No connected components found above threshold after cleaning.")
        return None, None

    # Find the label of the largest component (excluding background label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) # Add 1 because we slice stats from index 1

    # Create mask for the largest component
    region_mask = (labels == largest_label)

    # Bbox (optional, not used later but good practice)
    # x = stats[largest_label, cv2.CC_STAT_LEFT]
    # y = stats[largest_label, cv2.CC_STAT_TOP]
    # w = stats[largest_label, cv2.CC_STAT_WIDTH]
    # h = stats[largest_label, cv2.CC_STAT_HEIGHT]
    # bbox = (x, y, w, h)

    if not np.any(region_mask): # Check if the largest component mask is somehow empty
        print("Warning: Largest component mask is empty.")
        return None, None

    return region_mask # Return only the mask

def compute_temporal_gradient_region(frames, region_mask, interval_sec=5, fps=30):
    """
    Computes the mean absolute temporal gradient of the average intensity
    within the specified region over a time interval.
    """
    if frames is None or len(frames) == 0:
        print("Warning: No frames provided for temporal gradient calculation.")
        return np.nan
    if region_mask is None or not np.any(region_mask): # Check if mask is valid and not empty
        print("Warning: Invalid or empty region mask provided for temporal gradient calculation.")
        return np.nan

    num_interval_frames = int(interval_sec * fps)
    num_interval_frames = min(num_interval_frames, frames.shape[0]) # Don't exceed available frames

    if num_interval_frames <= 1: # Need at least 2 frames to calculate difference
        print(f"Warning: Not enough frames ({num_interval_frames}) in interval to compute temporal gradient.")
        return 0.0 # Or np.nan? 0.0 implies no change if only one frame.

    intensity_series = []
    for i in range(num_interval_frames):
        frame = frames[i]
        # Calculate mean intensity ONLY within the masked region
        # Check if mask is valid for frame shape (should be)
        if region_mask.shape == frame.shape:
             masked_pixels = frame[region_mask]
             if masked_pixels.size > 0: # Ensure there are pixels in the mask
                 avg_intensity = np.mean(masked_pixels)
                 intensity_series.append(avg_intensity)
             else:
                 print(f"Warning: Frame {i} has no pixels within the region mask. Appending NaN.")
                 intensity_series.append(np.nan) # Append NaN if mask somehow doesn't overlap
        else:
             print(f"Warning: Frame {i} shape {frame.shape} differs from mask shape {region_mask.shape}. Skipping frame.")
             intensity_series.append(np.nan) # Append NaN if shapes mismatch

    intensity_series = np.array(intensity_series)

    # Remove NaNs before calculating difference, if any occurred
    valid_intensities = intensity_series[~np.isnan(intensity_series)]

    if len(valid_intensities) <= 1: # Need at least 2 valid points for diff
        print(f"Warning: Not enough valid intensity values ({len(valid_intensities)}) to compute temporal gradient.")
        return 0.0 # Or np.nan

    # Calculate difference between consecutive valid average intensities
    gradient = np.diff(valid_intensities)
    mean_abs_gradient = np.mean(np.abs(gradient)) # Already handles empty gradient array case (returns nan)

    # Return 0.0 if nan to allow imputation later, consistent with no change.
    return mean_abs_gradient if np.isfinite(mean_abs_gradient) else 0.0


# --- Main Feature Extraction Function to be called from main.py ---

def extract_bright_region_features(frames):
    """
    Extracts features based on the consistently brightest region.
    Uses parameters from config.py or defaults.

    Args:
        frames (np.ndarray): Video frames (num_frames, height, width).

    Returns:
        dict: Dictionary containing 'bright_region_temp_grad'.
              Value is NaN if calculation fails.
    """
    feature_name = 'bright_region_temp_grad'
    features = {feature_name: np.nan} # Initialize with NaN

    # Get parameters (use defaults or from config if defined)
    threshold_factor = getattr(config, 'BRIGHT_REGION_THRESHOLD', 0.8)
    fps = getattr(config, 'VIDEO_FPS', 30)
    interval_sec = getattr(config, 'GRADIENT_INTERVAL_SEC', 5)

    if frames is None or len(frames) == 0:
        print("Warning: Empty frames array passed to extract_bright_region_features.")
        return features

    try:
        mean_frame = compute_mean_frame(frames)
        if mean_frame is None: return features # Return NaN if mean frame failed

        region_mask = extract_consistent_region(mean_frame, threshold_factor=threshold_factor)
        if region_mask is None: return features # Return NaN if region extraction failed

        temporal_gradient = compute_temporal_gradient_region(frames, region_mask,
                                                              interval_sec=interval_sec, fps=fps)

        features[feature_name] = temporal_gradient # Store the calculated gradient

    except Exception as e:
        print(f"Error during bright region feature extraction: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        # features[feature_name] remains NaN

    # Final check just in case
    if not np.isfinite(features[feature_name]):
        print(f"    Final value for {feature_name} is non-finite. Ensuring NaN.")
        features[feature_name] = np.nan

    return features