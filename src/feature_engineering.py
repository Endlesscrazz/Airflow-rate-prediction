# feature_engineering.py
"""Functions for extracting features based on the brightest region's temporal gradient over multiple intervals."""

import numpy as np
import warnings
import cv2  # Import OpenCV
import config

# --- compute_mean_frame (keep as is) ---
def compute_mean_frame(frames):
    # ... (no changes needed) ...
    if frames is None or len(frames) == 0: return None
    try: return np.mean(frames, axis=0, dtype=np.float32)
    except Exception as e: print(f"Error computing mean frame: {e}"); return None

# --- extract_consistent_region (keep as is) ---
def extract_consistent_region(mean_frame, threshold_factor=0.8):
    # ... (no changes needed) ...
    if mean_frame is None or mean_frame.size == 0: return None
    max_intensity = np.max(mean_frame)
    if max_intensity <= 0: return None
    thresh_value = threshold_factor * max_intensity
    binary_mask_uint8 = (mean_frame >= thresh_value).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(binary_mask_uint8, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    if num_labels <= 1: return None
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    region_mask = (labels == largest_label)
    if not np.any(region_mask): return None
    return region_mask

# --- compute_temporal_gradient_in_window (keep as is) ---
def compute_temporal_gradient_in_window(frames, region_mask, start_sec, end_sec, fps=30):
    # ... (no changes needed) ...
    if frames is None or frames.shape[0] == 0: return np.nan
    if region_mask is None or not np.any(region_mask): return np.nan
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    start_frame = min(start_frame, frames.shape[0])
    end_frame = min(end_frame, frames.shape[0])
    if end_frame - start_frame < 2: return 0.0 # Return 0 if not enough frames in window
    intensity_series = []
    for fidx in range(start_frame, end_frame):
        frame = frames[fidx]
        avg_intensity = np.mean(frame[region_mask])
        intensity_series.append(avg_intensity)
    intensity_series = np.array(intensity_series)
    valid_intensities = intensity_series[~np.isnan(intensity_series)]
    if len(valid_intensities) < 2: return 0.0
    gradient = np.diff(valid_intensities)
    mean_abs_gradient = np.mean(np.abs(gradient))
    return mean_abs_gradient if np.isfinite(mean_abs_gradient) else 0.0

# --- EDITED extract_bright_region_features ---
def extract_bright_region_features(frames):
    """
    Extracts multiple temporal gradient features from the consistently brightest region
    in fixed-second intervals (e.g., 5 seconds). Stops when intervals exceed video length.

    Args:
        frames (np.ndarray): Video frames with shape (num_frames, height, width).

    Returns:
        dict: A dictionary of features, e.g.:
          {
            "bright_region_grad_0_5": <float>,
            "bright_region_grad_5_10": <float>,
            ... (up to max_interval_secs or video length)
          }
          Returns an empty dict if basic processing fails.
    """
    features = {} # Initialize as empty dict

    # Get parameters from config or defaults
    threshold_factor = getattr(config, 'BRIGHT_REGION_THRESHOLD', 0.8)
    fps = getattr(config, 'VIDEO_FPS', 30)
    interval_duration_sec = getattr(config, 'GRADIENT_INTERVAL_SEC', 5) # Duration of each window
    # Define maximum number of intervals to calculate (prevents excessive features for long videos)
    max_intervals = getattr(config, 'MAX_GRADIENT_INTERVALS', 3) # e.g., calculate for 0-5, 5-10, 10-15s

    # Basic checks
    if frames is None or frames.shape[0] < 2: # Need at least 2 frames overall
        print("Warning: Not enough frames (< 2) to calculate any temporal gradient.")
        return features

    # 1) Compute mean frame & region mask
    mean_frame = compute_mean_frame(frames)
    if mean_frame is None:
        print("Warning: Could not compute mean frame, returning empty features.")
        return features

    region_mask = extract_consistent_region(mean_frame, threshold_factor=threshold_factor)
    if region_mask is None:
        print("Warning: Could not extract region mask, returning empty features.")
        return features

    # 2) Iterate over each time window
    video_duration_sec = frames.shape[0] / fps
    num_calculated_intervals = 0

    for i in range(max_intervals):
        start_sec = i * interval_duration_sec
        end_sec = (i + 1) * interval_duration_sec

        # ***EDIT: Stop if the start of the interval is already beyond the video length***
        if start_sec >= video_duration_sec:
            # No need to calculate further intervals
            print(f"  Info: Stopping interval calculation at {start_sec}s (video duration {video_duration_sec:.2f}s)")
            break

        # Feature name for this window, e.g. bright_region_grad_0_5, bright_region_grad_5_10
        feat_name = f"bright_region_grad_{int(start_sec)}_{int(end_sec)}"

        print(f"  Calculating gradient for interval: {start_sec:.1f}s - {end_sec:.1f}s ({feat_name})")
        mean_abs_grad = compute_temporal_gradient_in_window(
            frames, region_mask, start_sec, end_sec, fps=fps
        )
        features[feat_name] = mean_abs_grad
        num_calculated_intervals += 1

    if num_calculated_intervals == 0:
        print("Warning: No gradient intervals were successfully calculated.")

    # Note: Features for intervals beyond video length (but within max_intervals)
    # that were skipped by the 'break' will simply be absent from the dictionary.
    # Features for intervals *partially* overlapping will have been calculated based on available frames.

    return features