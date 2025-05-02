# --- START OF FILE feature_engineering.py ---

# feature_engineering.py
"""
Functions for extracting features using pre-computed hotspot masks.
"""

import numpy as np
import cv2
import os
import scipy
import config 
from scipy.stats import linregress
from tqdm import tqdm
from scipy import stats as sp_stats

from hotspot_mask_generation import apply_preprocessing

def calculate_weighted_slope_activity_map(frames_focus, fps, smooth_window,
                                           envir_para, errorweight, augment, roi_mask=None):
    # (Copy function from hotspot_mask_generation.py)
    H, W, T_focus = frames_focus.shape
    t_focus_vec = (np.arange(T_focus) / fps).astype(np.float64)
    final_activity_map = np.zeros((H, W), dtype=np.float64)
    if smooth_window > 1:
        smooth_len = len(np.convolve(np.ones(T_focus), np.ones(smooth_window), mode='valid'))
        if smooth_len < 2: print(f"ERROR: Smoothing window too large."); return None
        t_smoothed = t_focus_vec[:smooth_len]
    else: t_smoothed = t_focus_vec
    if roi_mask is not None:
        roi_indices = np.argwhere(roi_mask); iterator = tqdm(range(roi_indices.shape[0]), desc="Calculating Slope (ROI)", leave=False, ncols=80); get_coords = lambda idx: roi_indices[idx]
    else:
        total_pixels = H * W; iterator = tqdm(range(total_pixels), desc="Calculating Slope (Full)", leave=False, ncols=80); get_coords = lambda idx: np.unravel_index(idx, (H, W))
    for idx in iterator:
        r, c = get_coords(idx); y_raw = frames_focus[r, c, :]
        if smooth_window > 1:
            if len(y_raw) >= smooth_window: y = np.convolve(y_raw, np.ones(smooth_window)/smooth_window, mode='valid')
            else: y = y_raw; t_smoothed = t_focus_vec
        else: y = y_raw
        if len(y) < 2: continue
        try:
            if not np.all(np.isfinite(y)): continue
            current_t = t_smoothed[:len(y)]
            if len(current_t) < 2: continue
            res = sp_stats.linregress(current_t, y)
            if np.isfinite(res.slope) and np.isfinite(res.stderr):
                s = res.slope; se = res.stderr
                if s * envir_para < 0:
                    if abs(s) > 1e-9:
                        rel_err = abs(se / s); weight = np.exp(-rel_err * errorweight)
                        final_activity_map[r, c] = (abs(s) * weight)**augment
        except ValueError: pass
        except Exception as e_lr: print(f"\nWarning: linregress/weighting failed at ({r},{c}): {e_lr}"); pass
    return final_activity_map
# --- End of copied helper functions ---


def calculate_features_from_mask_and_activity(
    frames,
    hotspot_mask,
    activity_map, # Add activity map as input
    fps,
    focus_duration_frames
    ):
    """
    Calculates features based on temperature time series AND activity map statistics
    within the provided hotspot mask.

    Args:
        frames (np.ndarray): Full video frames (H, W, N), expected float64.
        hotspot_mask (np.ndarray): Boolean mask (H, W) for the hotspot.
        activity_map (np.ndarray): The weighted slope map (H, W) used to generate the mask.
        fps (float): Frames per second.
        focus_duration_frames (int): Number of initial frames to analyze for temp features.

    Returns:
        dict: Dictionary containing calculated features.
    """
    features = {
        'hotspot_area': np.nan,
        'hotspot_temp_change_rate': np.nan,      # Avg Temp Slope
        'hotspot_temp_change_magnitude': np.nan, # Avg Temp Change Magnitude
        # --- New Activity Features ---
        'activity_mean': np.nan,
        'activity_median': np.nan,
        'activity_std': np.nan,
        'activity_max': np.nan,
        'activity_sum': np.nan,
    }

    # Basic validation
    if frames is None or frames.ndim != 3: return features # Return NaNs
    if hotspot_mask is None or hotspot_mask.ndim != 2 or hotspot_mask.dtype != bool: return features
    if activity_map is None or activity_map.ndim != 2: return features
    if hotspot_mask.shape != frames.shape[:2] or activity_map.shape != frames.shape[:2]: return features

    # --- 1. Area ---
    features['hotspot_area'] = np.sum(hotspot_mask)
    if features['hotspot_area'] == 0:
        print("Warning: Hotspot mask has zero area.")
        return features # Return NaNs

    # --- 2. Temperature Time Series Features ---
    actual_focus_frames = min(focus_duration_frames, frames.shape[2])
    if actual_focus_frames >= 2:
        frames_focus = frames[:, :, :actual_focus_frames]
        hotspot_avg_temp_list = []
        for t in range(actual_focus_frames):
            masked_pixels = frames_focus[hotspot_mask, t]
            if masked_pixels.size > 0:
                avg_temp_t = np.mean(masked_pixels[np.isfinite(masked_pixels)])
                hotspot_avg_temp_list.append(avg_temp_t if np.isfinite(avg_temp_t) else np.nan)
            else: hotspot_avg_temp_list.append(np.nan)

        hotspot_avg_temp_series = np.array(hotspot_avg_temp_list)
        valid_temps_mask = np.isfinite(hotspot_avg_temp_series)
        num_valid_points = np.sum(valid_temps_mask)

        if num_valid_points >= 2:
            time_vector = (np.arange(actual_focus_frames) / fps).astype(np.float64)
            valid_times = time_vector[valid_temps_mask]
            valid_temps = hotspot_avg_temp_series[valid_temps_mask]

            # a) Avg Rate (Slope)
            try:
                slope, intercept, r_val, p_val, std_err = linregress(valid_times, valid_temps)
                if np.isfinite(slope): features['hotspot_temp_change_rate'] = slope
            except ValueError: pass # Keep NaN
            except Exception as e_lr: print(f"Warning: Error in temp rate linregress: {e_lr}")

            # b) Magnitude
            features['hotspot_temp_change_magnitude'] = valid_temps[-1] - valid_temps[0]
        else:
            print("Warning: Less than 2 valid avg temp points for rate/mag calc.")
    else:
        print("Warning: Less than 2 focus frames for rate/mag calc.")


    # --- 3. Activity Map Features ---
    try:
        # Extract activity values ONLY from within the mask
        activity_values_in_mask = activity_map[hotspot_mask]
        # Filter out potential NaNs/Infs from activity map itself
        valid_activity_values = activity_values_in_mask[np.isfinite(activity_values_in_mask)]

        if valid_activity_values.size > 0:
            features['activity_mean'] = np.mean(valid_activity_values)
            features['activity_median'] = np.median(valid_activity_values)
            features['activity_std'] = np.std(valid_activity_values)
            features['activity_max'] = np.max(valid_activity_values)
            features['activity_sum'] = np.sum(valid_activity_values)
        else:
            print("Warning: No valid finite activity values found within the hotspot mask.")

    except Exception as e:
        print(f"Error calculating activity map features: {e}")
        # Features remain NaN

    return features


# --- Main Feature Extraction Function (Modified) ---
def extract_features_with_mask(
    frames_or_path,
    mask_path,
    # Pass necessary parameters for recalculating activity map
    fps,
    focus_duration_sec,
    smooth_window,
    envir_para,
    errorweight,
    augment,
    fuselevel, # Added
    normalizeT, # Added
    roi_border_percent=None, # Added
    # Other args
    source_folder_name="Unknown",
    mat_filename_no_ext="Unknown"
    ):
    """
    Loads frames and mask, recalculates activity map, extracts features.
    """
    print(f"Extracting features for: {mat_filename_no_ext}.mat using mask: {os.path.basename(mask_path)}")
    print(f"  Recalculating activity map (Slope, Focus: {focus_duration_sec}s, Smooth: {smooth_window}, Fuse: {fuselevel}, NormT: {normalizeT}, ROI: {roi_border_percent is not None})")

    # --- 1. Load Mask ---
    if not os.path.exists(mask_path): print(f"Error: Mask file not found: {mask_path}"); return None
    try:
        hotspot_mask = np.load(mask_path); assert hotspot_mask.dtype == bool
    except Exception as e: print(f"Error loading mask file {mask_path}: {e}"); return None

    # --- 2. Load Frames ---
    if isinstance(frames_or_path, str):
        if not os.path.exists(frames_or_path): print(f"Error: Mat file not found: {frames_or_path}"); return None
        try:
            mat_data = scipy.io.loadmat(frames_or_path); frames_raw = mat_data.get(config.MAT_FRAMES_KEY, None)
            if frames_raw is None: raise KeyError(f"Key '{config.MAT_FRAMES_KEY}' not found.")
            frames_raw = frames_raw.astype(np.float64)
        except Exception as e: print(f"Error loading frames from {frames_or_path}: {e}"); return None
    elif isinstance(frames_or_path, np.ndarray): frames_raw = frames_or_path.astype(np.float64)
    else: print("Error: Invalid frames_or_path argument."); return None

    if frames_raw.ndim != 3 or frames_raw.shape[2] < 2: print("Error: Invalid frame dimensions."); return None
    if frames_raw.shape[:2] != hotspot_mask.shape: print("Error: Frame shape mismatch mask shape."); return None
    H, W, num_frames = frames_raw.shape

    # --- 3. Recalculate Activity Map (using same parameters as mask generation) ---
    # a) Pre-processing
    frames_proc = apply_preprocessing(frames_raw, normalizeT, fuselevel)
    # b) Focus Window
    focus_duration_frames = int(focus_duration_sec * fps)
    focus_duration_frames = max(2, min(focus_duration_frames, num_frames))
    frames_focus = frames_proc[:, :, :focus_duration_frames]
    # c) ROI Mask (optional)
    roi_mask = None
    if roi_border_percent is not None:
        border_h_roi = int(H * roi_border_percent); border_w_roi = int(W * roi_border_percent)
        roi_mask = np.zeros((H, W), dtype=bool); roi_mask[border_h_roi:-border_h_roi, border_w_roi:-border_w_roi] = True
    # d) Calculate Weighted Slope Map
    activity_map = calculate_weighted_slope_activity_map(
        frames_focus, fps, smooth_window, envir_para, errorweight, augment, roi_mask
    )
    if activity_map is None:
        print("  Error: Failed to recalculate activity map during feature extraction.")
        return None # Cannot calculate activity features

    # --- 4. Calculate All Features ---
    # Use focus_duration_frames consistent with activity map calculation
    extracted_features = calculate_features_from_mask_and_activity(
        frames_proc, # Use pre-processed frames for consistency if normalizeT/fuselevel applied
        hotspot_mask,
        activity_map,
        fps,
        focus_duration_frames
    )

    if extracted_features is None: print("  Feature calculation failed."); return None # Should not happen if area > 0

    print(f"  Finished extracting features. Keys: {list(extracted_features.keys())}")
    return extracted_features


# --- END OF FILE feature_engineering.py ---