# --- START OF FILE feature_engineering.py ---

# feature_engineering.py
"""
Functions for extracting features using pre-computed hotspot masks.
Calculates features based on average temp, peak pixel temp, temp distribution.
"""

import numpy as np
import cv2
import os
import scipy.io
import config
from scipy.stats import linregress
import warnings
# Imports related to activity map recalculation (can be kept if needed later, but commented out)
# from tqdm import tqdm
# from scipy import stats as sp_stats
# try:
#     from hotspot_mask_generation import apply_preprocessing, calculate_weighted_slope_activity_map
# except ImportError:
#     def apply_preprocessing(frames, normalizeT, fuselevel): return frames
#     def calculate_weighted_slope_activity_map(*args, **kwargs): return None


def calculate_hotspot_features( # Renamed back to be more general
    frames,
    hotspot_mask,
    fps,
    focus_duration_frames,
    envir_para=-1 # Needed for peak pixel determination
    ):
    """
    Calculates features based on Average Temp, PEAK pixel temp series,
    AND overall temp distribution stats within the hotspot mask.

    Args:
        frames (np.ndarray): Full video frames (H, W, N), expected float64.
        hotspot_mask (np.ndarray): Boolean mask (H, W) for the hotspot.
        fps (float): Frames per second.
        focus_duration_frames (int): Number of initial frames to analyze.
        envir_para (int): -1 for Winter (min temp peak), 1 for Summer (max temp peak).

    Returns:
        dict: Dictionary containing calculated features.
    """
    # Initialize ALL potential features
    features = {
        'hotspot_area': np.nan,
        # --- Avg Temp Features ---
        'hotspot_temp_change_rate': np.nan,
        'hotspot_temp_change_magnitude': np.nan,
        # --- Peak Pixel Features ---
        'peak_pixel_temp_change_rate': np.nan,
        'peak_pixel_temp_change_magnitude': np.nan,
        # --- Temp Distribution Features ---
        'temp_mean_avg': np.nan,
        'temp_std_avg': np.nan,
        'temp_min_overall': np.nan,
        'temp_max_overall': np.nan,
        # --- Activity features could be added back here if needed ---
    }

    # Basic validation
    if frames is None or frames.ndim != 3: print("Error: Invalid frames."); return features
    if hotspot_mask is None or hotspot_mask.ndim != 2 or hotspot_mask.dtype != bool: print("Error: Invalid mask."); return features
    if hotspot_mask.shape != frames.shape[:2]: print("Error: Mask shape mismatch."); return features

    # --- 1. Area ---
    features['hotspot_area'] = np.sum(hotspot_mask)
    if features['hotspot_area'] == 0: print("Warning: Mask area is 0."); return features

    # --- Get focus frames & valid pixels ---
    actual_focus_frames = min(focus_duration_frames, frames.shape[2])
    if actual_focus_frames < 2: print("Warning: Need >= 2 focus frames."); return features

    frames_focus = frames[:, :, :actual_focus_frames]
    time_vector = (np.arange(actual_focus_frames) / fps).astype(np.float64)

    # Extract valid pixel time series ONCE
    masked_pixels_ts = frames_focus[hotspot_mask, :]
    finite_pixel_mask_rows = np.all(np.isfinite(masked_pixels_ts), axis=1)
    valid_masked_pixels_ts = masked_pixels_ts[finite_pixel_mask_rows, :]
    num_valid_pixels = valid_masked_pixels_ts.shape[0]

    if num_valid_pixels == 0: print("Warning: No valid pixels in mask."); return features

    # --- 2. AVERAGE Temperature Time Series Features ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        hotspot_avg_temp_series = np.nanmean(valid_masked_pixels_ts, axis=0)

    valid_avg_temps_mask = np.isfinite(hotspot_avg_temp_series)
    num_valid_avg_points = np.sum(valid_avg_temps_mask)

    if num_valid_avg_points >= 2:
        valid_avg_times = time_vector[valid_avg_temps_mask]
        valid_avg_temps = hotspot_avg_temp_series[valid_avg_temps_mask]
        # a) Avg Rate (Slope)
        try:
            slope, intercept, r_val, p_val, std_err = linregress(valid_avg_times, valid_avg_temps)
            features['hotspot_temp_change_rate'] = slope if np.isfinite(slope) else np.nan
        except ValueError: pass
        except Exception as e_lr: print(f"Warning: Error in avg rate linregress: {e_lr}")
        # b) Avg Magnitude
        features['hotspot_temp_change_magnitude'] = valid_avg_temps[-1] - valid_avg_temps[0]
    # else: print("Warning: Less than 2 valid avg temp points.")

    # --- 3. PEAK Pixel Temp Time Series Features ---
    peak_pixel_temp_list = []
    peak_func = np.nanmin if envir_para < 0 else np.nanmax
    for t in range(actual_focus_frames):
        pixels_in_mask_t = frames_focus[hotspot_mask, t]
        valid_pixels_in_mask_t = pixels_in_mask_t[np.isfinite(pixels_in_mask_t)]
        if valid_pixels_in_mask_t.size > 0: peak_pixel_temp_list.append(peak_func(valid_pixels_in_mask_t))
        else: peak_pixel_temp_list.append(np.nan)

    peak_pixel_temp_series = np.array(peak_pixel_temp_list)
    valid_peak_temps_mask = np.isfinite(peak_pixel_temp_series)
    num_valid_peak_points = np.sum(valid_peak_temps_mask)

    if num_valid_peak_points >= 2:
        valid_peak_times = time_vector[valid_peak_temps_mask]
        valid_peak_temps = peak_pixel_temp_series[valid_peak_temps_mask]
        # a) Peak Rate (Slope)
        try:
            slope_peak, _, _, _, _ = linregress(valid_peak_times, valid_peak_temps)
            features['peak_pixel_temp_change_rate'] = slope_peak if np.isfinite(slope_peak) else np.nan
        except ValueError: pass
        except Exception as e_lr: print(f"Warning: Error during peak rate linregress: {e_lr}")
        # b) Peak Magnitude
        features['peak_pixel_temp_change_magnitude'] = valid_peak_temps[-1] - valid_peak_temps[0]
    # else: print("Warning: Less than 2 valid peak temp points.")

    # --- 4. Temperature Distribution Features ---
    try:
        # Use valid_masked_pixels_ts calculated earlier
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Use hotspot_avg_temp_series calculated in step 2 for mean_avg
            features['temp_mean_avg'] = np.nanmean(hotspot_avg_temp_series[valid_avg_temps_mask]) if num_valid_avg_points > 0 else np.nan
            # Avg Stdev
            std_temps_per_frame = np.nanstd(valid_masked_pixels_ts, axis=0)
            features['temp_std_avg'] = np.nanmean(std_temps_per_frame[np.isfinite(std_temps_per_frame)]) if np.any(np.isfinite(std_temps_per_frame)) else np.nan

        # Overall Min/Max across all valid pixels and time
        features['temp_min_overall'] = np.min(valid_masked_pixels_ts)
        features['temp_max_overall'] = np.max(valid_masked_pixels_ts)
    except Exception as e:
        print(f"Error calculating temperature distribution features: {e}")

    return features


# --- Main Feature Extraction Function ---
def extract_features_with_mask(
    frames_or_path,
    mask_path,
    # Pass only parameters needed by the calculation function
    fps,
    focus_duration_sec,
    envir_para, # Still needed for peak pixel
    # Optional args for logging
    source_folder_name="Unknown",
    mat_filename_no_ext="Unknown"
    ):
    """Loads frames and mask, extracts temp/peak/dist features."""
    # print(f"Extracting features for: {mat_filename_no_ext}.mat") # Less verbose

    # --- Load Mask & Frames ---
    # ... (keep loading logic same as before) ...
    if not os.path.exists(mask_path): print(f"Error: Mask file not found: {mask_path}"); return None
    try: hotspot_mask = np.load(mask_path); assert hotspot_mask.dtype == bool
    except Exception as e: print(f"Error loading mask file {mask_path}: {e}"); return None
    if isinstance(frames_or_path, str):
        if not os.path.exists(frames_or_path): print(f"Error: Mat file not found: {frames_or_path}"); return None
        try: mat_data = scipy.io.loadmat(frames_or_path); frames = mat_data.get(config.MAT_FRAMES_KEY, None); assert frames is not None; frames = frames.astype(np.float64)
        except Exception as e: print(f"Error loading frames from {frames_or_path}: {e}"); return None
    elif isinstance(frames_or_path, np.ndarray): frames = frames_or_path.astype(np.float64)
    else: print("Error: Invalid frames_or_path."); return None
    if frames.ndim!=3 or frames.shape[2]<2: print("Error: Invalid frames."); return None
    if frames.shape[:2]!=hotspot_mask.shape: print("Error: Shape mismatch."); return None

    # --- Calculate Focus Duration & Features ---
    focus_duration_frames = int(focus_duration_sec * fps)
    focus_duration_frames = max(2, min(focus_duration_frames, frames.shape[2]))

    # --- Call the unified stats function ---
    extracted_features = calculate_hotspot_features( # Use the general function
        frames,
        hotspot_mask,
        fps,
        focus_duration_frames,
        envir_para=envir_para
    )

    if extracted_features is None: print("Feature calculation failed."); return None

    # print(f"  Finished extracting features.") # Less verbose
    return extracted_features

# --- END OF FILE feature_engineering.py ---