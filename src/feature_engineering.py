# feature_engineering.py
"""
Functions for extracting features using pre-computed hotspot masks.
Includes calculation of features based on average temp, peak pixel temp,
temp distribution, TEMP DIFFERENCE frames using FIXED ABSOLUTE thresholding for area metrics.
"""

import numpy as np
# import cv2 
import os
import scipy.io
import config  
from scipy.stats import linregress
import warnings
import traceback


def calculate_hotspot_features(
    frames,
    hotspot_mask,
    fps,
    focus_duration_frames,  # Used for initial rate/magnitude features
    envir_para=-1,
    threshold_abs_change=0.5  # BACK TO FIXED: Absolute threshold value
):
    """
    Calculates features using FIXED ABSOLUTE thresholding for area metrics.

    Args:
        frames (np.ndarray): Full video frames (H, W, N_total), float64.
        hotspot_mask (np.ndarray): Boolean mask (H, W).
        fps (float): Frames per second.
        focus_duration_frames (int): Frames for initial rate/magnitude analysis.
        envir_para (int): -1 for Winter (min temp), 1 for Summer (max temp).
        threshold_abs_change (float): Fixed absolute temperature change threshold (°C)
                                      used for significant area calculation.

    Returns:
        dict: Dictionary containing calculated features.
    """
    features = {
        'hotspot_area': np.nan,
        # Initial phase features
        'hotspot_avg_temp_change_rate_initial': np.nan,
        'hotspot_avg_temp_change_magnitude_initial': np.nan,
        'peak_pixel_temp_change_rate_initial': np.nan,
        'peak_pixel_temp_change_magnitude_initial': np.nan,
        'temp_mean_avg_initial': np.nan,
        'temp_std_avg_initial': np.nan,
        'temp_min_overall_initial': np.nan,
        'temp_max_overall_initial': np.nan,
        # Difference frame & stabilized features (Full duration)
        'stabilized_mean_deltaT': np.nan,
        'overall_mean_deltaT': np.nan,
        'max_abs_mean_deltaT': np.nan,
        'stabilized_std_deltaT': np.nan,
        'overall_std_deltaT': np.nan,
        # Area of Significant Change Features (Full duration)
        'mean_area_significant_change': np.nan,
        'stabilized_area_significant_change': np.nan,
        'max_area_significant_change': np.nan,
    }

    # Basic validation
    # ... (keep validation logic same as before) ...
    if frames is None or frames.ndim != 3:
        print("Error: Invalid frames.")
        return features
    if hotspot_mask is None or hotspot_mask.ndim != 2 or hotspot_mask.dtype != bool:
        print("Error: Invalid mask.")
        return features
    if hotspot_mask.shape != frames.shape[:2]:
        print("Error: Mask shape mismatch.")
        return features
    H, W, N_total = frames.shape
    if N_total < 2:
        print("Warning: Need >= 2 total frames.")
        return features

    # --- 1. Area ---
    features['hotspot_area'] = np.sum(hotspot_mask)
    if features['hotspot_area'] == 0:
        print("Warning: Mask area is 0.")
        return features

    # --- Calculate Initial Features (Focus Duration) ---
    actual_focus_frames = min(focus_duration_frames, N_total)
    if actual_focus_frames >= 2:
        frames_focus = frames[:, :, :actual_focus_frames]
        time_vector_focus = (
            np.arange(actual_focus_frames) / fps).astype(np.float64)
        masked_pixels_ts_focus = frames_focus[hotspot_mask, :]
        finite_pixel_mask_rows_focus = np.all(
            np.isfinite(masked_pixels_ts_focus), axis=1)
        valid_masked_pixels_ts_focus = masked_pixels_ts_focus[finite_pixel_mask_rows_focus, :]
        num_valid_pixels_focus = valid_masked_pixels_ts_focus.shape[0]
        if num_valid_pixels_focus > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
            hotspot_avg_temp_series_focus = np.nanmean(
                valid_masked_pixels_ts_focus, axis=0)
            valid_avg_temps_mask_focus = np.isfinite(
                hotspot_avg_temp_series_focus)
            if np.sum(valid_avg_temps_mask_focus) >= 2:
                valid_avg_times_f = time_vector_focus[valid_avg_temps_mask_focus]
                valid_avg_temps_f = hotspot_avg_temp_series_focus[valid_avg_temps_mask_focus]
                try:
                    slope, _, _, _, _ = linregress(
                        valid_avg_times_f, valid_avg_temps_f)
                    features['hotspot_avg_temp_change_rate_initial'] = slope if np.isfinite(
                        slope) else np.nan
                except ValueError:
                    pass
                features['hotspot_avg_temp_change_magnitude_initial'] = valid_avg_temps_f[-1] - \
                    valid_avg_temps_f[0]
            peak_pixel_temp_list_focus = []
            peak_func = np.nanmin if envir_para < 0 else np.nanmax
            for t in range(actual_focus_frames):
                pixels_in_mask_t_f = valid_masked_pixels_ts_focus[:, t]
                peak_pixel_temp_list_focus.append(
                    peak_func(pixels_in_mask_t_f) if pixels_in_mask_t_f.size > 0 else np.nan)
            peak_pixel_temp_series_focus = np.array(peak_pixel_temp_list_focus)
            valid_peak_temps_mask_focus = np.isfinite(
                peak_pixel_temp_series_focus)
            if np.sum(valid_peak_temps_mask_focus) >= 2:
                valid_peak_times_f = time_vector_focus[valid_peak_temps_mask_focus]
                valid_peak_temps_f = peak_pixel_temp_series_focus[valid_peak_temps_mask_focus]
                try:
                    slope_peak, _, _, _, _ = linregress(
                        valid_peak_times_f, valid_peak_temps_f)
                    features['peak_pixel_temp_change_rate_initial'] = slope_peak if np.isfinite(
                        slope_peak) else np.nan
                except ValueError:
                    pass
                features['peak_pixel_temp_change_magnitude_initial'] = valid_peak_temps_f[-1] - \
                    valid_peak_temps_f[0]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                if np.sum(valid_avg_temps_mask_focus) > 0:
                    features['temp_mean_avg_initial'] = np.nanmean(
                        hotspot_avg_temp_series_focus[valid_avg_temps_mask_focus])
                std_temps_per_frame_focus = np.nanstd(
                    valid_masked_pixels_ts_focus, axis=0)
                valid_std_mask_focus = np.isfinite(std_temps_per_frame_focus)
                if np.any(valid_std_mask_focus):
                    features['temp_std_avg_initial'] = np.nanmean(
                        std_temps_per_frame_focus[valid_std_mask_focus])
                if valid_masked_pixels_ts_focus.size > 0:
                    features['temp_min_overall_initial'] = np.nanmin(
                        valid_masked_pixels_ts_focus)
                    features['temp_max_overall_initial'] = np.nanmax(
                        valid_masked_pixels_ts_focus)
            except Exception as e:
                print(f"Warn: Init dist features failed: {e}")

    # --- Calculate Difference Frames & Full Duration Features ---
    try:
        initial_frame = frames[:, :, 0:1]
        if not np.all(np.isfinite(initial_frame[hotspot_mask, :])):
            print("Warning: Non-finite values in initial frame within mask.")

        frames_diff = frames - initial_frame

        masked_pixels_diff_ts = frames_diff[hotspot_mask, :]
        finite_pixel_mask_rows_full = np.all(
            np.isfinite(masked_pixels_diff_ts), axis=1)
        valid_masked_pixels_diff_ts = masked_pixels_diff_ts[finite_pixel_mask_rows_full, :]
        num_valid_pixels_full = valid_masked_pixels_diff_ts.shape[0]

        if num_valid_pixels_full > 0:
            # --- Calculate Average & Std Dev of Temp Difference ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
            hotspot_avg_temp_diff_series = np.nanmean(
                valid_masked_pixels_diff_ts, axis=0)
            hotspot_std_temp_diff_series = np.nanstd(
                valid_masked_pixels_diff_ts, axis=0)
            valid_avg_diff_mask = np.isfinite(hotspot_avg_temp_diff_series)
            valid_std_diff_mask = np.isfinite(hotspot_std_temp_diff_series)
            stabilization_start_frame = int(N_total * 0.75)
            if np.sum(valid_avg_diff_mask) > 0:
                valid_avg_diff_vals = hotspot_avg_temp_diff_series[valid_avg_diff_mask]
                features['overall_mean_deltaT'] = np.mean(valid_avg_diff_vals)
                features['max_abs_mean_deltaT'] = np.max(
                    np.abs(valid_avg_diff_vals))
                if stabilization_start_frame < N_total:
                    stabilized_mask = valid_avg_diff_mask[stabilization_start_frame:]
                    features['stabilized_mean_deltaT'] = np.mean(
                        hotspot_avg_temp_diff_series[stabilization_start_frame:][stabilized_mask]) if np.sum(stabilized_mask) > 0 else np.nan
            if np.sum(valid_std_diff_mask) > 0:
                features['overall_std_deltaT'] = np.mean(
                    hotspot_std_temp_diff_series[valid_std_diff_mask])
                if stabilization_start_frame < N_total:
                    stabilized_mask_std = valid_std_diff_mask[stabilization_start_frame:]
                    features['stabilized_std_deltaT'] = np.mean(
                        hotspot_std_temp_diff_series[stabilization_start_frame:][stabilized_mask_std]) if np.sum(stabilized_mask_std) > 0 else np.nan

            # --- Area of Significant Change (using FIXED Threshold) ---
            # Check if the passed fixed threshold is valid
            if threshold_abs_change is not None and np.isfinite(threshold_abs_change) and threshold_abs_change >= 0:
                area_significant_change_series = np.full(N_total, np.nan)
                for t in range(N_total):
                    # Already filtered for NaNs across time
                    valid_pixels_diff_t = valid_masked_pixels_diff_ts[:, t]
                    if valid_pixels_diff_t.size > 0:
                        # Use the passed fixed absolute threshold
                        area_significant_change_series[t] = np.sum(
                            np.abs(valid_pixels_diff_t) > threshold_abs_change)

                valid_area_mask = np.isfinite(area_significant_change_series)
                if np.sum(valid_area_mask) > 0:
                    valid_area_vals = area_significant_change_series[valid_area_mask]
                    features['mean_area_significant_change'] = np.mean(
                        valid_area_vals)
                    features['max_area_significant_change'] = np.max(
                        valid_area_vals)
                    # Stabilized Area
                    if stabilization_start_frame < N_total:
                        stabilized_mask_area = valid_area_mask[stabilization_start_frame:]
                        if np.sum(stabilized_mask_area) > 0:
                            features['stabilized_area_significant_change'] = np.mean(
                                area_significant_change_series[stabilization_start_frame:][stabilized_mask_area])
            else:
                print(
                    f"Skipping area features due to invalid fixed threshold: {threshold_abs_change}")
                features['mean_area_significant_change'] = np.nan
                features['stabilized_area_significant_change'] = np.nan
                features['max_area_significant_change'] = np.nan

        else:
            print("Warning: No valid pixels over full duration for difference features.")

    except Exception as e:
        print(f"Error calculating difference/stabilized/area features: {e}")
        traceback.print_exc()
        diff_keys = ['stabilized_mean_deltaT', 'overall_mean_deltaT', 'max_abs_mean_deltaT', 'stabilized_std_deltaT',
                     'overall_std_deltaT', 'mean_area_significant_change', 'stabilized_area_significant_change', 'max_area_significant_change']
        for k in diff_keys:
            features[k] = np.nan

    return features

# Main extraction function
def extract_features_with_mask(
    frames_or_path,
    mask_path,
    fps,
    focus_duration_sec,
    envir_para,
    threshold_abs_change=0.5,       # fixed threshold
    source_folder_name="Unknown",
    mat_filename_no_ext="Unknown"
):
    """Loads frames and mask, extracts features using FIXED threshold."""

    # --- Load Mask & Frames ---
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found: {mask_path}")
        return None
    try:
        hotspot_mask = np.load(mask_path)
        assert hotspot_mask.dtype == bool
    except Exception as e:
        print(f"Error loading mask file {mask_path}: {e}")
        return None
    if isinstance(frames_or_path, str):
        if not os.path.exists(frames_or_path):
            print(f"Error: Mat file not found: {frames_or_path}")
            return None
        try:
            mat_key = getattr(config, 'MAT_FRAMES_KEY', 'TempFrames')
            mat_data = scipy.io.loadmat(frames_or_path)
            frames = mat_data.get(mat_key, None)
            assert frames is not None
            frames = frames.astype(np.float64)
        except Exception as e:
            print(f"Error loading frames from {frames_or_path}: {e}")
            return None
    elif isinstance(frames_or_path, np.ndarray):
        frames = frames_or_path.astype(np.float64)
    else:
        print("Error: Invalid frames_or_path.")
        return None
    if frames.ndim != 3 or frames.shape[2] < 2:
        print("Error: Invalid frames.")
        return None
    if frames.shape[:2] != hotspot_mask.shape:
        print("Error: Shape mismatch.")
        return None

    focus_duration_frames = int(focus_duration_sec * fps)
    focus_duration_frames = max(2, min(focus_duration_frames, frames.shape[2]))

    extracted_features = calculate_hotspot_features(
        frames=frames,
        hotspot_mask=hotspot_mask,
        fps=fps,
        focus_duration_frames=focus_duration_frames,
        envir_para=envir_para,
        threshold_abs_change=threshold_abs_change  # Pass the fixed threshold
    )

    if extracted_features is None:
        print("Feature calculation failed.")
        return None

    return extracted_features
