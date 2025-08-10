# src_cnn/feature_engineering.py
"""
Contains functions for calculating a comprehensive suite of handcrafted summary-statistic
features from thermal videos. This module is driven by the central configuration 
file (src_cnn/config.py) and is a refactored version of the original feature
engineering logic to ensure all capabilities are preserved.
"""
import numpy as np
import cv2
import scipy.io
from scipy.stats import skew, kurtosis, mstats
import warnings
import traceback

# Import the central config file
from src_cnn import config as cfg

def calculate_radial_profile(frame_data, mask, center_y, center_x, num_bins=5):
    """Calculates the mean temperature in concentric radial bins around a center point."""
    if not np.any(mask): return [np.nan] * num_bins
    
    y_coords, x_coords = np.where(mask)
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_radius = np.max(distances)
    
    if max_radius == 0:
        val = frame_data[int(center_y), int(center_x)]
        return [val] + [np.nan] * (num_bins - 1)

    bin_width = max_radius / num_bins
    masked_values = frame_data[y_coords, x_coords]
    
    bin_indices = np.floor(distances / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    profile = [np.nanmean(masked_values[bin_indices == i]) if np.any(bin_indices == i) else np.nan for i in range(num_bins)]
    return profile

def calculate_hotspot_features(frames, hotspot_mask, envir_para=-1, threshold_abs_change=0.5):
    """
    Calculates a suite of handcrafted features for a given video sequence and mask.
    All parameters are pulled from the central config file.
    """
    features = {name: np.nan for name in cfg.HANDCRAFTED_FEATURES_TO_EXTRACT}
    
    if frames is None or frames.ndim != 3 or frames.shape[2] < 2: return features
    if hotspot_mask is None or not np.any(hotspot_mask): return features

    H, W, N_total = frames.shape
    mask_uint8 = hotspot_mask.astype(np.uint8)

    # --- 1. Area & Basic Geometry ---
    if 'hotspot_area' in features:
        features['hotspot_area'] = np.sum(hotspot_mask)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if 'num_hotspots' in features:
        features['num_hotspots'] = num_labels - 1

    cX, cY = np.nan, np.nan
    try:
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.concatenate(contours, axis=0)
            M = cv2.moments(all_points)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            if 'bbox_area' in features or 'bbox_aspect_ratio' in features:
                x, y, w, h = cv2.boundingRect(all_points)
                features['bbox_area'] = w * h
                features['bbox_aspect_ratio'] = w / h if h > 0 else 0
            
            if 'hotspot_solidity' in features:
                hull = cv2.convexHull(all_points)
                hull_area = cv2.contourArea(hull)
                features['hotspot_solidity'] = features.get('hotspot_area', 0) / hull_area if hull_area > 0 else np.nan

            if 'centroid_distance' in features:
                features['centroid_distance'] = 0
                if features.get('num_hotspots', 0) == 2 and len(centroids) > 2:
                    p1, p2 = centroids[1], centroids[2]
                    features['centroid_distance'] = np.sqrt(np.sum((p1 - p2)**2))
    except Exception as e:
        print(f"Warning: Geometry feature calculation failed: {e}")

    if np.isfinite(cX) and np.isfinite(cY):
        if any(f.startswith('radial_profile_') for f in features):
            median_frame = np.median(frames, axis=2)
            radial_vals = calculate_radial_profile(median_frame, hotspot_mask, cY, cX, num_bins=5)
            for i in range(5):
                if f'radial_profile_{i}' in features:
                    features[f'radial_profile_{i}'] = radial_vals[i]

    # --- 2. Initial Features (within Focus Duration) ---
    focus_duration_frames = int(cfg.FOCUS_DURATION_SECONDS * cfg.TRUE_FPS)
    actual_focus_frames = min(focus_duration_frames, N_total)
    
    if actual_focus_frames >= 2:
        frames_focus = frames[:, :, :actual_focus_frames]
        time_vector = (np.arange(actual_focus_frames) / cfg.TRUE_FPS)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            masked_pixels = frames_focus[hotspot_mask, :]
            
            mean_series = np.nanmean(masked_pixels, axis=0)
            std_series = np.nanstd(masked_pixels, axis=0)
            peak_func = np.nanmin if envir_para < 0 else np.nanmax
            peak_series = np.array([peak_func(masked_pixels[:, t]) for t in range(actual_focus_frames)])
            
            if 'hotspot_avg_temp_change_rate_initial' in features:
                slope, _, _, _ = mstats.theilslopes(mean_series, time_vector)
                features['hotspot_avg_temp_change_rate_initial'] = slope
            if 'hotspot_avg_temp_change_magnitude_initial' in features:
                features['hotspot_avg_temp_change_magnitude_initial'] = mean_series[-1] - mean_series[0]
            if 'peak_pixel_temp_change_rate_initial' in features:
                slope, _, _, _ = mstats.theilslopes(peak_series, time_vector)
                features['peak_pixel_temp_change_rate_initial'] = slope
            if 'peak_pixel_temp_change_magnitude_initial' in features:
                features['peak_pixel_temp_change_magnitude_initial'] = peak_series[-1] - peak_series[0]
            if 'rate_of_std_change_initial' in features:
                slope, _, _, _ = mstats.theilslopes(std_series, time_vector)
                features['rate_of_std_change_initial'] = slope

            if 'temp_mean_avg_initial' in features: features['temp_mean_avg_initial'] = np.nanmean(mean_series)
            if 'temp_std_avg_initial' in features: features['temp_std_avg_initial'] = np.nanmean(std_series)
            if 'temp_min_overall_initial' in features: features['temp_min_overall_initial'] = np.nanmin(masked_pixels)
            if 'temp_max_overall_initial' in features: features['temp_max_overall_initial'] = np.nanmax(masked_pixels)
            
            if 'time_to_peak_mean_temp' in features:
                diff_series = mean_series - mean_series[0]
                peak_idx = np.argmax(np.abs(diff_series))
                features['time_to_peak_mean_temp'] = time_vector[peak_idx]

            if 'peak_to_average_ratio' in features and features.get('temp_mean_avg_initial', 0) != 0:
                features['peak_to_average_ratio'] = features.get('temp_max_overall_initial', np.nan) / features.get('temp_mean_avg_initial', np.nan)
            
            flat_pixels = masked_pixels.flatten()
            if 'temperature_skewness' in features: features['temperature_skewness'] = skew(flat_pixels)
            if 'temperature_kurtosis' in features: features['temperature_kurtosis'] = kurtosis(flat_pixels)

    # --- 3. Full Duration Features ---
    try:
        initial_frame_temp = np.nanmean(frames[hotspot_mask, 0])
        if np.isfinite(initial_frame_temp):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                masked_pixels_full = frames[hotspot_mask, :]
                
                mean_full_series = np.nanmean(masked_pixels_full, axis=0)
                std_full_series = np.nanstd(masked_pixels_full, axis=0)
                
                stabilization_start = int(N_total * 0.75)
                
                if 'overall_mean_deltaT' in features: features['overall_mean_deltaT'] = np.nanmean(mean_full_series - initial_frame_temp)
                if 'max_abs_mean_deltaT' in features: features['max_abs_mean_deltaT'] = np.nanmax(np.abs(mean_full_series - initial_frame_temp))
                if 'overall_std_deltaT' in features: features['overall_std_deltaT'] = np.nanmean(std_full_series)
                
                if 'stabilized_mean_deltaT' in features: features['stabilized_mean_deltaT'] = np.nanmean(mean_full_series[stabilization_start:] - initial_frame_temp)
                if 'stabilized_std_deltaT' in features: features['stabilized_std_deltaT'] = np.nanmean(std_full_series[stabilization_start:])
                
                if any(f in features for f in ['mean_area_significant_change', 'max_area_significant_change', 'stabilized_area_significant_change']):
                    diffs = masked_pixels_full - initial_frame_temp
                    area_series = np.sum(np.abs(diffs) > threshold_abs_change, axis=0)
                    if 'mean_area_significant_change' in features: features['mean_area_significant_change'] = np.nanmean(area_series)
                    if 'max_area_significant_change' in features: features['max_area_significant_change'] = np.nanmax(area_series)
                    if 'stabilized_area_significant_change' in features: features['stabilized_area_significant_change'] = np.nanmean(area_series[stabilization_start:])
    except Exception as e:
        print(f"Warning: Full duration feature calculation failed: {e}")

    return features

def calculate_features_from_video(mat_filepath, mask_paths):
    """
    High-level wrapper to load a video and masks, combine them, and extract all features
    defined in the config file.

    Args:
        mat_filepath (str): Path to the .mat video file.
        mask_paths (list): A list of paths to the .npy mask files.

    Returns:
        dict: A dictionary of calculated features.
    """
    try:
        mat_data = scipy.io.loadmat(mat_filepath)
        frames = mat_data.get('TempFrames').astype(np.float64)
    except Exception as e:
        print(f"Error loading frames from {mat_filepath}: {e}")
        return {name: np.nan for name in cfg.HANDCRAFTED_FEATURES_TO_EXTRACT}

    if not mask_paths:
        print("Error: No mask paths provided.")
        return {name: np.nan for name in cfg.HANDCRAFTED_FEATURES_TO_EXTRACT}

    combined_mask = np.zeros(frames.shape[:2], dtype=bool)
    for mask_path in mask_paths:
        try:
            mask = np.load(mask_path)
            combined_mask = np.logical_or(combined_mask, mask)
        except Exception as e:
            print(f"Warning: Could not load or combine mask {mask_path}: {e}")
            continue

    if not np.any(combined_mask):
        print("Warning: Combined mask is empty.")
        return {name: np.nan for name in cfg.HANDCRAFTED_FEATURES_TO_EXTRACT}

    # The envir_para is typically -1 for cooling (leaks)
    return calculate_hotspot_features(frames=frames, hotspot_mask=combined_mask, envir_para=1)