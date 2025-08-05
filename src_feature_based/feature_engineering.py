# feature_engineering.py
"""
(Corrected Version: Fixes UnboundLocalError and Pylance warnings)
- Ensures all geometry and derived initial features are safely calculated.
- Corrects the flow for radial profile, bbox features.
"""

import numpy as np
import cv2
import os
import scipy.io
from src_feature_based import config
from scipy.stats import linregress, skew, kurtosis, mstats
import warnings
import traceback

# --- SINGLE SOURCE OF TRUTH FOR ALL FEATURE NAMES ---
ALL_POSSIBLE_FEATURE_NAMES = [
    'hotspot_area', 'hotspot_avg_temp_change_rate_initial', 'hotspot_avg_temp_change_magnitude_initial',
    'peak_pixel_temp_change_rate_initial', 'peak_pixel_temp_change_magnitude_initial', 'temp_mean_avg_initial',
    'temp_std_avg_initial', 'temp_min_overall_initial', 'temp_max_overall_initial',
    'stabilized_mean_deltaT', 'overall_mean_deltaT', 'max_abs_mean_deltaT',
    'stabilized_std_deltaT', 'overall_std_deltaT', 'mean_area_significant_change',
    'stabilized_area_significant_change', 'max_area_significant_change',
    'num_hotspots', 'hotspot_solidity', 'centroid_distance', 'time_to_peak_mean_temp',
    'temperature_skewness', 'temperature_kurtosis', 'rate_of_std_change_initial', 'peak_to_average_ratio',
    'radial_profile_0', 'radial_profile_1', 'radial_profile_2', 'radial_profile_3', 'radial_profile_4',
    'bbox_area', 'bbox_aspect_ratio' # Added bbox features if you want them
]

# --- RADIAL PROFILE FEATURE FUNCTION (No changes needed here) ---
def calculate_radial_profile(frame_data, mask, center_y, center_x, num_bins=5, max_radius=None):

    H, W = frame_data.shape
    if mask.sum() == 0: return [np.nan] * num_bins

    y_coords, x_coords = np.where(mask)
    
    if max_radius is None:
        if y_coords.size == 0: return [np.nan] * num_bins
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_radius = np.max(distances)
        if max_radius == 0: return [frame_data[center_y, center_x]] + [np.nan] * (num_bins - 1)

    radial_bins_data = [[] for _ in range(num_bins)]
    bin_width = max_radius / num_bins

    for r_idx in range(H):
        for c_idx in range(W):
            if mask[r_idx, c_idx] and np.isfinite(frame_data[r_idx, c_idx]):
                dist = np.sqrt((c_idx - center_x)**2 + (r_idx - center_y)**2)
                if dist <= max_radius:
                    bin_idx = min(int(dist / bin_width), num_bins - 1)
                    radial_bins_data[bin_idx].append(frame_data[r_idx, c_idx])

    profile = [np.nanmean(bin_values) if bin_values else np.nan for bin_values in radial_bins_data]
    return profile

def calculate_hotspot_features(
    frames,
    hotspot_mask,
    fps,
    focus_duration_frames,
    envir_para=-1,
    threshold_abs_change=0.5
):
    features = {name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES}

    if frames is None or frames.ndim != 3 or frames.shape[2] < 2: return features
    if hotspot_mask is None or hotspot_mask.ndim != 2 or hotspot_mask.dtype != bool or hotspot_mask.shape != frames.shape[:2]: return features
    
    H, W, N_total = frames.shape

    # --- 1. Area & Basic Geometry ---
    features['hotspot_area'] = np.sum(hotspot_mask)
    if features['hotspot_area'] == 0: return features

    mask_uint8 = hotspot_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    features['num_hotspots'] = num_labels - 1

    # Initialize M, cX, cY to safe defaults outside the try block
    M = None
    cX, cY = np.nan, np.nan
    all_points_in_contours = None # Initialize for bounding box too

    try:
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours: # Proceed only if contours are found
            M = cv2.moments(mask_uint8) # Calculate moments of the entire mask
            if M["m00"] != 0: # Ensure denominator is not zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else: # If m00 is zero, centroid is undefined
                cX, cY = np.nan, np.nan
            
            all_points_in_contours = np.concatenate(contours, axis=0) # Use this for overall geometry
            
            # Solidity
            hull = cv2.convexHull(all_points_in_contours)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                features['hotspot_solidity'] = features['hotspot_area'] / hull_area

            # Centroid Distance (only for 2 holes)
            if features['num_hotspots'] == 2 and len(centroids) > 2: # Ensure enough centroids for 2 holes
                p1, p2 = centroids[1], centroids[2] 
                features['centroid_distance'] = np.sqrt(np.sum((p1 - p2)**2))
            else:
                features['centroid_distance'] = 0 
                
            # Bounding Box features for the entire masked region (UNCOMMENTED)
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(all_points_in_contours)
            features['bbox_area'] = w_bbox * h_bbox
            if h_bbox > 0: features['bbox_aspect_ratio'] = w_bbox / h_bbox

    except Exception as e:
        print(f"Warning: Geometry feature calculation failed: {e}")
        traceback.print_exc() # For debugging, if still getting errors here

    # --- Radial Profile Features (now safely using cX, cY, M) ---
    if np.isfinite(cX) and np.isfinite(cY) and M is not None and M["m00"] != 0:
        median_frame = np.median(frames, axis=2)
        num_radial_bins = 5
        radial_profile_values = calculate_radial_profile(
            median_frame, hotspot_mask, cY, cX, num_bins=num_radial_bins
        )
        for i in range(num_radial_bins):
            features[f'radial_profile_{i}'] = radial_profile_values[i]
    else: # If centroid or moments were invalid, set radial features to NaN
        for i in range(5): features[f'radial_profile_{i}'] = np.nan

    # --- 2. Initial Features (Focus Duration) - REORDERED LOGIC (Already fixed) ---
    actual_focus_frames = min(focus_duration_frames, N_total)
    if actual_focus_frames >= 2:
        frames_focus = frames[:, :, :actual_focus_frames]
        time_vector_focus = (np.arange(actual_focus_frames) / fps).astype(np.float64)
        masked_pixels_ts_focus = frames_focus[hotspot_mask, :]
        finite_pixel_mask_rows_focus = np.all(np.isfinite(masked_pixels_ts_focus), axis=1)
        valid_masked_pixels_ts_focus = masked_pixels_ts_focus[finite_pixel_mask_rows_focus, :]
        num_valid_pixels_focus = valid_masked_pixels_ts_focus.shape[0]

        if num_valid_pixels_focus > 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    hotspot_avg_temp_series_focus = np.nanmean(valid_masked_pixels_ts_focus, axis=0)
                    valid_avg_temps_mask_focus = np.isfinite(hotspot_avg_temp_series_focus)
                    
                    std_temps_per_frame_focus = np.nanstd(valid_masked_pixels_ts_focus, axis=0)
                    valid_std_mask_focus = np.isfinite(std_temps_per_frame_focus)

                    # Derived features using base stats
                    if np.sum(valid_avg_temps_mask_focus) >= 2:
                        valid_avg_times_f = time_vector_focus[valid_avg_temps_mask_focus]
                        valid_avg_temps_f = hotspot_avg_temp_series_focus[valid_avg_temps_mask_focus]
                        slope, _, _, _ = mstats.theilslopes(valid_avg_temps_f, valid_avg_times_f, 0.95)
                        features['hotspot_avg_temp_change_rate_initial'] = slope if np.isfinite(slope) else np.nan
                        features['hotspot_avg_temp_change_magnitude_initial'] = valid_avg_temps_f[-1] - valid_avg_temps_f[0]
                        diff_series = valid_avg_temps_f - valid_avg_temps_f[0]
                        peak_idx = np.argmax(np.abs(diff_series))
                        features['time_to_peak_mean_temp'] = valid_avg_times_f[peak_idx] if len(valid_avg_times_f) > peak_idx else np.nan

                    peak_func = np.nanmin if envir_para < 0 else np.nanmax
                    peak_pixel_temp_series_focus = np.array([peak_func(valid_masked_pixels_ts_focus[:, t]) if valid_masked_pixels_ts_focus[:, t].size > 0 and np.any(np.isfinite(valid_masked_pixels_ts_focus[:, t])) else np.nan for t in range(actual_focus_frames)])
                    valid_peak_temps_mask_focus = np.isfinite(peak_pixel_temp_series_focus)
                    if np.sum(valid_peak_temps_mask_focus) >= 2:
                        valid_peak_times_f = time_vector_focus[valid_peak_temps_mask_focus]
                        valid_peak_temps_f = peak_pixel_temp_series_focus[valid_peak_temps_mask_focus]
                        slope_peak, _, _, _ = mstats.theilslopes(valid_peak_temps_f, valid_peak_times_f, 0.95)
                        features['peak_pixel_temp_change_rate_initial'] = slope_peak if np.isfinite(slope_peak) else np.nan
                        features['peak_pixel_temp_change_magnitude_initial'] = valid_peak_temps_f[-1] - valid_peak_temps_f[0]

                    if np.sum(valid_std_mask_focus) >= 2:
                        valid_std_series = std_temps_per_frame_focus[valid_std_mask_focus]
                        valid_std_times = time_vector_focus[valid_std_mask_focus]
                        std_slope, _, _, _ = mstats.theilslopes(valid_std_series, valid_std_times, 0.95)
                        features['rate_of_std_change_initial'] = std_slope if np.isfinite(std_slope) else np.nan

                    features['temp_mean_avg_initial'] = np.nanmean(hotspot_avg_temp_series_focus[valid_avg_temps_mask_focus])
                    features['temp_std_avg_initial'] = np.nanmean(std_temps_per_frame_focus[valid_std_mask_focus])
                    features['temp_min_overall_initial'] = np.nanmin(valid_masked_pixels_ts_focus)
                    features['temp_max_overall_initial'] = np.nanmax(valid_masked_pixels_ts_focus)
                    
                    if np.isfinite(features['temp_max_overall_initial']) and np.isfinite(features['temp_mean_avg_initial']) and features['temp_mean_avg_initial'] != 0:
                        features['peak_to_average_ratio'] = features['temp_max_overall_initial'] / features['temp_mean_avg_initial']
                    
                    all_pixel_values_focus = valid_masked_pixels_ts_focus.flatten()
                    if all_pixel_values_focus.size > 0:
                        features['temperature_skewness'] = skew(all_pixel_values_focus)
                        features['temperature_kurtosis'] = kurtosis(all_pixel_values_focus)

            except Exception as e:
                print(f"Warning: Initial feature calculation failed: {e}")
                traceback.print_exc()

    # --- 3. Full Duration Features --- (Corrected and streamlined)
    try:
        initial_frame_val = frames[hotspot_mask, 0].mean() # Use average initial temp in mask
        if not np.isfinite(initial_frame_val): raise ValueError("Invalid initial frame temp")

        masked_pixels_full_ts = frames[hotspot_mask, :]
        valid_rows_full = np.all(np.isfinite(masked_pixels_full_ts), axis=1)
        valid_pixels_full_ts = masked_pixels_full_ts[valid_rows_full, :]

        if valid_pixels_full_ts.shape[0] > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) # Suppress NaN warnings
            
            # Overall DeltaT features
            hotspot_avg_temp_full_series = np.nanmean(valid_pixels_full_ts, axis=0)
            if np.any(np.isfinite(hotspot_avg_temp_full_series)):
                features['overall_mean_deltaT'] = np.mean(hotspot_avg_temp_full_series - initial_frame_val)
                features['max_abs_mean_deltaT'] = np.max(np.abs(hotspot_avg_temp_full_series - initial_frame_val))

            # Overall Std DeltaT
            hotspot_std_temp_full_series = np.nanstd(valid_pixels_full_ts, axis=0)
            if np.any(np.isfinite(hotspot_std_temp_full_series)):
                features['overall_std_deltaT'] = np.mean(hotspot_std_temp_full_series)

            # Stabilized features (last 25% of frames)
            stabilization_start_frame = int(N_total * 0.75)
            if N_total - stabilization_start_frame > 0:
                stabilized_avg_series = hotspot_avg_temp_full_series[stabilization_start_frame:]
                if np.any(np.isfinite(stabilized_avg_series)):
                    features['stabilized_mean_deltaT'] = np.mean(stabilized_avg_series - initial_frame_val)
                
                stabilized_std_series = hotspot_std_temp_full_series[stabilization_start_frame:]
                if np.any(np.isfinite(stabilized_std_series)):
                    features['stabilized_std_deltaT'] = np.mean(stabilized_std_series)
            
            # Area of Significant Change
            if threshold_abs_change is not None and np.isfinite(threshold_abs_change) and threshold_abs_change >= 0:
                # Calculate differences from initial_frame_val for all frames
                temp_diffs_from_initial = valid_pixels_full_ts - initial_frame_val
                # Count pixels exceeding threshold for each frame
                area_significant_change_series = np.sum(np.abs(temp_diffs_from_initial) > threshold_abs_change, axis=0)
                
                if np.any(np.isfinite(area_significant_change_series)):
                    features['mean_area_significant_change'] = np.mean(area_significant_change_series)
                    features['max_area_significant_change'] = np.max(area_significant_change_series)
                    if N_total - stabilization_start_frame > 0:
                        stabilized_area_series = area_significant_change_series[stabilization_start_frame:]
                        if np.any(np.isfinite(stabilized_area_series)):
                            features['stabilized_area_significant_change'] = np.mean(stabilized_area_series)
        else:
            print("Warning: No valid pixels in mask for full duration analysis.")
    except Exception as e:
        print(f"Error calculating full duration features: {e}")
        traceback.print_exc()

    return features

def extract_aggregate_features(frames_or_path, mask_paths, fps, focus_duration_sec, envir_para, threshold_abs_change=0.5):
    """
    Loads a video and a LIST of masks, combines them, and extracts aggregate features.
    """
    if isinstance(frames_or_path, str):
        try:
            mat_key = getattr(config, 'MAT_FRAMES_KEY', 'TempFrames')
            mat_data = scipy.io.loadmat(frames_or_path)
            frames = mat_data.get(mat_key, None).astype(np.float64)
        except Exception as e:
            print(f"Error loading frames from {frames_or_path}: {e}")
            return None
    else: frames = frames_or_path.astype(np.float64)

    if not mask_paths: print("Error: No mask paths provided."); return None
        
    combined_mask = np.zeros(frames.shape[:2], dtype=bool)
    for mask_path in mask_paths:
        try:
            mask = np.load(mask_path)
            combined_mask = np.logical_or(combined_mask, mask)
        except Exception as e:
            print(f"Warning: Could not load or combine mask {mask_path}: {e}"); continue

    if np.sum(combined_mask) == 0:
        print("Warning: Combined mask is empty after loading all paths."); return None

    focus_duration_frames = int(focus_duration_sec * fps)
    focus_duration_frames = max(2, min(focus_duration_frames, frames.shape[2]))

    extracted_features = calculate_hotspot_features(
        frames=frames,
        hotspot_mask=combined_mask,
        fps=fps,
        focus_duration_frames=focus_duration_frames,
        envir_para=envir_para,
        threshold_abs_change=threshold_abs_change
    )
    return extracted_features