# vis_focus_area_two.py
"""
Standalone script to visualize the DYNAMIC focus area for a single thermal .mat file,
based on processing an initial time window using interval gradients or slope,
with tunable morphology and optional pre-blurring.

Workflow:
1. Define an initial focus duration (e.g., first 5 seconds).
2. Calculate gradient maps for intervals within this duration OR calculate slope map over this duration.
3. Combine interval maps based on selected strategy (mean, max, first) OR use slope map directly as the activity map.
4. Optionally apply Gaussian blur to the activity map.
5. Identify hotspot from the (potentially blurred) activity map using quantile thresholding and selected morphology.
6. Save diagnostic images: original, mean, interval grads (from focus window), activity map, mask, overlay.

Saves images into a structured output directory:
<output_base_dir>/focused_images/<source_folder_name>/<mat_file_name>/
"""

import numpy as np
import cv2
import os
import scipy.io
import argparse
import sys
import traceback # Added for better error reporting

def compute_mean_frame(frames):
    """Computes the pixel-wise mean frame across all video frames."""
    if frames is None or not isinstance(frames, np.ndarray) or frames.ndim != 3 or frames.shape[2] == 0:
        print("Warning: Cannot compute mean frame. Invalid input.")
        return None
    try:
        return np.mean(frames, axis=2, dtype=np.float64)
    except Exception as e:
        print(f"Error computing mean frame: {e}")
        traceback.print_exc()
        return None

def compute_interval_gradient_map(frames, start_frame, end_frame):
    """
    Calculates a 2D map of mean absolute temporal gradient per pixel
    ONLY for frames within the specified interval [start_frame, end_frame).
    Clips end_frame to available frames.
    """
    if frames is None or frames.ndim != 3:
        print("Warning: Cannot compute interval gradient. Invalid frames input.")
        return None
    total_frames_available = frames.shape[2]
    actual_start = max(0, start_frame)
    # Ensure end_frame does not exceed available frames
    actual_end = min(total_frames_available, end_frame)

    if actual_end - actual_start < 2:
        # print(f"Debug: Need at least 2 frames for interval {start_frame}-{end_frame}, clipped to {actual_start}-{actual_end}, got {actual_end - actual_start}.")
        return None # Need at least 2 frames in interval

    height, width = frames.shape[:2]
    gradient_sum = np.zeros((height, width), dtype=np.float64)
    valid_diff_count = 0
    # Iterate up to actual_end-1 to compare frame i and i-1
    for i in range(actual_start + 1, actual_end):
        try:
            frame_i = frames[:, :, i].astype(np.float64)
            frame_prev = frames[:, :, i-1].astype(np.float64)
            diff = np.abs(frame_i - frame_prev)
            gradient_sum += diff
            valid_diff_count += 1
        except IndexError:
             print(f"Warning: Index error accessing frame {i} or {i-1} (bounds {actual_start}-{actual_end}).")
             continue
        except Exception as e:
             print(f"Warning: Error calculating diff frame {i} vs {i-1}: {e}")
             continue

    if valid_diff_count == 0:
        # print(f"Warning: No valid diffs calculated for interval {start_frame}-{end_frame} (clipped: {actual_start}-{actual_end})")
        return None
    return gradient_sum / valid_diff_count

# --- UPDATED to calculate slope over a specific segment ---
def compute_slope_map(frames, fps=30.0, start_frame=0, duration_frames=None):
    """
    Calculates the pixel-wise slope of temperature over a specified time segment
    using linear regression.

    Args:
        frames (np.ndarray): Input frames (H, W, N).
        fps (float): Frames per second.
        start_frame (int): Starting frame index for the segment.
        duration_frames (int, optional): Number of frames to include in the segment.
                                         If None, uses all frames from start_frame.

    Returns:
        np.ndarray: Map of absolute slopes, or None on error.
    """
    if frames is None or frames.ndim != 3:
        print("Warning: Cannot compute slope map. Invalid frames input.")
        return None

    total_frames = frames.shape[2]
    actual_start = max(0, start_frame)

    if duration_frames is None:
        actual_end = total_frames
    else:
        actual_end = min(actual_start + duration_frames, total_frames)

    T_segment = actual_end - actual_start # Number of frames in the segment

    if T_segment < 2:
        print(f"Warning: Need at least 2 frames for slope calculation in segment {actual_start}-{actual_end}. Got {T_segment}.")
        return None

    # Extract the segment of frames
    frames_segment = frames[:, :, actual_start:actual_end]

    try:
        dt = 1.0 / fps # Time step
        # Time vector relative to the start of the segment
        t = np.arange(T_segment, dtype=np.float64) * dt
        t_mean = t.mean()
        # Variance of time vector for the segment
        var_t = ((t - t_mean)**2).sum()
        if var_t < 1e-9:
             print(f"Warning: Time vector variance is near zero in segment {actual_start}-{actual_end}. Cannot compute slope accurately.")
             return np.zeros(frames.shape[:2], dtype=np.float64) # Return zero map

        f = frames_segment.astype(np.float64) # Pixel values over time in the segment
        mean_pix = f.mean(axis=2, keepdims=True) # Mean pixel value over segment time
        diff_f = f - mean_pix # Difference from mean pixel value in segment
        diff_t = t - t_mean # Difference from mean time in segment

        # Covariance between pixel values and time within the segment
        cov = np.tensordot(diff_f, diff_t, axes=([2], [0]))

        # Slope = Covariance / Variance(time)
        slopes = cov / var_t
        return np.abs(slopes) # Return absolute slope as activity measure

    except Exception as e:
        print(f"Error computing slope map for segment {actual_start}-{actual_end}: {e}")
        traceback.print_exc()
        return None


def extract_hotspot_from_map(activity_map, threshold_quantile=0.95, morphology_op='close',
                             apply_blur=False, blur_kernel_size=(3, 3)):
    """
    Extracts the hotspot mask from a 2D activity map, with optional pre-blurring.
    (Code from previous version is suitable, no changes needed here based on new requirements)
    """
    if activity_map is None or activity_map.size == 0:
        print("Warning: Activity map is None or empty in extract_hotspot_from_map.")
        return None, np.nan

    # --- Optional Pre-filtering ---
    map_to_process = activity_map.copy()
    if apply_blur:
        try:
            # Ensure kernel dimensions are odd
            k_h = blur_kernel_size[0] if blur_kernel_size[0] % 2 != 0 else blur_kernel_size[0] + 1
            k_w = blur_kernel_size[1] if blur_kernel_size[1] % 2 != 0 else blur_kernel_size[1] + 1
            map_to_process = cv2.GaussianBlur(map_to_process, (k_w, k_h), 0)
            # print(f"  Applied Gaussian Blur with kernel ({k_w}, {k_h})") # Less verbose
        except Exception as e:
            print(f"Warning: Failed to apply Gaussian Blur: {e}. Proceeding without blur.")
            map_to_process = activity_map.copy() # Revert to original if blur fails

    # --- Continue with processing the (potentially blurred) map ---
    proc_map = map_to_process
    nan_mask = np.isnan(proc_map)
    if np.all(nan_mask):
        print("Warning: Activity map is all NaNs after potential blur.")
        return None, np.nan # All NaNs

    valid_pixels = proc_map[~nan_mask]
    if valid_pixels.size == 0:
        print("Warning: No valid pixels found in activity map after potential blur.")
        return np.zeros_like(activity_map, dtype=bool), 0.0 # All valid pixels were NaN?

    map_std = np.std(valid_pixels)
    map_max = np.max(valid_pixels)

    # Check for empty or flat map
    if map_max < 1e-9 or map_std < 1e-9:
        # print("Warning: Activity map has near-zero variation. Hotspot detection may fail.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    # Calculate threshold
    try:
        threshold_value = np.percentile(valid_pixels, threshold_quantile * 100)
    except IndexError:
        print("Warning: Percentile calculation failed, using max value as threshold.")
        threshold_value = map_max # Fallback if percentile fails

    # Ensure threshold isn't effectively zero if max is non-zero
    if threshold_value <= 1e-9 and map_max > 1e-9:
        threshold_value = 1e-9

    # --- Thresholding ---
    binary_mask = (np.nan_to_num(proc_map, nan=-np.inf) >= threshold_value).astype(np.uint8)
    if not np.any(binary_mask):
        # print(f"Warning: Binary mask empty after thresholding activity map (quantile={threshold_quantile}).")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    # --- Morphological Operations ---
    mask_processed = binary_mask.copy()
    kernel_size_dim = max(min(3, activity_map.shape[0]//20, activity_map.shape[1]//20), 1)
    kernel_close = np.ones((kernel_size_dim, kernel_size_dim), np.uint8)
    kernel_open = np.ones((2, 2), np.uint8)

    try:
        if morphology_op == 'close':
            mask_processed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
        elif morphology_op == 'open_close':
            mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
            mask_processed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
        elif morphology_op == 'none':
            pass
        else:
             print(f"Warning: Unknown morphology_op '{morphology_op}'. Defaulting to 'close'.")
             mask_processed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    except cv2.error as e:
        print(f"Warning: OpenCV error during morphology '{morphology_op}': {e}. Using raw binary mask.")
        mask_processed = binary_mask

    # --- Connected Components ---
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_processed, connectivity=8)
    except cv2.error as e:
        print(f"Error during connectedComponentsWithStats: {e}. Cannot extract hotspot.")
        return None, np.nan

    hotspot_mask = np.zeros_like(activity_map, dtype=bool)
    hotspot_area = 0.0
    if num_labels > 1:
        if stats.shape[0] > 1:
             foreground_stats = stats[1:]
             if foreground_stats.size > 0:
                 largest_label_idx = np.argmax(foreground_stats[:, cv2.CC_STAT_AREA])
                 largest_label = largest_label_idx + 1
                 hotspot_mask = (labels == largest_label)
                 hotspot_area = np.sum(hotspot_mask)
    return hotspot_mask, hotspot_area


def save_visualizations(
    original_frame, mean_frame, interval_gradient_maps, activity_map,
    mask, index_prefix, save_dir, orig_colormap_enum, grad_colormap_enum
):
    """
    Saves multiple visualization images.
    (Code from previous version is suitable, no changes needed here based on new requirements)
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created visualization directory: {save_dir}")

        num_saved = 0
        h, w = (original_frame.shape[:2]) if original_frame is not None else (240, 320)
        blank_img = np.zeros((h, w), dtype=np.uint8)

        # --- Helper to normalize and save ---
        def _normalize_save(img_data, filename_base, description, colormap=None):
            nonlocal num_saved
            if img_data is None: print(f"  Skipping save: {description} data is None."); return None
            try:
                d_min, d_max = np.nanmin(img_data), np.nanmax(img_data)
                is_flat = np.isnan(d_min) or np.isnan(d_max) or (d_max - d_min) < 1e-9

                valid_mask_np = ~np.isnan(img_data)
                valid_mask_cv = valid_mask_np.astype(np.uint8)

                if is_flat:
                    norm_img = np.full(img_data.shape[:2], 0 if np.isnan(d_min) else 128, dtype=np.uint8)
                else:
                    norm_img = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=valid_mask_cv)
                    #cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dst=norm_img, dtype=cv2.CV_8U, mask=valid_mask_cv)

                path_gray = os.path.join(save_dir, f"{filename_base}_gray.png")
                cv2.imwrite(path_gray, norm_img)
                num_saved += 1

                if colormap is not None:
                    try:
                        color_img = cv2.applyColorMap(norm_img, colormap)
                        if not np.all(valid_mask_np):
                            color_img[~valid_mask_np] = [0, 0, 0]
                        path_color = os.path.join(save_dir, f"{filename_base}_color.png")
                        cv2.imwrite(path_color, color_img)
                        num_saved += 1
                    except Exception as cmap_e:
                        print(f"  Warning: Failed applying colormap for {description}: {cmap_e}")
                return norm_img # Return normalized gray image
            except Exception as e:
                print(f"  Error processing/saving normalized {description}: {e}")
                traceback.print_exc()
                return None

        # --- Start Saving ---
        print(f"  Saving visualization images with prefix: {index_prefix}...")

        # 1. RAW Original Frame Representations
        orig_gray_for_display = None
        orig_color_for_display = None
        if original_frame is not None:
            try:
                raw_orig_normalized = cv2.normalize(original_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                orig_gray_path = os.path.join(save_dir, f"{index_prefix}_1_original_raw_gray.png")
                cv2.imwrite(orig_gray_path, raw_orig_normalized)
                num_saved += 1
                orig_gray_for_display = raw_orig_normalized

                raw_orig_color = cv2.applyColorMap(raw_orig_normalized, orig_colormap_enum)
                orig_color_path = os.path.join(save_dir, f"{index_prefix}_1_original_raw_color.png")
                cv2.imwrite(orig_color_path, raw_orig_color)
                num_saved += 1
                orig_color_for_display = raw_orig_color
            except Exception as e:
                print(f"  Error processing/saving raw original frame: {e}")
        else:
            print("  Skipping raw original frame saving: original_frame is None.")

        # 2. NORMALIZED Original Frame
        vis_orig_norm = _normalize_save(original_frame, f"{index_prefix}_2_original_norm", "Normalized Original Frame", orig_colormap_enum)
        vis_orig_norm = vis_orig_norm if vis_orig_norm is not None else blank_img.copy()

        # 3. Mean Frame
        _normalize_save(mean_frame, f"{index_prefix}_3_mean_frame", "Mean Frame", orig_colormap_enum)

        # 4. Interval Gradient Maps (if calculated and relevant)
        if interval_gradient_maps and not index_prefix.split('_')[1] == 'slope':
             # print(f"  Saving {len(interval_gradient_maps)} interval gradient maps...")
             for i, grad_map in enumerate(interval_gradient_maps):
                 # Only save if grad_map is not None
                 if grad_map is not None:
                     _normalize_save(grad_map, f"{index_prefix}_4_interval_{i}_grad", f"Interval {i} Gradient", grad_colormap_enum)
                 else:
                     print(f"  Skipping save for interval {i} gradient map as it's None.")


        # 5. Final Activity Map
        _normalize_save(activity_map, f"{index_prefix}_5_activity_map", "Activity Map", grad_colormap_enum)

        # 6. Hotspot Mask
        if mask is not None:
            try:
                mask_img = (mask.astype(np.uint8) * 255)
                mask_path = os.path.join(save_dir, f"{index_prefix}_6_hotspot_mask.png")
                cv2.imwrite(mask_path, mask_img)
                num_saved += 1
            except Exception as e: print(f"  Error saving hotspot mask: {e}")
        else: print("  Skipping mask saving: Hotspot mask is None.")

        # 7. Overlay on RAW Original Grayscale
        if mask is not None and orig_gray_for_display is not None:
             try:
                 orig_gray_bgr = cv2.cvtColor(orig_gray_for_display, cv2.COLOR_GRAY2BGR)
                 overlay_img_raw_gray = orig_gray_bgr.copy()
                 overlay_color = [0, 255, 255]; # Yellow BGR
                 overlay_img_raw_gray[mask] = overlay_color
                 alpha = 0.4;
                 blended_img = cv2.addWeighted(overlay_img_raw_gray, alpha, orig_gray_bgr, 1 - alpha, 0)
                 overlay_path = os.path.join(save_dir, f"{index_prefix}_7_overlay_on_raw_gray.png")
                 cv2.imwrite(overlay_path, blended_img)
                 num_saved += 1
             except Exception as e: print(f"  Error saving overlay on raw gray: {e}")
        elif mask is None: print("  Skipping overlay on raw gray: Mask is None.")
        elif orig_gray_for_display is None: print("  Skipping overlay on raw gray: Raw gray image unavailable.")

        # 8. Overlay on RAW Original Colorized
        if mask is not None and orig_color_for_display is not None:
            try:
                overlay_img_raw_color = orig_color_for_display.copy()
                overlay_color = [0, 255, 255]; # Yellow BGR
                overlay_img_raw_color[mask] = overlay_color
                alpha = 0.4;
                blended_img = cv2.addWeighted(overlay_img_raw_color, alpha, orig_color_for_display, 1 - alpha, 0)
                overlay_path = os.path.join(save_dir, f"{index_prefix}_8_overlay_on_raw_color.png")
                cv2.imwrite(overlay_path, blended_img)
                num_saved += 1
            except Exception as e: print(f"  Error saving overlay on raw color: {e}")
        elif mask is None: print("  Skipping overlay on raw color: Mask is None.")
        elif orig_color_for_display is None: print("  Skipping overlay on raw color: Raw color image unavailable.")

        # 9. Overlay on NORMALIZED Original Grayscale
        if mask is not None and vis_orig_norm is not None:
             try:
                 vis_orig_norm_bgr = cv2.cvtColor(vis_orig_norm, cv2.COLOR_GRAY2BGR)
                 overlay_img_norm = vis_orig_norm_bgr.copy()
                 overlay_color = [0, 255, 255]; # Yellow BGR
                 overlay_img_norm[mask] = overlay_color
                 alpha = 0.4;
                 blended_img = cv2.addWeighted(overlay_img_norm, alpha, vis_orig_norm_bgr, 1 - alpha, 0)
                 overlay_path = os.path.join(save_dir, f"{index_prefix}_9_overlay_on_norm_gray.png")
                 cv2.imwrite(overlay_path, blended_img)
                 num_saved += 1
             except Exception as e: print(f"  Error saving overlay on normalized gray: {e}")

        print(f"Saved {num_saved} visualization images to '{save_dir}'")

    except Exception as e:
        print(f"!!! Error during visualization saving: {e}")
        traceback.print_exc()


# --- UPDATED Main Execution Logic ---
def run_single_visualization(mat_file_path, output_base_dir, mat_key,
                             interval_size, # Still needed for 'mean', 'max', 'first' within focus window
                             quantile,
                             combination_strategy, morphology_op,
                             apply_blur, blur_kernel_size,
                             focus_duration_sec, # New argument for focus window
                             fps, # Needed for frame calculation and slope
                             orig_cmap_enum, grad_cmap_enum):
    """Orchestrates loading, processing focusing on initial duration, and visualization."""
    print("-" * 50)
    print(f"Processing: {mat_file_path}")
    print(f"  Settings: focus_duration={focus_duration_sec}s, quantile={quantile:.3f}") # Increased quantile precision display
    print(f"  Strategy: combination='{combination_strategy}', morphology='{morphology_op}'")
    print(f"  Blurring: apply={apply_blur}, kernel={blur_kernel_size if apply_blur else 'N/A'}")
    if combination_strategy != 'slope': print(f"  Interval Size (within focus): {interval_size}")


    # --- Load Data ---
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        if mat_key not in mat_data: raise KeyError(f"Key '{mat_key}' not found in {mat_file_path}.")
        frames = mat_data[mat_key]
        if not isinstance(frames, np.ndarray): raise TypeError("Frame data is not a numpy array.")
        if frames.ndim != 3: raise ValueError(f"Expected 3D array (H, W, N), got {frames.ndim}D.")
        if frames.shape[2] < 2: raise ValueError(f"Need at least 2 frames, got {frames.shape[2]}.")
        print(f"  Loaded frames: {frames.shape}, dtype: {frames.dtype}")
        num_frames = frames.shape[2]
    except Exception as e:
        print(f"  Error loading data from {mat_file_path}: {e}")
        traceback.print_exc()
        return False

    # --- Calculate Focus Window ---
    focus_duration_frames = int(focus_duration_sec * fps)
    # Ensure we don't request more frames than available, and have at least 2
    focus_duration_frames = max(2, min(focus_duration_frames, num_frames))
    print(f"  Focusing on first {focus_duration_sec}s => {focus_duration_frames} frames.")

    # --- Calculations ---
    # 1. Calculate mean frame (over ALL frames, still useful)
    mean_frame = compute_mean_frame(frames)

    # 2. Determine Activity Map based on Strategy using FOCUS WINDOW
    activity_map = None
    interval_gradient_maps = [] # Store maps calculated within focus window

    if combination_strategy == 'slope':
        print(f"  Calculating activity map using slope over first {focus_duration_frames} frames...")
        activity_map = compute_slope_map(frames, fps=fps, start_frame=0, duration_frames=focus_duration_frames)
        if activity_map is None:
            print("  Error: Failed to compute slope map for focus duration.")
            return False # Critical error if chosen strategy fails
    else:
        # Calculate Interval Gradient Maps WITHIN the focus window
        print(f"  Calculating interval gradient maps within first {focus_duration_frames} frames...")
        # Calculate how many intervals fit *within* the focus duration
        # Note: max_intervals arg is now less relevant, focus_duration dictates range
        num_intervals_in_focus = (focus_duration_frames + interval_size -1) // interval_size # Ceiling division

        for i in range(num_intervals_in_focus):
            start_frame = i * interval_size
            # Important: Ensure end_frame for calculation does not exceed focus_duration_frames
            end_frame = min(start_frame + interval_size, focus_duration_frames)

            # Skip if interval starts beyond the focus duration (or gives less than 2 frames)
            if start_frame >= focus_duration_frames - 1:
                continue

            interval_map = compute_interval_gradient_map(frames, start_frame, end_frame)
            if interval_map is not None:
                interval_gradient_maps.append(interval_map)
            # else: print(f"  Warning: Failed interval gradient map for {start_frame}-{end_frame}")

        if not interval_gradient_maps:
            print("  Error: No interval gradient maps calculated within the focus duration.")
            return False # Critical error

        # Combine the calculated interval maps
        print(f"  Combining {len(interval_gradient_maps)} interval map(s) from focus window using '{combination_strategy}'...")
        valid_maps = [m for m in interval_gradient_maps if m is not None and m.shape == interval_gradient_maps[0].shape]

        if not valid_maps:
            print("  Error: No valid/consistent interval maps to combine from focus window.")
            return False

        try:
            if combination_strategy == 'mean':
                activity_map = np.nanmean(np.stack(valid_maps, axis=0), axis=0)
            elif combination_strategy == 'max':
                activity_map = np.nanmax(np.stack(valid_maps, axis=0), axis=0)
            elif combination_strategy == 'first_interval':
                activity_map = valid_maps[0] # Use the first calculated interval map
            else: # Should not happen
                print(f"  Internal Error: Unexpected combination_strategy '{combination_strategy}'.")
                return False
        except Exception as e:
            print(f"  Error combining maps from focus window using '{combination_strategy}': {e}")
            traceback.print_exc()
            return False

    # Check if activity map calculation failed
    if activity_map is None:
         print("  Error: Activity map is None after calculation/combination step.")
         return False

    # 3. Extract Hotspot from the Activity Map (derived from focus window)
    print(f"  Extracting hotspot from activity map (quantile={quantile:.3f}, morphology='{morphology_op}', blur={apply_blur})...")
    hotspot_mask, hotspot_area = extract_hotspot_from_map(
        activity_map,
        threshold_quantile=quantile,
        morphology_op=morphology_op,
        apply_blur=apply_blur,
        blur_kernel_size=blur_kernel_size
    )
    if hotspot_mask is None:
         print("  Warning: Failed hotspot extraction. Mask will not be saved/overlaid.")
    else:
        # Reduce verbosity, print only if area > 0
        if hotspot_area > 0: print(f"  Hotspot area: {hotspot_area:.0f} pixels")
        elif hotspot_area == 0: print("  Warning: Hotspot area is 0 pixels.")


    # --- Prepare Output Path ---
    try:
        mat_filename = os.path.basename(mat_file_path)
        mat_filename_no_ext, _ = os.path.splitext(mat_filename)
        source_folder_name = os.path.basename(os.path.dirname(mat_file_path))
        if not source_folder_name: source_folder_name = "_root_"

        final_save_dir = os.path.join(output_base_dir, "focused_images", source_folder_name, mat_filename_no_ext)
    except Exception as e:
        print(f"  Error constructing output path: {e}")
        traceback.print_exc()
        return False

    # --- Save Visualizations ---
    first_frame = frames[:, :, 0] if frames.shape[2] > 0 else None
    # --- UPDATED index_prefix to include focus duration ---
    blur_suffix = f"_blur{blur_kernel_size[0]}x{blur_kernel_size[1]}" if apply_blur else "_noblur"
    vis_prefix = f"q{quantile:.3f}_dur{focus_duration_sec}s_{combination_strategy}_{morphology_op}{blur_suffix}"

    save_visualizations(
        original_frame=first_frame,
        mean_frame=mean_frame, # Mean over full duration is still useful context
        interval_gradient_maps=interval_gradient_maps, # Pass maps calculated within focus window
        activity_map=activity_map, # Pass the map used for detection
        mask=hotspot_mask, # Pass the potentially None mask
        index_prefix=vis_prefix, # Use descriptive prefix
        save_dir=final_save_dir,
        orig_colormap_enum=orig_cmap_enum,
        grad_colormap_enum=grad_cmap_enum
    )
    print("-" * 50)
    return True

# --- Command-Line Argument Parsing ---
if __name__ == "__main__":

    # Build available colormap dictionary robustly
    colormap_choices = {name: val for name, val in cv2.__dict__.items() if name.startswith('COLORMAP_')}
    valid_cv2_colormaps = {}
    for name, enum_val in colormap_choices.items():
         simple_name = name.replace('COLORMAP_', '').lower()
         valid_cv2_colormaps[simple_name] = enum_val

    parser = argparse.ArgumentParser(
        description="Visualize dynamic focus area using an initial time window from a thermal .mat file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input / Output
    parser.add_argument("mat_file", help="Path to the input .mat file.")
    parser.add_argument("output_dir", help="Path to the BASE output directory (subdirs will be created).")
    parser.add_argument("-k", "--key", default="TempFrames", help="Key in the .mat file for frame data.")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second of the recording.")

    # Focus Window & Strategy
    parser.add_argument("--focus_duration_sec", type=float, default=3.0,
                        help="Duration (in seconds) from the start of the video to use for generating the activity map.")
    parser.add_argument("--cm", "--combination_strategy", default="slope", choices=['mean', 'max', 'first_interval', 'slope'], dest='combination_strategy',
                        help="Strategy for activity map: 'mean'/'max'/'first_interval' use interval gradients within focus duration, 'slope' uses linear regression over focus duration.")
    parser.add_argument("-i", "--interval_size", type=int, default=50,
                        help="Number of frames per interval (used only if --cm is 'mean'/'max'/'first_interval').")

    # Hotspot Extraction Parameters
    parser.add_argument("-q", "--quantile", type=float, default=0.98, # Changed default based on feedback
                        help="Quantile threshold (0-1) for hotspot extraction on activity map.")
    parser.add_argument("--morph_op", default="none", choices=['close', 'open_close', 'none'], # Changed default
                        help="Morphological operation applied to the thresholded mask.")
    parser.add_argument("--blur", action='store_true', help="Apply Gaussian blur to the activity map before thresholding.")
    parser.add_argument("--blur_kernel", type=int, nargs=2, default=[3, 3], metavar=('H', 'W'),
                        help="Kernel size (height width) for Gaussian blur (use odd numbers). Used only if --blur is specified.")

    # Visualization
    parser.add_argument("-c", "--colormap", default="hot", choices=list(valid_cv2_colormaps.keys()), help="Colormap for original/mean image visualization.")
    parser.add_argument("-g", "--grad_colormap", default="inferno", choices=list(valid_cv2_colormaps.keys()), help="Colormap for gradient/activity/slope map visualization.")

    # Check if enough arguments provided
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # --- Argument Validation ---
    if not 0.0 < args.quantile <= 1.0:
         print("Error: Quantile must be between 0 (exclusive) and 1 (inclusive).")
         sys.exit(1)
    if args.focus_duration_sec <= 0:
         print("Error: Focus duration must be positive.")
         sys.exit(1)
    if args.fps <= 0:
         print("Error: FPS must be positive.")
         sys.exit(1)
    if args.combination_strategy != 'slope' and args.interval_size <= 0:
         print("Error: Interval size must be positive when using mean/max/first_interval strategies.")
         sys.exit(1)
    if args.blur:
        blur_k_h, blur_k_w = args.blur_kernel
        if blur_k_h <= 0 or blur_k_w <= 0:
             print("Error: Blur kernel dimensions must be positive.")
             sys.exit(1)
        # Warning for even kernel sizes, handled in function
        if blur_k_h % 2 == 0 or blur_k_w % 2 == 0:
            print("Warning: Blur kernel dimensions should ideally be odd. Function will adjust.")

    # Get selected colormap enums
    selected_orig_cmap_enum = valid_cv2_colormaps[args.colormap]
    selected_grad_cmap_enum = valid_cv2_colormaps[args.grad_colormap]

    # Run the visualization process
    success = run_single_visualization(
        mat_file_path=args.mat_file,
        output_base_dir=args.output_dir,
        mat_key=args.key,
        interval_size=args.interval_size,
        # max_intervals arg is no longer needed here, focus duration determines loops
        quantile=args.quantile,
        combination_strategy=args.combination_strategy,
        morphology_op=args.morph_op,
        apply_blur=args.blur,
        blur_kernel_size=tuple(args.blur_kernel),
        focus_duration_sec=args.focus_duration_sec, # Pass new arg
        fps=args.fps,                               # Pass fps
        orig_cmap_enum=selected_orig_cmap_enum,
        grad_cmap_enum=selected_grad_cmap_enum
    )

    if not success:
        print("Script finished with errors.")
        sys.exit(1)
    else:
        print("Script finished successfully.")
        sys.exit(0)