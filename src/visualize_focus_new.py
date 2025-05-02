# vis_focus_area_two.py
"""
Standalone script to visualize the DYNAMIC focus area for a single thermal .mat file,
based on processing an initial time window using interval gradients or slope,
with tunable morphology and optional pre-blurring.

NEW: Slope calculation now uses pixel-wise linear regression with error estimation
     and filtering/weighting based on professor's MATLAB logic.
"""

import numpy as np
import cv2
import os
import scipy.io
import argparse
import sys
import traceback # Added for better error reporting
from scipy import stats as sp_stats # For linregress
from tqdm import tqdm # <--- Added import

# --- Helper Functions (compute_mean_frame, compute_interval_gradient_map) ---
def compute_mean_frame(frames):
    # (Keep the function from the previous version)
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
    # (Keep the function from the previous version)
    if frames is None or frames.ndim != 3:
        print("Warning: Cannot compute interval gradient. Invalid frames input.")
        return None
    total_frames_available = frames.shape[2]
    actual_start = max(0, start_frame)
    actual_end = min(total_frames_available, end_frame)

    if actual_end - actual_start < 2:
        return None

    height, width = frames.shape[:2]
    gradient_sum = np.zeros((height, width), dtype=np.float64)
    valid_diff_count = 0
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
        return None
    return gradient_sum / valid_diff_count

# --- extract_hotspot_from_map (unchanged from ROI version) ---
def extract_hotspot_from_map(activity_map, threshold_quantile=0.95, morphology_op='close',
                             apply_blur=False, blur_kernel_size=(3, 3),
                             roi_mask=None):
    # (Keep the function from the previous version with ROI logic)
    if activity_map is None or activity_map.size == 0:
        print("Warning: Activity map is None or empty in extract_hotspot_from_map.")
        return None, np.nan
    if roi_mask is not None:
        if roi_mask.shape != activity_map.shape:
            print("Warning: ROI mask shape mismatch. Ignoring ROI mask.")
            map_to_process = activity_map.copy()
        else:
            map_to_process = activity_map.copy()
            map_to_process[~roi_mask] = np.nan
    else:
        map_to_process = activity_map.copy()
    if apply_blur:
        if np.all(np.isnan(map_to_process)):
             print("Warning: map_to_process is all NaN after ROI masking, skipping blur.")
        else:
            try:
                k_h = blur_kernel_size[0] if blur_kernel_size[0] % 2 != 0 else blur_kernel_size[0] + 1
                k_w = blur_kernel_size[1] if blur_kernel_size[1] % 2 != 0 else blur_kernel_size[1] + 1
                map_to_process = cv2.GaussianBlur(map_to_process, (k_w, k_h), 0)
            except Exception as e:
                print(f"Warning: Failed to apply Gaussian Blur: {e}. Proceeding without blur.")
                if roi_mask is not None: map_to_process[~roi_mask] = np.nan
                else: map_to_process = activity_map.copy()
    proc_map = map_to_process
    nan_mask = np.isnan(proc_map)
    if np.all(nan_mask):
        print("Warning: Activity map is all NaNs after ROI/blur.")
        return None, np.nan
    valid_pixels = proc_map[~nan_mask]
    if valid_pixels.size == 0:
        print("Warning: No valid (non-NaN) pixels found in activity map.")
        return np.zeros_like(activity_map, dtype=bool), 0.0
    map_std = np.std(valid_pixels)
    map_max = np.max(valid_pixels)
    if map_max < 1e-9 or map_std < 1e-9:
         return np.zeros_like(activity_map, dtype=bool), 0.0
    try:
        threshold_value = np.percentile(valid_pixels, threshold_quantile * 100)
    except IndexError:
        print("Warning: Percentile calculation failed, using max value as threshold.")
        threshold_value = map_max
    if threshold_value <= 1e-9 and map_max > 1e-9:
        threshold_value = 1e-9
    binary_mask = (np.nan_to_num(proc_map, nan=-np.inf) >= threshold_value).astype(np.uint8)
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
             mask_processed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    except cv2.error as e:
        print(f"Warning: OpenCV error during morphology '{morphology_op}': {e}. Using raw binary mask.")
        mask_processed = binary_mask
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

# --- save_visualizations ---
def save_visualizations(
    original_frame, mean_frame, interval_gradient_maps, activity_map,
    mask, index_prefix, save_dir, orig_colormap_enum, grad_colormap_enum
):
    # (Keep the function from the previous version, ensuring _normalize_save is correct)
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created visualization directory: {save_dir}")

        num_saved = 0
        h, w = (original_frame.shape[:2]) if original_frame is not None else (240, 320)
        blank_img = np.zeros((h, w), dtype=np.uint8)

        # --- Helper to normalize and save (Corrected) ---
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
                    # Corrected normalize call
                    norm_img = cv2.normalize(img_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=valid_mask_cv)

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
        # print(f"  Saving visualization images with prefix: {index_prefix}...") # Reduce verbosity

        # 1. RAW Original Frame Representations
        orig_gray_for_display = None
        orig_color_for_display = None
        if original_frame is not None:
            try:
                # Use the corrected _normalize_save for consistency in saving uint8 representations
                orig_gray_for_display = _normalize_save(original_frame, f"{index_prefix}_1_original_raw", "Raw Original Frame")
                if orig_gray_for_display is not None:
                    raw_orig_color = cv2.applyColorMap(orig_gray_for_display, orig_colormap_enum) # Apply colormap to normalized gray
                    orig_color_path = os.path.join(save_dir, f"{index_prefix}_1_original_raw_color.png")
                    cv2.imwrite(orig_color_path, raw_orig_color)
                    num_saved += 1
                    orig_color_for_display = raw_orig_color
            except Exception as e:
                print(f"  Error processing/saving raw original frame representations: {e}")

        # 2. NORMALIZED Original Frame
        vis_orig_norm = _normalize_save(original_frame, f"{index_prefix}_2_original_norm", "Normalized Original Frame", orig_colormap_enum)
        vis_orig_norm = vis_orig_norm if vis_orig_norm is not None else blank_img.copy()

        # 3. Mean Frame
        # Make sure mean_frame is defined even if calculation failed earlier
        if 'mean_frame' not in locals(): mean_frame = None # Define if potentially not defined
        _normalize_save(mean_frame, f"{index_prefix}_3_mean_frame", "Mean Frame", orig_colormap_enum)

        # 4. Interval Gradient Maps (if calculated and relevant)
        if interval_gradient_maps and activity_map is not None and not index_prefix.split('_')[2] == 'slope': # Adjusted index check
             for i, grad_map in enumerate(interval_gradient_maps):
                 if grad_map is not None:
                     _normalize_save(grad_map, f"{index_prefix}_4_interval_{i}_grad", f"Interval {i} Gradient", grad_colormap_enum)

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
             except Exception as e: 
                 print(f"Error while overlaying: {e}")

        print(f"Saved {num_saved} visualization images to '{save_dir}'")

    except Exception as e:
        print(f"!!! Error during visualization saving: {e}")
        traceback.print_exc()


# --- Main Execution Logic ---
def run_single_visualization(mat_file_path, output_base_dir, mat_key,
                             interval_size, quantile,
                             combination_strategy, morphology_op,
                             apply_blur, blur_kernel_size,
                             focus_duration_sec, fps,
                             envir_para, errorweight, augment,
                             smooth_window, fuselevel, normalizeT,
                             orig_cmap_enum, grad_cmap_enum):
    print("-" * 50)
    print(f"Processing: {mat_file_path}")
    # (Keep print statements for settings)
    print(f"  Focus Duration: {focus_duration_sec}s | Strategy: {combination_strategy} | Quantile: {quantile:.3f}")
    print(f"  Morphology: {morphology_op} | Blur: {apply_blur} | Spatial Fuse: {fuselevel} | Temporal Smooth: {smooth_window}")
    print(f"  NormalizeT: {normalizeT} | Envir: {envir_para} | ErrWeight: {errorweight} | Augment: {augment}")

    # --- Load Data ---
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        if mat_key not in mat_data: raise KeyError(f"Key '{mat_key}' not found in {mat_file_path}.")
        frames_raw = mat_data[mat_key].astype(np.float64)
        if not isinstance(frames_raw, np.ndarray): raise TypeError("Frame data is not a numpy array.")
        if frames_raw.ndim != 3: raise ValueError(f"Expected 3D array, got {frames_raw.ndim}D.")
        if frames_raw.shape[2] < 2: raise ValueError(f"Need >= 2 frames, got {frames_raw.shape[2]}.")
        print(f"  Loaded frames: {frames_raw.shape}, dtype: {frames_raw.dtype}")
        num_frames = frames_raw.shape[2]
        H, W = frames_raw.shape[:2]
    except Exception as e: print(f"Error loading data: {e}"); traceback.print_exc(); return False

    # --- Define mean_frame early, handle potential error ---
    mean_frame = compute_mean_frame(frames_raw) # Calculate mean on raw data before potential modifications

    # --- 1. Pre-processing ---
    frames = frames_raw.copy()
    if normalizeT:
        print("  Applying per-frame normalization...")
        frame_means = np.mean(frames, axis=(0, 1), keepdims=True)
        frame_means[frame_means == 0] = 1.0
        frames /= frame_means
    # else: print("  Using raw temperature values.") # Reduce verbosity

    if fuselevel > 0:
        print(f"  Applying spatial fuse (level {fuselevel})...")
        kernel_size = 2 * fuselevel + 1
        try:
            frames_fused = np.zeros_like(frames)
            for i in range(num_frames): frames_fused[:, :, i] = cv2.boxFilter(frames[:, :, i], -1, (kernel_size, kernel_size))
            frames = frames_fused
        except Exception as e: print(f"Warning: Spatial fuse failed: {e}")

    # --- Calculate Focus Window ---
    focus_duration_frames = int(focus_duration_sec * fps)
    focus_duration_frames = max(2, min(focus_duration_frames, num_frames))
    print(f"  Processing first {focus_duration_sec}s => {focus_duration_frames} frames.")
    t_focus = (np.arange(focus_duration_frames) / fps).astype(np.float64)
    frames_focus = frames[:, :, :focus_duration_frames]

    # --- Define ROI Mask ---
    border_h_roi = int(H * 0.20); border_w_roi = int(W * 0.20)
    roi_mask = np.zeros((H, W), dtype=bool)
    roi_mask[border_h_roi:-border_h_roi, border_w_roi:-border_w_roi] = True

    # --- 2. Determine Activity Map ---
    activity_map = None
    interval_gradient_maps = []

    if combination_strategy == 'slope':
        print(f"  Calculating weighted slope map (Smooth Window: {smooth_window})...")
        slope_map = np.zeros((H, W), dtype=np.float64)
        # slope_error_map = np.zeros((H, W), dtype=np.float64) + np.inf # Not strictly needed if not saving error map
        final_activity_map = np.zeros((H, W), dtype=np.float64)

        # Pre-calculate time vector for smoothing if needed
        if smooth_window > 1:
             # Use 'valid' mode, which shortens the output. Calculate length.
             smooth_len = len(np.convolve(np.ones(focus_duration_frames), np.ones(smooth_window), mode='valid'))
             if smooth_len < 2:
                  print(f"ERROR: Smoothing window ({smooth_window}) too large for focus duration ({focus_duration_frames}). Cannot proceed with slope.")
                  return False
             t_smoothed = t_focus[:smooth_len] # Time vector matching smoothed data length
        else:
             t_smoothed = t_focus # No smoothing, use original time vector

        # Iterate only within ROI to save time
        roi_indices = np.argwhere(roi_mask) # Get indices where ROI is True

        for idx in tqdm(range(roi_indices.shape[0]), desc="Calculating Slope (ROI)"):
            r, c = roi_indices[idx] # Get row, col from ROI list
            y_raw = frames_focus[r, c, :]

            if smooth_window > 1:
                # Ensure convolution input length matches smooth_window requirement
                if len(y_raw) >= smooth_window:
                     y = np.convolve(y_raw, np.ones(smooth_window)/smooth_window, mode='valid')
                else: # Should not happen if focus_duration_frames >= 2 and smooth_window >= 1, but safety
                     y = y_raw # Skip smoothing if data is shorter than window
                     t_smoothed = t_focus # Revert time vector
            else:
                y = y_raw
                # t_smoothed is already t_focus

            # Check length again after potential smoothing
            if len(y) < 2: continue

            # --- Corrected Try/Except Block ---
            try:
                if not np.all(np.isfinite(y)): continue
                # Use the potentially shortened t_smoothed vector
                res = sp_stats.linregress(t_smoothed[:len(y)], y) # Ensure time vector matches y length
                if np.isfinite(res.slope) and np.isfinite(res.stderr):
                    s = res.slope; se = res.stderr
                    # slope_map[r, c] = s # Not strictly needed unless saving slope map
                    # slope_error_map[r, c] = se # Not strictly needed unless saving error map
                    if s * envir_para < 0:
                        if abs(s) > 1e-9:
                            rel_err = abs(se / s)
                            weight = np.exp(-rel_err * errorweight)
                            final_activity_map[r, c] = (abs(s) * weight)**augment
            except ValueError as ve: # Handles linregress specific value errors
                pass # Keep activity 0
            except Exception as e_lr: # Catch other unexpected errors
                print(f"Warning: Error in linregress/weighting for pixel ({r},{c}): {e_lr}")
                pass # Keep activity 0
            # --- End Try/Except Block ---

        activity_map = final_activity_map

    else: # --- Interval Strategies ---
        print(f"  Calculating interval maps (Interval Size: {interval_size})...")
        num_intervals_in_focus = (focus_duration_frames + interval_size - 1) // interval_size
        for i in range(num_intervals_in_focus):
            start_frame = i * interval_size
            end_frame = min(start_frame + interval_size, focus_duration_frames)
            if start_frame >= focus_duration_frames - 1: continue
            # Use potentially pre-processed frames_focus
            interval_map = compute_interval_gradient_map(frames_focus, start_frame, end_frame)
            if interval_map is not None: interval_gradient_maps.append(interval_map)

        if not interval_gradient_maps: print("Error: No interval maps calculated."); return False

        print(f"  Combining {len(interval_gradient_maps)} maps using '{combination_strategy}'...")
        valid_maps = [m for m in interval_gradient_maps if m is not None and m.shape == interval_gradient_maps[0].shape]
        if not valid_maps: print("Error: No valid maps to combine."); return False

        try:
            if combination_strategy == 'mean': activity_map = np.nanmean(np.stack(valid_maps, axis=0), axis=0)
            elif combination_strategy == 'max': activity_map = np.nanmax(np.stack(valid_maps, axis=0), axis=0)
            elif combination_strategy == 'first_interval': activity_map = valid_maps[0]
            else: print(f"Internal Error: Bad strategy '{combination_strategy}'."); return False
        except Exception as e: print(f"Error combining maps: {e}"); traceback.print_exc(); return False

    # Check activity map validity
    if activity_map is None or not np.any(np.isfinite(activity_map)):
         print("  Error: Activity map invalid after calculation.")
         # Optionally save intermediate maps here for debugging
         return False

    # --- 3. Extract Hotspot ---
    print(f"  Extracting hotspot (Quantile: {quantile:.3f}, Morph: {morphology_op}, Blur: {apply_blur})...")
    # Apply ROI mask during extraction
    hotspot_mask, hotspot_area = extract_hotspot_from_map(
        activity_map, quantile, morphology_op, apply_blur, blur_kernel_size, roi_mask=roi_mask)

    if hotspot_mask is None: print("Warning: Hotspot extraction failed.")
    elif hotspot_area > 0: print(f"  Hotspot area: {hotspot_area:.0f} pixels")
    else: print("Warning: Hotspot area is 0 pixels.")

    # --- Prepare Output Path ---
    # (Keep the path logic from before)
    try:
        mat_filename = os.path.basename(mat_file_path)
        mat_filename_no_ext, _ = os.path.splitext(mat_filename)
        source_folder_name = os.path.basename(os.path.dirname(mat_file_path))
        if not source_folder_name: source_folder_name = "_root_"
        final_save_dir = os.path.join(output_base_dir, "focused_images", source_folder_name, mat_filename_no_ext)
    except Exception as e: print(f"Error constructing output path: {e}"); traceback.print_exc(); return False

    # --- Save Visualizations ---
    first_frame = frames_raw[:, :, 0] if frames_raw.shape[2] > 0 else None
    blur_suffix = f"_blur{blur_kernel_size[0]}x{blur_kernel_size[1]}" if apply_blur else "_noblur"
    fuse_suffix = f"_fuse{fuselevel}" if fuselevel > 0 else ""
    smooth_suffix = f"_smooth{smooth_window}" if combination_strategy == 'slope' and smooth_window > 1 else ""
    norm_suffix = "_normT" if normalizeT else ""
    vis_prefix = f"q{quantile:.3f}_dur{focus_duration_sec}s_{combination_strategy}_{morphology_op}{blur_suffix}{fuse_suffix}{smooth_suffix}{norm_suffix}"
    if combination_strategy == 'slope': vis_prefix += f"_err{errorweight}_aug{augment}"

    save_visualizations(
        original_frame=first_frame, mean_frame=mean_frame,
        interval_gradient_maps=interval_gradient_maps, activity_map=activity_map,
        mask=hotspot_mask, index_prefix=vis_prefix, save_dir=final_save_dir,
        orig_colormap_enum=orig_cmap_enum, grad_colormap_enum=grad_cmap_enum
    )
    print("-" * 50)
    return True


# --- Command-Line Argument Parsing (keep as before) ---
if __name__ == "__main__":
    # (Keep the argument parsing and validation logic from the previous version)
    # Build available colormap dictionary robustly
    colormap_choices = {name: val for name, val in cv2.__dict__.items() if name.startswith('COLORMAP_')}
    valid_cv2_colormaps = {}
    for name, enum_val in colormap_choices.items():
         simple_name = name.replace('COLORMAP_', '').lower()
         valid_cv2_colormaps[simple_name] = enum_val

    parser = argparse.ArgumentParser(
        description="Visualize dynamic focus area using slope or interval gradients within an initial time window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input / Output
    parser.add_argument("mat_file", help="Path to the input .mat file.")
    parser.add_argument("output_dir", help="Path to the BASE output directory.")
    parser.add_argument("-k", "--key", default="TempFrames", help="Key in the .mat file for frame data.")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second of the original recording (used for time calculations).")

    # Focus Window & Strategy
    parser.add_argument("--focus_duration_sec", type=float, default=5.0,
                        help="Duration (in seconds) from the start of the video to use for generating the activity map.")
    parser.add_argument("--cm", "--combination_strategy", default="slope", choices=['mean', 'max', 'first_interval', 'slope'], dest='combination_strategy',
                        help="Strategy for activity map generation.")
    parser.add_argument("-i", "--interval_size", type=int, default=50,
                        help="Frames per interval (used only if --cm is 'mean'/'max'/'first_interval').")

    # Pre-processing (MATLAB style)
    parser.add_argument("--normalizeT", type=int, default=0, choices=[0, 1],
                        help="Normalize temperature per frame? (0=No, 1=Yes).")
    parser.add_argument("--fuselevel", type=int, default=0,
                        help="Spatial fuse level (box filter radius before analysis, 0=None).")
    parser.add_argument("--smooth_window", type=int, default=3,
                        help="Temporal smoothing window size (moving average before slope fit, use odd number >= 1). Only used if --cm is 'slope'.")

    # Hotspot Extraction Parameters
    parser.add_argument("-q", "--quantile", type=float, default=0.98,
                        help="Quantile threshold (0-1) for hotspot extraction.")
    parser.add_argument("--morph_op", default="none", choices=['close', 'open_close', 'none'],
                        help="Morphological operation on thresholded mask.")
    parser.add_argument("--blur", action='store_true', help="Apply Gaussian blur to activity map before thresholding.")
    parser.add_argument("--blur_kernel", type=int, nargs=2, default=[3, 3], metavar=('H', 'W'),
                        help="Kernel size for Gaussian blur (use odd numbers). Used only if --blur.")

    # Slope Weighting Parameters (MATLAB style, only used if --cm is 'slope')
    parser.add_argument("--envir_para", type=int, default=-1, choices=[-1, 1],
                        help="Environment parameter (-1=Winter/cooling leak, 1=Summer/heating leak).")
    parser.add_argument("--errorweight", type=float, default=0.5,
                        help="Weighting factor for slope standard error.")
    parser.add_argument("--augment", type=float, default=1.0,
                        help="Exponent for augmenting the final weighted slope value.")

    # Visualization
    parser.add_argument("-c", "--colormap", default="hot", choices=list(valid_cv2_colormaps.keys()), help="Colormap for original/mean image.")
    parser.add_argument("-g", "--grad_colormap", default="inferno", choices=list(valid_cv2_colormaps.keys()), help="Colormap for gradient/activity map.")

    if len(sys.argv) < 3: parser.print_help(); sys.exit(1)
    args = parser.parse_args()

    # --- Argument Validation ---
    if not 0.0 < args.quantile <= 1.0: print("Error: Quantile must be > 0 and <= 1."); sys.exit(1)
    if args.focus_duration_sec <= 0: print("Error: Focus duration must be positive."); sys.exit(1)
    if args.fps <= 0: print("Error: FPS must be positive."); sys.exit(1)
    if args.combination_strategy != 'slope' and args.interval_size <= 0: print("Error: Interval size must be positive."); sys.exit(1)
    if args.blur:
        blur_k_h, blur_k_w = args.blur_kernel
        if blur_k_h <= 0 or blur_k_w <= 0: print("Error: Blur kernel dimensions must be positive."); sys.exit(1)
        if blur_k_h % 2 == 0 or blur_k_w % 2 == 0: print("Warning: Blur kernel dimensions should ideally be odd.")
    if args.fuselevel < 0: print("Error: Fuse level cannot be negative."); sys.exit(1)
    if args.smooth_window < 1: print("Error: Smooth window must be at least 1."); sys.exit(1)
    if args.errorweight < 0: print("Error: Error weight cannot be negative."); sys.exit(1)
    if args.augment <= 0: print("Error: Augment exponent must be positive."); sys.exit(1)

    selected_orig_cmap_enum = valid_cv2_colormaps[args.colormap]
    selected_grad_cmap_enum = valid_cv2_colormaps[args.grad_colormap]

    success = run_single_visualization(
        mat_file_path=args.mat_file, output_base_dir=args.output_dir, mat_key=args.key,
        interval_size=args.interval_size, quantile=args.quantile,
        combination_strategy=args.combination_strategy, morphology_op=args.morph_op,
        apply_blur=args.blur, blur_kernel_size=tuple(args.blur_kernel),
        focus_duration_sec=args.focus_duration_sec, fps=args.fps,
        envir_para=args.envir_para, errorweight=args.errorweight, augment=args.augment,
        smooth_window=args.smooth_window, fuselevel=args.fuselevel, normalizeT=args.normalizeT,
        orig_cmap_enum=selected_orig_cmap_enum, grad_cmap_enum=selected_grad_cmap_enum
    )

    if not success: print("Script finished with errors."); sys.exit(1)
    else: print("Script finished successfully."); sys.exit(0)