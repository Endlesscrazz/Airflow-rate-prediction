# visualize_focus_area_standalone.py
"""
Standalone script to visualize the DYNAMIC focus area for a single thermal .mat file,
based on COMBINED or selected INTERVAL GRADIENT maps, with tunable morphology.

Workflow:
1. Calculate gradient map for each specified time interval.
2. Combine interval maps based on selected strategy (mean, max, first).
3. Identify hotspot from the resulting activity map using quantile thresholding and selected morphology.
4. Save diagnostic images: original, mean, interval grads, activity map, mask, overlay.

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
    if frames is None or not isinstance(frames, np.ndarray) or frames.ndim != 3 or frames.shape[2] == 0: return None
    try: return np.mean(frames, axis=2, dtype=np.float64)
    except Exception as e: print(f"Error computing mean frame: {e}"); return None

# --- Modified to calculate map for ONE Interval ---
def compute_interval_gradient_map(frames, start_frame, end_frame):
    """
    Calculates a 2D map of mean absolute temporal gradient per pixel
    ONLY for frames within the specified interval [start_frame, end_frame).
    """
    if frames is None or frames.ndim != 3: return None
    total_frames_available = frames.shape[2]
    actual_start = max(0, start_frame)
    actual_end = min(total_frames_available, end_frame)

    if actual_end - actual_start < 2: return None # Need at least 2 frames in interval

    height, width = frames.shape[:2]
    gradient_sum = np.zeros((height, width), dtype=np.float64)
    valid_diff_count = 0
    for i in range(actual_start + 1, actual_end):
        try:
            # Ensure frames are float for subtraction
            frame_i = frames[:, :, i].astype(np.float64)
            frame_prev = frames[:, :, i-1].astype(np.float64)
            diff = np.abs(frame_i - frame_prev)
            gradient_sum += diff
            valid_diff_count += 1
        except IndexError: # Should be caught by loop range, but safety
             print(f"Warning: Index error accessing frame {i} or {i-1}.")
             continue
        except Exception as e:
             print(f"Warning: Error calculating diff frame {i} vs {i-1}: {e}")
             continue # Skip problematic frame pair

    if valid_diff_count == 0:
        # print(f"Warning: No valid diffs calculated for interval {start_frame}-{end_frame}")
        return None
    return gradient_sum / valid_diff_count

def compute_slope_map(frames, fps=30.0):
    T = frames.shape[2]
    dt = 1.0 / fps
    t = np.arange(T, dtype=np.float64) * dt
    t_mean = t.mean()
    var_t = ((t - t_mean)**2).sum()
    f = frames.astype(np.float64)
    mean_pix = f.mean(axis=2, keepdims=True)
    diff = f - mean_pix
    cov = np.tensordot(diff, (t - t_mean), axes=([2], [0]))
    slopes = cov / var_t
    return np.abs(slopes)

# --- Extract hotspot from a generic activity map ---
# --- UPDATED to include morphology options ---
def extract_hotspot_from_map(activity_map, threshold_quantile=0.95, morphology_op='close'):
    """
    Extracts the hotspot mask from a 2D activity map.

    Args:
        activity_map (np.ndarray): The 2D map to analyze.
        threshold_quantile (float): Quantile for thresholding.
        morphology_op (str): Morphological operation ('close', 'open_close', 'none').

    Returns:
        tuple: (hotspot_mask, hotspot_area)
    """
    if activity_map is None or activity_map.size == 0: return None, np.nan
    proc_map = activity_map.copy(); nan_mask = np.isnan(proc_map)
    if np.all(nan_mask): return None, np.nan # All NaNs
    valid_pixels = proc_map[~nan_mask]
    if valid_pixels.size == 0: return np.zeros_like(activity_map, dtype=bool), 0.0 # All valid pixels were NaN?

    map_std = np.std(valid_pixels); map_max = np.max(valid_pixels)
    # Check for empty or flat map
    if map_max < 1e-9 or map_std < 1e-9: # Increased tolerance slightly
        # print("Warning: Activity map has near-zero variation. Hotspot detection may fail.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    # Calculate threshold
    try: threshold_value = np.percentile(valid_pixels, threshold_quantile * 100)
    except IndexError: threshold_value = map_max # Fallback if percentile fails

    # Ensure threshold isn't effectively zero if max is non-zero
    if threshold_value <= 1e-9 and map_max > 1e-9: threshold_value = 1e-9

    # --- Thresholding ---
    # Handle NaNs explicitly during thresholding
    binary_mask = (np.nan_to_num(proc_map, nan=-np.inf) >= threshold_value).astype(np.uint8)
    if not np.any(binary_mask):
        # print(f"Warning: Binary mask empty after thresholding activity map (quantile={threshold_quantile}).")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    # --- Morphological Operations ---
    mask_processed = binary_mask.copy() # Start with the thresholded mask
    kernel_size_dim = max(min(3, activity_map.shape[0]//20, activity_map.shape[1]//20), 1) # Adaptive kernel size
    kernel_close = np.ones((kernel_size_dim, kernel_size_dim), np.uint8)
    kernel_open = np.ones((2, 2), np.uint8) # Smaller kernel typically for opening

    try:
        if morphology_op == 'close':
            mask_processed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
            # print("  Applied MORPH_CLOSE")
        elif morphology_op == 'open_close':
            mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
            mask_processed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
            # print("  Applied MORPH_OPEN then MORPH_CLOSE")
        elif morphology_op == 'none':
            # print("  Skipped morphological operations")
            pass # mask_processed remains binary_mask
        else:
             print(f"Warning: Unknown morphology_op '{morphology_op}'. Defaulting to 'close'.")
             mask_processed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)

    except cv2.error as e:
        print(f"Warning: OpenCV error during morphology '{morphology_op}': {e}. Using raw binary mask.")
        mask_processed = binary_mask # Fallback to raw mask

    # --- Connected Components ---
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_processed, connectivity=8)
    except cv2.error as e:
        print(f"Error during connectedComponentsWithStats: {e}. Cannot extract hotspot.")
        return None, np.nan

    hotspot_mask = np.zeros_like(activity_map, dtype=bool); hotspot_area = 0.0
    if num_labels > 1: # num_labels includes the background (label 0)
        # Find the label index corresponding to the largest area (excluding background at index 0)
        if stats.shape[0] > 1: # Check if there are any foreground components
             largest_label_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) # Index within the foreground stats
             largest_label = largest_label_idx + 1 # Adjust index to match the 'labels' array
             hotspot_mask = (labels == largest_label)
             hotspot_area = np.sum(hotspot_mask) # Calculate area from the boolean mask
             # if hotspot_area == 0: print("Warning: Largest component mask (hotspot) empty after morphology.") # Reduce noise
        # else: print("Warning: No foreground components found after morphology.") # Reduce noise
    # else: print("Warning: No connected components found after morphology.") # Reduce noise

    return hotspot_mask, hotspot_area


# --- UPDATED Visualization Saving Function ---
def save_visualizations(
    original_frame, mean_frame, interval_gradient_maps, activity_map, # Changed combined_gradient_map -> activity_map
    mask, index_prefix, save_dir, orig_colormap_enum, grad_colormap_enum
):
    """
    Saves multiple visualization images, including interval maps, the final activity map,
    raw original frame representations, and overlays on both raw and normalized originals.
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created visualization directory: {save_dir}")

        num_saved = 0
        h, w = (original_frame.shape[:2]) if original_frame is not None else (240, 320) # Sensible default size
        blank_img = np.zeros((h, w), dtype=np.uint8)

        # --- Helper to normalize and save (primarily for mean, gradients, activity map) ---
        # This remains useful for seeing relative changes in those specific maps.
        def _normalize_save(img_data, filename_base, description, colormap=None):
            nonlocal num_saved
            if img_data is None: print(f"  Skipping save: {description} data is None."); return None
            try:
                d_min, d_max = np.nanmin(img_data), np.nanmax(img_data)
                is_flat = np.isnan(d_min) or np.isnan(d_max) or d_max <= d_min

                valid_mask_np = ~np.isnan(img_data)
                valid_mask_cv = valid_mask_np.astype(np.uint8)

                if is_flat:
                    norm_img = np.full(img_data.shape[:2], 0 if np.isnan(d_min) else 128, dtype=np.uint8)
                    # Commenting out warnings for less verbose output during successful runs
                    # if np.isnan(d_min): print(f"  Warning: All NaN data for {description}, saving black image.")
                    # else: print(f"  Warning: Flat data for {description}, saving gray image.")
                else:
                    norm_img = np.zeros(img_data.shape[:2], dtype=np.uint8)
                    cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dst=norm_img, dtype=cv2.CV_8U, mask=valid_mask_cv)

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
                    except Exception as cmap_e: print(f"  Warning: Failed applying colormap for {description}: {cmap_e}")
                return norm_img # Return normalized gray
            except Exception as e: print(f"  Error processing/saving normalized {description}: {e}"); traceback.print_exc(); return None

        # --- Start Saving ---
        print(f"  Saving visualization images with prefix: {index_prefix}...")

        # --- 1. Process and Save RAW Original Frame Representations ---
        orig_gray_for_display = None
        orig_color_for_display = None
        if original_frame is not None:
            try:
                # Normalize the RAW frame just for display/saving uint8 (preserves internal range)
                raw_orig_normalized = cv2.normalize(original_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Save Raw Original Grayscale
                orig_gray_path = os.path.join(save_dir, f"{index_prefix}_1_original_raw_gray.png")
                cv2.imwrite(orig_gray_path, raw_orig_normalized)
                num_saved += 1
                orig_gray_for_display = raw_orig_normalized # Keep for overlay

                # Save Raw Original Colorized
                raw_orig_color = cv2.applyColorMap(raw_orig_normalized, orig_colormap_enum)
                orig_color_path = os.path.join(save_dir, f"{index_prefix}_1_original_raw_color.png")
                cv2.imwrite(orig_color_path, raw_orig_color)
                num_saved += 1
                orig_color_for_display = raw_orig_color # Keep for overlay

            except Exception as e:
                print(f"  Error processing/saving raw original frame: {e}")
                traceback.print_exc()
        else:
            print("  Skipping raw original frame saving: original_frame is None.")

        # --- 2. Save NORMALIZED Original Frame (for comparison) ---
        # Use the helper function, but adjust filename
        vis_orig_norm = _normalize_save(original_frame, f"{index_prefix}_2_original_norm", "Normalized Original Frame", orig_colormap_enum)
        # Fallback to a blank image if normalization failed
        vis_orig_norm = vis_orig_norm if vis_orig_norm is not None else blank_img.copy()

        # --- 3. Save Mean Frame ---
        _normalize_save(mean_frame, f"{index_prefix}_3_mean_frame", "Mean Frame", orig_colormap_enum)

        # --- 4. Save Interval Gradient Maps ---
        if interval_gradient_maps:
            for i, grad_map in enumerate(interval_gradient_maps):
                _normalize_save(grad_map, f"{index_prefix}_4_interval_{i}_grad", f"Interval {i} Gradient", grad_colormap_enum)
        else: print("  Skipping save: No interval gradient maps.")

        # --- 5. Save Final Activity Map ---
        _normalize_save(activity_map, f"{index_prefix}_5_activity_map", "Activity Map", grad_colormap_enum)

        # --- 6. Save Hotspot Mask ---
        if mask is not None:
            try:
                mask_img = (mask.astype(np.uint8) * 255)
                mask_path = os.path.join(save_dir, f"{index_prefix}_6_hotspot_mask.png")
                cv2.imwrite(mask_path, mask_img)
                num_saved += 1
            except Exception as e: print(f"  Error saving hotspot mask: {e}")
        else: print("  Skipping mask saving: Hotspot mask is None.")

        # --- 7. Save Overlay on RAW Original Grayscale ---
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

        # --- 8. Save Overlay on RAW Original Colorized ---
        if mask is not None and orig_color_for_display is not None:
            try:
                overlay_img_raw_color = orig_color_for_display.copy() # Already BGR
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

        # --- 9. Save Overlay on NORMALIZED Original Grayscale (Previous Step 6 Behavior) ---
        if mask is not None and vis_orig_norm is not None:
             try:
                 vis_orig_norm_bgr = cv2.cvtColor(vis_orig_norm, cv2.COLOR_GRAY2BGR)
                 overlay_img_norm = vis_orig_norm_bgr.copy()
                 overlay_color = [0, 255, 255]; # Yellow BGR
                 overlay_img_norm[mask] = overlay_color
                 alpha = 0.4;
                 blended_img = cv2.addWeighted(overlay_img_norm, alpha, vis_orig_norm_bgr, 1 - alpha, 0)
                 # Renamed file
                 overlay_path = os.path.join(save_dir, f"{index_prefix}_9_overlay_on_norm_gray.png")
                 cv2.imwrite(overlay_path, blended_img)
                 num_saved += 1
             except Exception as e: print(f"  Error saving overlay on normalized gray: {e}")
        # No need for explicit skips here as they are covered by previous steps

        print(f"Saved {num_saved} visualization images to '{save_dir}'")

    except Exception as e:
        print(f"!!! Error during visualization saving: {e}")
        traceback.print_exc()


# --- Main Execution Logic ---
def run_single_visualization(mat_file_path, output_base_dir, mat_key,
                             interval_size, max_intervals, quantile,
                             combination_strategy, morphology_op, # Added arguments
                             orig_cmap_enum, grad_cmap_enum):
    """Orchestrates loading, processing, and visualization based on interval gradients."""
    print("-" * 50)
    print(f"Processing: {mat_file_path}")
    print(f"  Settings: interval={interval_size}, max_intervals={max_intervals}, quantile={quantile:.2f}")
    print(f"  Strategy: combination='{combination_strategy}', morphology='{morphology_op}'")

    # --- Load Data ---
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        if mat_key not in mat_data: raise KeyError(f"Key '{mat_key}' not found.")
        frames = mat_data[mat_key]
        # Perform basic validation on frames
        if not isinstance(frames, np.ndarray): raise TypeError("Frame data is not a numpy array.")
        if frames.ndim != 3: raise ValueError(f"Expected 3D array (H, W, N), got {frames.ndim}D.")
        if frames.shape[2] < 2: raise ValueError(f"Need at least 2 frames, got {frames.shape[2]}.")
        print(f"  Loaded frames: {frames.shape}, dtype: {frames.dtype}")
        num_frames = frames.shape[2]
    except Exception as e: print(f"  Error loading data: {e}"); return False

    # --- Calculations ---
    # 1. Calculate mean frame (for comparison)
    mean_frame = compute_mean_frame(frames)

    # 2. Calculate Interval Gradient Maps
    interval_gradient_maps = []
    print("  Calculating interval gradient maps...")
    for i in range(max_intervals):
        start_frame = i * interval_size
        if start_frame >= num_frames - 1: # Stop if interval starts too late (needs frame i and i+1)
            print(f"  Stopping interval calculation at interval {i}, start_frame {start_frame} >= num_frames-1 {num_frames-1}")
            break
        end_frame = start_frame + interval_size # compute_interval_gradient_map handles end > num_frames
        interval_map = compute_interval_gradient_map(frames, start_frame, end_frame)
        if interval_map is not None:
            interval_gradient_maps.append(interval_map)
        else:
            # Be less verbose, only warn if needed
            # print(f"  Warning: Failed interval gradient map for {start_frame}-{end_frame}")
            pass

    if not interval_gradient_maps: print("  Error: No interval gradient maps calculated."); return False

    # 3. Combine Interval Maps based on Strategy
    print(f"  Combining {len(interval_gradient_maps)} interval map(s) using '{combination_strategy}' strategy...")
    activity_map = None
    valid_maps = [m for m in interval_gradient_maps if m is not None and m.shape == interval_gradient_maps[0].shape] # Ensure shape consistency

    if not valid_maps:
        print("  Error: No valid/consistent interval maps to combine.")
        return False

    try:
        if combination_strategy == 'mean':
            # Use nanmean for robustness if some intermediate calcs resulted in NaN
            activity_map = np.nanmean(np.stack(valid_maps, axis=0), axis=0)
        elif combination_strategy == 'max':
            # Use nanmax
            activity_map = np.nanmax(np.stack(valid_maps, axis=0), axis=0)
        elif combination_strategy == 'first_interval':
            activity_map = valid_maps[0] # Already validated existence and consistency
        elif combination_strategy == 'slope':
            print("  Calculating activity map using full duration slope...")
            activity_map = compute_slope_map(frames, fps=30.0) # Adjust fps if known/needed
            if activity_map is None:
                print("  Error: Failed to compute slope map.")
                return False
        else:
            print(f"  Error: Unknown combination_strategy '{combination_strategy}'.")
            return False
    except Exception as e:
        print(f"  Error combining maps using '{combination_strategy}': {e}")
        traceback.print_exc()
        return False

    if activity_map is None:
         print("  Error: Activity map is None after combination step.")
         return False

    # 4. Extract Hotspot from the Activity Map
    print(f"  Extracting hotspot from activity map (quantile={quantile:.2f}, morphology='{morphology_op}')...")
    hotspot_mask, hotspot_area = extract_hotspot_from_map(activity_map,
                                                          threshold_quantile=quantile,
                                                          morphology_op=morphology_op) # Pass morphology op
    # Check if mask extraction failed (returned None)
    if hotspot_mask is None:
         print("  Warning: Failed hotspot extraction (extract_hotspot_from_map returned None).")
         # We can still proceed to save visualizations of maps, just no mask/overlay
    else:
        print(f"  Hotspot area: {hotspot_area} pixels")


    # --- Prepare Output Path ---
    try:
        mat_filename = os.path.basename(mat_file_path)
        mat_filename_no_ext, _ = os.path.splitext(mat_filename)
        # Handle cases where the mat file might be directly in the base directory
        source_folder_rel_path = os.path.dirname(os.path.relpath(mat_file_path, start=os.path.dirname(output_base_dir))) # More robust pathing
        source_folder_name = os.path.basename(os.path.dirname(mat_file_path)) # Simpler fallback
        if not source_folder_name or source_folder_name == '.': source_folder_name = "_root_"

        final_save_dir = os.path.join(output_base_dir, "focused_images", source_folder_name, mat_filename_no_ext)
    except Exception as e: print(f"  Error constructing output path: {e}"); return False

    # --- Save Visualizations ---
    first_frame = frames[:, :, 0] if frames.shape[2] > 0 else None
    # --- UPDATED index_prefix ---
    vis_prefix = f"q{quantile:.2f}_{combination_strategy}_{morphology_op}"

    save_visualizations(
        original_frame=first_frame,
        mean_frame=mean_frame,
        interval_gradient_maps=interval_gradient_maps,
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
         # Extract simple name (e.g., 'jet' from 'COLORMAP_JET')
         simple_name = name.replace('COLORMAP_', '').lower()
         valid_cv2_colormaps[simple_name] = enum_val

    parser = argparse.ArgumentParser(
        description="Visualize dynamic focus area (from interval gradients) for a single thermal .mat file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Makes help text clearer
    )
    parser.add_argument("mat_file", help="Path to the input .mat file.")
    parser.add_argument("output_dir", help="Path to the BASE output directory (subdirs will be created).")
    parser.add_argument("-k", "--key", default="TempFrames", help="Key in the .mat file for frame data.")
    parser.add_argument("-q", "--quantile", type=float, default=0.95, help="Quantile threshold for hotspot extraction on activity map.") # Default 0.98 from feature_eng
    parser.add_argument("-i", "--interval_size", type=int, default=50, help="Number of frames per interval for gradient calculation.")
    parser.add_argument("-m", "--max_intervals", type=int, default=3, help="Maximum number of intervals to process.")
    # --- NEW ARGUMENTS ---
    parser.add_argument("--cm", default="mean", choices=['mean', 'max', 'first_interval','slope'],
                        help="Strategy to combine interval gradient maps into the activity map.")
    parser.add_argument("--morph_op", default="close", choices=['close', 'open_close', 'none'],
                        help="Morphological operation applied to the thresholded mask.")
    # --- Colormap Arguments ---
    parser.add_argument("-c", "--colormap", default="hot", choices=list(valid_cv2_colormaps.keys()), help="Colormap for original/mean image visualization.")
    parser.add_argument("-g", "--grad_colormap", default="inferno", choices=list(valid_cv2_colormaps.keys()), help="Colormap for gradient/activity map visualization.")

    # Check if enough arguments provided
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Validate quantile range
    if not 0.0 < args.quantile <= 1.0:
         print("Error: Quantile must be between 0 (exclusive) and 1 (inclusive).")
         sys.exit(1)

    # Get selected colormap enums
    selected_orig_cmap_enum = valid_cv2_colormaps[args.colormap]
    selected_grad_cmap_enum = valid_cv2_colormaps[args.grad_colormap]

    # Run the visualization process
    success = run_single_visualization(
        mat_file_path=args.mat_file,
        output_base_dir=args.output_dir,
        mat_key=args.key,
        interval_size=args.interval_size,
        max_intervals=args.max_intervals,
        quantile=args.quantile,
        combination_strategy=args.cm, # Pass new arg
        morphology_op=args.morph_op,             # Pass new arg
        orig_cmap_enum=selected_orig_cmap_enum,
        grad_cmap_enum=selected_grad_cmap_enum
    )

    if not success:
        print("Script finished with errors.")
        sys.exit(1)
    else:
        print("Script finished successfully.")
        sys.exit(0)