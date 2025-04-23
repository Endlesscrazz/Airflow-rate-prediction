# visualize_focus_area_standalone.py
"""
Standalone script to visualize the DYNAMIC focus area for a single thermal .mat file,
based on COMBINED INTERVAL GRADIENT maps.

Workflow:
1. Calculate gradient map for each specified time interval.
2. Combine interval maps (by averaging) into a single activity map.
3. Identify hotspot from the combined activity map using quantile thresholding.
4. Save diagnostic images: original, mean, interval grads, combined grad, mask, overlay.

Saves images into a structured output directory:
<output_base_dir>/focused_images/<source_folder_name>/<mat_file_name>/
"""

import numpy as np
import cv2
import os
import scipy.io
import argparse
import sys


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
            diff = np.abs(frames[:, :, i].astype(np.float64) - frames[:, :, i-1].astype(np.float64))
            gradient_sum += diff
            valid_diff_count += 1
        except Exception: continue # Skip problematic frame pair
    if valid_diff_count == 0: return None
    return gradient_sum / valid_diff_count

# --- Extract hotspot from a generic activity map ---
def extract_hotspot_from_map(activity_map, threshold_quantile=0.95):
    """Extracts the hotspot mask from a 2D activity map."""
    if activity_map is None or activity_map.size == 0: return None, np.nan
    proc_map = activity_map.copy(); nan_mask = np.isnan(proc_map)
    if np.all(nan_mask): return None, np.nan
    valid_pixels = proc_map[~nan_mask]
    if valid_pixels.size == 0: return np.zeros_like(activity_map, dtype=bool), 0.0

    map_std = np.std(valid_pixels); map_max = np.max(valid_pixels)
    if map_max < 1e-6 or map_std < 1e-6: return np.zeros_like(activity_map, dtype=bool), 0.0

    try: threshold_value = np.percentile(valid_pixels, threshold_quantile * 100)
    except IndexError: threshold_value = map_max
    
    if threshold_value <= 1e-6 and map_max > 1e-6: threshold_value = 1e-6
    binary_mask = (np.nan_to_num(proc_map, nan=-np.inf) >= threshold_value).astype(np.uint8)
    if not np.any(binary_mask): return np.zeros_like(activity_map, dtype=bool), 0.0
    kernel_size = max(min(3, activity_map.shape[0]//20, activity_map.shape[1]//20), 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    try: mask_clean = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    except cv2.error: mask_clean = binary_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    hotspot_mask = np.zeros_like(activity_map, dtype=bool); hotspot_area = 0.0
    if num_labels > 1:
        largest_label_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_label = largest_label_idx + 1
        hotspot_mask = (labels == largest_label)
        hotspot_area = np.sum(hotspot_mask)
    return hotspot_mask, hotspot_area


# --- UPDATED Visualization Saving Function ---
def save_visualizations(
    original_frame, mean_frame, interval_gradient_maps, combined_gradient_map,
    mask, index_prefix, save_dir, orig_colormap_enum, grad_colormap_enum
):
    """Saves multiple visualization images, including interval maps."""
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
            if img_data is None: print(f"  Skipping save: {description} data is None."); return
            try:
                d_min, d_max = np.nanmin(img_data), np.nanmax(img_data) # Use nanmin/max
                if np.isnan(d_min) or np.isnan(d_max): # Handle all NaN case
                     norm_img = np.zeros(img_data.shape[:2], dtype=np.uint8)
                     print(f"  Warning: All NaN data for {description}, saving black image.")
                elif d_max > d_min:
                    norm_img = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                else: norm_img = np.full(img_data.shape[:2], 128, dtype=np.uint8) # Flat image -> gray

                path_gray = os.path.join(save_dir, f"{filename_base}_gray.png")
                cv2.imwrite(path_gray, norm_img)
                num_saved += 1

                if colormap is not None:
                    try:
                        # Apply colormap to the *normalized* grayscale image
                        color_img = cv2.applyColorMap(norm_img, colormap)
                        path_color = os.path.join(save_dir, f"{filename_base}_color.png")
                        cv2.imwrite(path_color, color_img)
                        num_saved += 1
                    except Exception as cmap_e: print(f"  Warning: Failed applying colormap for {description}: {cmap_e}")
                return norm_img # Return normalized gray for overlay use
            except Exception as e: print(f"  Error processing/saving {description}: {e}"); return None


        # --- Save Diagnostic Images ---
        # 1. Original Frame
        vis_orig_norm = _normalize_save(original_frame, f"{index_prefix}_1_original", "Original Frame", orig_colormap_enum)
        vis_orig_norm = vis_orig_norm if vis_orig_norm is not None else blank_img # Fallback

        # 2. Mean Frame
        _normalize_save(mean_frame, f"{index_prefix}_2_mean_frame", "Mean Frame", orig_colormap_enum)

        # 3. Interval Gradient Maps
        if interval_gradient_maps:
            for i, grad_map in enumerate(interval_gradient_maps):
                _normalize_save(grad_map, f"{index_prefix}_3_interval_{i}_grad", f"Interval {i} Gradient", grad_colormap_enum)

        # 4. Combined Gradient Map
        _normalize_save(combined_gradient_map, f"{index_prefix}_4_combined_grad", "Combined Gradient", grad_colormap_enum)

        # 5. Hotspot Mask
        if mask is not None:
            try:
                mask_img = (mask.astype(np.uint8) * 255)
                mask_path = os.path.join(save_dir, f"{index_prefix}_5_hotspot_mask.png")
                cv2.imwrite(mask_path, mask_img)
                num_saved += 1
            except Exception as e: print(f"  Error saving hotspot mask: {e}")

        # 6. Hotspot Overlay (on Original Grayscale)
        if mask is not None and vis_orig_norm is not None:
             try:
                 vis_orig_bgr = cv2.cvtColor(vis_orig_norm, cv2.COLOR_GRAY2BGR)
                 overlay_img = vis_orig_bgr.copy()
                 overlay_color = [0, 255, 255]; overlay_img[mask] = overlay_color # Yellow
                 alpha = 0.4; blended_img = cv2.addWeighted(overlay_img, alpha, vis_orig_bgr, 1 - alpha, 0)
                 overlay_path = os.path.join(save_dir, f"{index_prefix}_6_hotspot_overlay.png")
                 cv2.imwrite(overlay_path, blended_img)
                 num_saved += 1
             except Exception as e: print(f"  Error saving overlay: {e}")

        print(f"Saved {num_saved} visualization images to '{save_dir}'")

    except Exception as e:
        print(f"!!! Error during visualization saving: {e}")
        import traceback; traceback.print_exc()


# --- Main Execution Logic ---
def run_single_visualization(mat_file_path, output_base_dir, mat_key,
                             interval_size, max_intervals, quantile,
                             orig_cmap_enum, grad_cmap_enum):
    """Orchestrates loading, processing, and visualization based on interval gradients."""
    print("-" * 50)
    print(f"Processing: {mat_file_path}")

    # --- Load Data ---
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        if mat_key not in mat_data: raise KeyError(f"Key '{mat_key}' not found.")
        frames = mat_data[mat_key]
        if not isinstance(frames, np.ndarray) or frames.ndim != 3 or frames.shape[2] < 2: # Need >= 2 frames
            raise ValueError("Invalid or insufficient frame data.")
        print(f"  Loaded frames: {frames.shape}")
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
        if start_frame >= num_frames - 1: break # Stop if interval starts too late
        end_frame = start_frame + interval_size
        interval_map = compute_interval_gradient_map(frames, start_frame, end_frame)
        if interval_map is not None:
            interval_gradient_maps.append(interval_map)
        else:
            print(f"  Warning: Failed interval gradient map for {start_frame}-{end_frame}")

    if not interval_gradient_maps: print("  Error: No interval gradient maps calculated."); return False

    # 3. Combine Interval Maps (Averaging)
    print(f"  Combining {len(interval_gradient_maps)} interval map(s)...")
    try:
        # Use nanmean to be robust if some maps failed and were None (though list should only contain arrays)
        # Filter out None values just in case before stacking
        valid_maps = [m for m in interval_gradient_maps if m is not None]
        if not valid_maps: raise ValueError("No valid maps to combine.")
        combined_gradient_map = np.mean(np.stack(valid_maps, axis=0), axis=0)
    except Exception as e: print(f"  Error combining maps: {e}"); return False

    # 4. Extract Hotspot from Combined Map
    print(f"  Extracting hotspot from combined map (quantile={quantile:.2f})...")
    hotspot_mask, hotspot_area = extract_hotspot_from_map(combined_gradient_map, threshold_quantile=quantile)
    if hotspot_mask is None: print("  Error: Failed hotspot extraction."); return False
    print(f"  Hotspot area: {hotspot_area} pixels")

    # --- Prepare Output Path ---
    try:
        mat_filename = os.path.basename(mat_file_path)
        mat_filename_no_ext, _ = os.path.splitext(mat_filename)
        source_folder_name = os.path.basename(os.path.dirname(mat_file_path))
        if not source_folder_name: source_folder_name = "_root_"
        final_save_dir = os.path.join(output_base_dir, "focused_images", source_folder_name, mat_filename_no_ext)
    except Exception as e: print(f"  Error constructing output path: {e}"); return False

    # --- Save Visualizations ---
    first_frame = frames[:, :, 0] if frames.shape[2] > 0 else None
    save_visualizations(
        original_frame=first_frame,
        mean_frame=mean_frame,
        interval_gradient_maps=interval_gradient_maps,
        combined_gradient_map=combined_gradient_map,
        mask=hotspot_mask,
        index_prefix=f"dynamic_{quantile:.2f}q", # Use descriptive prefix
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("mat_file", help="Path to the input .mat file.")
    parser.add_argument("output_dir", help="Path to the BASE output directory.")
    parser.add_argument("-k", "--key", default="TempFrames", help="Key in the .mat file for frame data.")
    parser.add_argument("-q", "--quantile", type=float, default=0.95, help="Quantile threshold for hotspot extraction on COMBINED gradient map.")
    parser.add_argument("-i", "--interval_size", type=int, default=50, help="Number of frames per interval for gradient calculation.")
    parser.add_argument("-m", "--max_intervals", type=int, default=3, help="Maximum number of intervals to process.")
    parser.add_argument("-c", "--colormap", default="hot", choices=list(valid_cv2_colormaps.keys()), help="Colormap for original image visualization.")
    parser.add_argument("-g", "--grad_colormap", default="inferno", choices=list(valid_cv2_colormaps.keys()), help="Colormap for gradient map visualization.")

    if len(sys.argv) < 3: parser.print_help(); sys.exit(1)
    args = parser.parse_args()

    selected_orig_cmap_enum = valid_cv2_colormaps[args.colormap]
    selected_grad_cmap_enum = valid_cv2_colormaps[args.grad_colormap]

    success = run_single_visualization(
        mat_file_path=args.mat_file,
        output_base_dir=args.output_dir,
        mat_key=args.key,
        interval_size=args.interval_size, # Pass interval args
        max_intervals=args.max_intervals, # Pass interval args
        quantile=args.quantile,
        orig_cmap_enum=selected_orig_cmap_enum,
        grad_cmap_enum=selected_grad_cmap_enum
    )

    if not success: print("Script finished with errors."); sys.exit(1)
    else: print("Script finished successfully."); sys.exit(0)