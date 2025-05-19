# hotspot_mask_generation.py
"""
Pre-processing script to generate and save hotspot masks for thermal .mat files.
Includes enhanced debugging for slope calculation.
"""

import numpy as np
import cv2
import os
import scipy.io
import argparse
import sys
import traceback
from scipy import stats as sp_stats
from tqdm import tqdm
import fnmatch

# --- Import from data_utils and config ---
try:
    import data_utils # Assuming this handles parsing airflow, delta_T etc.
    import config     # For MAT_FRAMES_KEY, DATASET_FOLDER, and MASK_ defaults
except ImportError:
    print("Error: Failed to import data_utils.py or config.py.")
    print("Please ensure these files are in the same directory or accessible in PYTHONPATH.")
    # Create mock config for basic script functionality if config is missing
    class MockConfig:
        MAT_FRAMES_KEY = 'TempFrames'
        DATASET_FOLDER = './dataset_cleaned' # Placeholder
        MASK_FPS = 5.0
        MASK_FOCUS_DURATION_SEC = 5.0
        MASK_SMOOTH_WINDOW = 1
        MASK_P_VALUE_THRESHOLD = 0.10
        ENVIR_PARA = 1 # Default based on recent discussion
        MASK_AUGMENT_SLOPE = 1.0
        MASK_NORMALIZE_TEMP_FRAMES = False
        MASK_FUSE_LEVEL = 0
        MASK_ROI_BORDER_PERCENT = 0.0
        MASK_ACTIVITY_QUANTILE = 0.99
        MASK_MORPHOLOGY_OP = 'none'
        MASK_APPLY_BLUR = False
        MASK_BLUR_KERNEL_SIZE = (3,3)

    config = MockConfig()
    print("Warning: Using mock config values as config.py was not found.")


# --- Helper Functions ---
def apply_preprocessing(frames, normalizeT, fuselevel):
    frames_proc = frames.copy()
    if normalizeT:
        # Avoid division by zero if a frame is all zeros
        frame_means = np.mean(frames_proc, axis=(0, 1), keepdims=True)
        # Add a small epsilon to prevent division by zero if mean is exactly 0
        frame_means[frame_means == 0] = 1e-9
        frames_proc = frames_proc / frame_means # Element-wise division
    if fuselevel > 0:
        kernel_size = 2 * fuselevel + 1
        try:
            num_frames_in_stack = frames_proc.shape[2] # Assuming frames are (H, W, NumFrames)
            frames_fused = np.zeros_like(frames_proc)
            for i in range(num_frames_in_stack):
                frames_fused[:, :, i] = cv2.boxFilter(
                    frames_proc[:, :, i], -1, (kernel_size, kernel_size))
            frames_proc = frames_fused
        except Exception as e:
            print(f"Warning: Spatial fuse failed: {e}")
    return frames_proc

# --- Activity Map Calculation with P-value Filter & Debug Prints ---
def calculate_filtered_slope_activity_map(frames_focus, fps, smooth_window,
                                          envir_para, augment, p_value_threshold,
                                          roi_mask=None, debug_pixel=None):
    H, W, T_focus = frames_focus.shape
    final_activity_map = np.zeros((H, W), dtype=np.float64)
    processed_pixel_count = 0
    passed_filter_count = 0

    if T_focus < 2: # Need at least 2 points for linregress
        print("Error: Less than 2 time points for slope calculation.")
        return None, 0, 0 # Return counts

    # Temporal smoothing
    if smooth_window > 1:
        frames_focus_smoothed = np.zeros_like(frames_focus)
        for r_idx in range(H):
            for c_idx in range(W):
                # Ensure there are enough points for convolution mode 'valid'
                if T_focus >= smooth_window:
                    smoothed_series = np.convolve(frames_focus[r_idx, c_idx, :], np.ones(
                        smooth_window)/smooth_window, mode='valid')
                    # Pad to keep original length for easier indexing, using edge padding
                    pad_len_before = (T_focus - len(smoothed_series)) // 2
                    pad_len_after = T_focus - len(smoothed_series) - pad_len_before
                    if len(smoothed_series) > 0: # Should always be true if T_focus >= smooth_window
                        frames_focus_smoothed[r_idx, c_idx, :] = np.pad(
                            smoothed_series, (pad_len_before, pad_len_after), mode='edge')
                    else: # Should not happen if T_focus >= smooth_window
                        frames_focus_smoothed[r_idx, c_idx, :] = frames_focus[r_idx, c_idx, :] # Fallback
                else: # Not enough points to smooth with the given window
                    frames_focus_smoothed[r_idx, c_idx, :] = frames_focus[r_idx, c_idx, :]
        t_smoothed = (np.arange(T_focus) / fps).astype(np.float64)
    else:
        frames_focus_smoothed = frames_focus.copy() # Use a copy to avoid modifying original
        t_smoothed = (np.arange(T_focus) / fps).astype(np.float64)


    # Determine iteration indices
    iterator_desc = "Calculating Filtered Slope"
    if roi_mask is not None:
        iterator_desc += " (ROI)"
        # Ensure roi_mask is boolean
        if roi_mask.dtype != bool:
            roi_mask = roi_mask.astype(bool)
        roi_indices = np.argwhere(roi_mask)
        iterator_range = range(roi_indices.shape[0])
        def get_coords(idx_iter): return roi_indices[idx_iter]
    else:
        iterator_desc += " (Full)"
        total_pixels_to_process = H * W
        iterator_range = range(total_pixels_to_process)
        def get_coords(idx_iter): return np.unravel_index(idx_iter, (H, W))

    for idx_loop in tqdm(iterator_range, desc=iterator_desc, leave=False, ncols=100):
        r, c = get_coords(idx_loop)
        processed_pixel_count += 1

        y_pixel_series = frames_focus_smoothed[r, c, :]
        valid_mask_y = np.isfinite(y_pixel_series) # Mask for finite values in y
        y_valid = y_pixel_series[valid_mask_y]
        t_valid = t_smoothed[valid_mask_y] # Use corresponding time points

        if len(y_valid) < 2: # Need at least 2 valid points for linregress
            if debug_pixel and r == debug_pixel[0] and c == debug_pixel[1]:
                print(f"  DEBUG PIXEL ({r},{c}): Skipped - Not enough valid data points ({len(y_valid)}) after NaN filter.")
            continue
        try:
            res = sp_stats.linregress(t_valid, y_valid)
            slope_ok = False
            p_value_ok = False

            if np.isfinite(res.slope):
                # envir_para = 1 (heating expected): slope > 0
                # envir_para = -1 (cooling expected): slope < 0
                # (res.slope * envir_para > 0) checks this
                if (res.slope * envir_para > 0):
                    slope_ok = True

            if np.isfinite(res.pvalue):
                if res.pvalue < p_value_threshold:
                    p_value_ok = True
            
            if debug_pixel and r == debug_pixel[0] and c == debug_pixel[1]:
                print(f"\n  DEBUG PIXEL ({r},{c}):")
                print(f"    Raw y_focus: {frames_focus[r,c,:5]}... (len {T_focus})")
                print(f"    Smoothed y: {y_pixel_series[:5]}... (len {len(y_pixel_series)})")
                print(f"    y_valid (for linregress): {y_valid[:5]}... (len {len(y_valid)})")
                print(f"    t_valid (for linregress): {t_valid[:5]}... (len {len(t_valid)})")
                print(f"    Slope: {res.slope:.4f}, P-value: {res.pvalue:.4f}")
                print(f"    envir_para: {envir_para}, p_value_threshold: {p_value_threshold}")
                print(f"    Slope OK (direction matches envir_para)? {slope_ok}")
                print(f"    P-value OK (significant)? {p_value_ok}")

            if slope_ok and p_value_ok:
                activity_value = np.abs(res.slope)
                final_activity_map[r, c] = np.power(
                    activity_value, augment) if augment != 1.0 else activity_value
                passed_filter_count += 1
                if debug_pixel and r == debug_pixel[0] and c == debug_pixel[1]:
                    print(f"    PASSED FILTERS! Activity map value set to: {final_activity_map[r,c]:.4f}")
            elif debug_pixel and r == debug_pixel[0] and c == debug_pixel[1]:
                 print(f"    FAILED FILTERS.")


        except ValueError as ve: # linregress can raise ValueError for insufficient data
            if debug_pixel and r == debug_pixel[0] and c == debug_pixel[1]:
                print(f"  DEBUG PIXEL ({r},{c}): Linregress ValueError: {ve}")
        except Exception as e_lr: # Catch other potential errors during regression
            if debug_pixel and r == debug_pixel[0] and c == debug_pixel[1]:
                print(f"  DEBUG PIXEL ({r},{c}): Linregress/filter unexpected error: {e_lr}")
            # Optionally, log these errors more formally if they are frequent
    
    print(f"  Slope calculation: Processed {processed_pixel_count} pixels, {passed_filter_count} passed filters.")
    return final_activity_map, processed_pixel_count, passed_filter_count


def extract_hotspot_from_map(activity_map, threshold_quantile=0.95, morphology_op='close',
                             apply_blur=False, blur_kernel_size=(3, 3),
                             roi_mask=None): # roi_mask here is for processing the activity_map
    if activity_map is None or activity_map.size == 0:
        print("Warning: Activity map is None or empty in extract_hotspot_from_map.")
        return np.array([]), 0.0 # Return empty array and 0 area

    map_to_process = activity_map.copy()
    # Apply ROI to the activity map if provided
    if roi_mask is not None:
        if roi_mask.shape == activity_map.shape:
            if roi_mask.dtype != bool:
                roi_mask = roi_mask.astype(bool)
            map_to_process[~roi_mask] = np.nan # Set non-ROI areas to NaN
        else:
            print("Warning: ROI mask shape mismatch in extract_hotspot. Ignoring ROI for extraction.")

    # Blur if requested
    if apply_blur:
        # Check if map_to_process is all NaN (e.g., if ROI excluded everything or map was empty)
        if np.all(np.isnan(map_to_process)):
            print("Warning: Activity map is all NaN after ROI application, skipping blur.")
        else:
            try:
                # GaussianBlur needs float32 and no NaNs for input
                map_for_blur_no_nan = np.nan_to_num(map_to_process.astype(np.float32), nan=0.0)
                k_h = blur_kernel_size[0] + (1 - blur_kernel_size[0] % 2) # Ensure odd
                k_w = blur_kernel_size[1] + (1 - blur_kernel_size[1] % 2) # Ensure odd
                blurred_map = cv2.GaussianBlur(map_for_blur_no_nan, (k_w, k_h), 0)
                # Put NaNs back where they were if ROI was applied
                if roi_mask is not None and roi_mask.shape == map_to_process.shape:
                    blurred_map[~roi_mask] = np.nan
                map_to_process = blurred_map
            except Exception as e:
                print(f"Warning: GaussianBlur failed: {e}. Using unblurred map (post-ROI).")

    # --- Debug print for activity map state before thresholding ---
    non_nan_pixels_before_thresh = map_to_process[~np.isnan(map_to_process)]
    if non_nan_pixels_before_thresh.size > 0:
        min_act = np.min(non_nan_pixels_before_thresh)
        max_act = np.max(non_nan_pixels_before_thresh)
        non_zero_count = np.sum(non_nan_pixels_before_thresh > 1e-9) # Count effectively non-zero
        print(f"Debug: Activity map before extraction - Min: {min_act:.2e}, Max: {max_act:.2e}, Non-zero count: {non_zero_count}")
    else:
        print("Debug: Activity map before extraction is all NaN or empty.")


    # Thresholding
    nan_mask = np.isnan(map_to_process)
    if np.all(nan_mask): # If all values are NaN (e.g. ROI excluded everything or map was bad)
        print("Warning: Activity map all NaNs after ROI/blur in extract_hotspot_from_map.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    valid_pixels_for_threshold = map_to_process[~nan_mask]
    if valid_pixels_for_threshold.size == 0:
        print("Warning: No valid pixels found in activity map for thresholding.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    map_std_val = np.std(valid_pixels_for_threshold)
    map_max_val = np.max(valid_pixels_for_threshold)

    # If max activity is effectively zero or there's no variation, return empty mask
    if map_max_val < 1e-9 or (map_std_val < 1e-9 and valid_pixels_for_threshold.size > 1):
        print("Warning: Max activity is near zero or no variation in activity map.")
        return np.zeros_like(activity_map, dtype=bool), 0.0
    
    try:
        threshold_value = np.percentile(valid_pixels_for_threshold, threshold_quantile * 100)
    except IndexError: # Should not happen if valid_pixels_for_threshold is not empty
        threshold_value = map_max_val
    
    print(f"  Threshold value for activity map: {threshold_value:.4e} (Quantile: {threshold_quantile*100}%)")


    # Ensure threshold is not too low if max is significant, but also not too high
    if threshold_value <= 1e-9 and map_max_val > 1e-9:
        threshold_value = min(1e-9, map_max_val / 2.0) # Ensure some pixels can pass if map_max_val is small but >0

    # Create binary mask using the threshold
    # np.nan_to_num replaces NaNs with a very small number to ensure they are below threshold
    binary_mask = (np.nan_to_num(map_to_process, nan=-np.inf) >= threshold_value).astype(np.uint8)

    if not np.any(binary_mask):
        print("Warning: No pixels passed the activity threshold. Empty binary mask.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    # Morphological operations
    mask_processed = binary_mask.copy()
    # Adaptive kernel sizes based on image dimensions
    kernel_size_dim_close = max(min(3, activity_map.shape[0]//20, activity_map.shape[1]//20), 1)
    kernel_close = np.ones((kernel_size_dim_close, kernel_size_dim_close), np.uint8)
    
    kernel_size_dim_open = max(min(2, activity_map.shape[0]//30, activity_map.shape[1]//30), 1)
    kernel_open = np.ones((kernel_size_dim_open, kernel_size_dim_open), np.uint8)

    try:
        if morphology_op == 'close':
            mask_processed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
        elif morphology_op == 'open_close':
            mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
            mask_processed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
        elif morphology_op != 'none':
            print(f"Warning: Unknown morph_op '{morphology_op}'. Using 'none'.")
    except cv2.error as e:
        print(f"Warning: Morphology error: {e}. Using raw binary mask.")
        mask_processed = binary_mask # Fallback to pre-morphology mask

    # Connected component analysis to find the largest component
    try:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_processed, connectivity=8)
    except cv2.error as e:
        print(f"Error in connectedComponentsWithStats: {e}. Returning empty mask.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    hotspot_mask_final = np.zeros_like(activity_map, dtype=bool)
    hotspot_area = 0.0

    if num_labels > 1 and stats.shape[0] > 1: # Found at least one component beyond background
        # Find the label corresponding to the component with the largest area (excluding background label 0)
        # stats[0] is background.
        if stats.shape[0] > 1: # Ensure there are foreground components
            areas = stats[1:, cv2.CC_STAT_AREA] # Areas of all components except background
            if areas.size > 0:
                largest_component_idx = np.argmax(areas)
                largest_label = largest_component_idx + 1 # Add 1 because we sliced off background
                hotspot_mask_final = (labels == largest_label)
                hotspot_area = np.sum(hotspot_mask_final)
            else:
                print("Warning: No foreground components found after morphology.")
        else:
            print("Warning: Only background component found by connectedComponentsWithStats.")

    return hotspot_mask_final, hotspot_area


# --- Main Processing Function ---
def generate_masks_for_dataset(dataset_root_dir, output_base_dir, mat_key, fps, focus_duration_sec,
                               quantile, morphology_op, apply_blur, blur_kernel_size,
                               envir_para, augment, smooth_window, p_value_threshold,
                               fuselevel, normalizeT,
                               roi_border_percent=None, debug_pixel_coords=None):
    print(f"Starting mask generation process...")
    print(f"Dataset Root: {dataset_root_dir}")
    print(f"Mask Output Base: {output_base_dir}")
    print(f"--- Localization Parameters (Using P-Value Filtered Slope) ---")
    print(f"  Focus Duration: {focus_duration_sec}s | FPS: {fps} | Smooth Window: {smooth_window}")
    print(f"  P-Value Thresh: {p_value_threshold:.3f} | Quantile for Activity Map: {quantile:.3f} | Morphology: {morphology_op}")
    print(f"  Blur Activity Map: {apply_blur} (Kernel: {blur_kernel_size if apply_blur else 'N/A'})")
    print(f"  Frame Preprocessing - Fuse Level: {fuselevel} | NormalizeT: {normalizeT}")
    print(f"  Slope Filtering - Envir_para: {envir_para} | Augment Slope: {augment}")
    if roi_border_percent is not None and 0.0 <= roi_border_percent < 0.5 :
        print(f"  ROI Border for Slope Calc & Extraction: {roi_border_percent*100:.0f}%")
    else:
        print("  ROI Border: Not applied (processing full frame or invalid %)")
    if debug_pixel_coords:
        print(f"  DEBUGGING PIXEL: Row={debug_pixel_coords[0]}, Col={debug_pixel_coords[1]}")
    print("-" * 30)

    processed_count = 0
    error_count = 0
    total_files_found = 0
    for root, _, files in os.walk(dataset_root_dir):
        mat_files_in_dir = fnmatch.filter(files, '*.mat')
        total_files_found += len(mat_files_in_dir)
        for mat_file in mat_files_in_dir:
            mat_file_path = os.path.join(root, mat_file)
            # Construct relative path for output directory structure
            try:
                relative_path_dir = os.path.relpath(root, dataset_root_dir)
            except ValueError: # Happens if root is not under dataset_root_dir (e.g. if root is '.')
                relative_path_dir = "" # Store in top level of output_base_dir

            print(f"\nProcessing: {os.path.join(relative_path_dir, mat_file if relative_path_dir else mat_file)}")

            # Load Data
            try:
                mat_data = scipy.io.loadmat(mat_file_path)
                frames_raw = mat_data[mat_key].astype(np.float64)
                if frames_raw.ndim != 3 or frames_raw.shape[2] < 2: # Need at least 2 frames
                    raise ValueError(f"Invalid frame data shape: {frames_raw.shape}")
                H, W, num_frames = frames_raw.shape
            except Exception as e:
                print(f"  Error loading data from {mat_file_path}: {e}. Skipping.")
                error_count += 1
                continue

            # Pre-processing for slope calculation
            frames_proc_for_slope = apply_preprocessing(frames_raw, normalizeT, fuselevel)

            # Focus Window for slope calculation
            focus_duration_frames = int(focus_duration_sec * fps)
            # Ensure at least 2 frames, and not more than available
            focus_duration_frames = max(2, min(focus_duration_frames, num_frames))
            frames_focus = frames_proc_for_slope[:, :, :focus_duration_frames]

            # --- Define ROI Mask for slope calculation AND extraction ---
            current_roi_mask = None # Process full frame by default
            if roi_border_percent is not None and 0.0 <= roi_border_percent < 0.5:
                border_h = int(H * roi_border_percent)
                border_w = int(W * roi_border_percent)
                # Ensure ROI dimensions are valid
                if H - 2 * border_h > 0 and W - 2 * border_w > 0:
                    current_roi_mask = np.zeros((H, W), dtype=bool)
                    current_roi_mask[border_h : H - border_h, border_w : W - border_w] = True
                else:
                    print(f"  Warning: ROI border {roi_border_percent*100:.0f}% too large for image size ({H}x{W}). Processing full frame.")


            # --- Calculate Activity Map (P-Value Filtered Slope) ---
            activity_map, pixels_processed_slope, pixels_passed_slope = calculate_filtered_slope_activity_map(
                frames_focus, fps, smooth_window, envir_para, augment, p_value_threshold,
                roi_mask=current_roi_mask, # Pass ROI mask for slope calculation iteration
                debug_pixel=debug_pixel_coords
            )

            if activity_map is None: # Critical failure in slope calculation
                print("  Error: Failed activity map generation (calculate_filtered_slope_activity_map returned None). Skipping mask extraction.")
                error_count += 1
                continue
            
            if pixels_passed_slope == 0:
                print("  Warning: No pixels passed the slope and p-value filters. Activity map will be empty.")
                # Continue to extraction, which should handle an all-zero/NaN map gracefully

            # --- Extract Hotspot Mask ---
            # The same current_roi_mask can be passed to extract_hotspot_from_map
            # to ensure consistency if ROI is used for processing the activity map itself.
            hotspot_mask, hotspot_area = extract_hotspot_from_map(
                activity_map, quantile, morphology_op, apply_blur, tuple(blur_kernel_size),
                roi_mask=current_roi_mask # Pass ROI mask for activity map processing
            )

            if hotspot_mask.size == 0 : # Check if an empty array was returned
                print("  Error: Failed mask extraction (extract_hotspot_from_map returned empty array).")
                error_count += 1
                continue
            elif hotspot_area == 0:
                print("  Warning: Zero area mask generated for this file.")
            else:
                print(f"  Generated mask area: {hotspot_area:.0f} pixels.")


            # --- Save Mask ---
            try:
                mask_filename_out = os.path.splitext(mat_file)[0] + '_mask.npy'
                output_subdir_for_mask = os.path.join(output_base_dir, relative_path_dir)
                os.makedirs(output_subdir_for_mask, exist_ok=True)
                mask_save_path_final = os.path.join(output_subdir_for_mask, mask_filename_out)
                np.save(mask_save_path_final, hotspot_mask)
                #print(f"  Mask saved to: {mask_save_path_final}")
                processed_count += 1
            except Exception as e:
                print(f"  Error saving mask for {mat_file}: {e}")
                error_count += 1
    
    if total_files_found == 0:
        print(f"No .mat files found in the dataset directory: {dataset_root_dir}")

    print("-" * 30)
    print(f"Mask generation finished. Successfully processed: {processed_count} files. Errors/Skipped: {error_count} files.")


# --- Command-Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate hotspot masks using p-value filtered slope analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    # IO Args
    parser.add_argument("output_dir", help="Root directory to save masks.")
    parser.add_argument("--dataset_folder", default=config.DATASET_FOLDER,
                        help="Path to the root dataset folder containing .mat files.")
    parser.add_argument("--mat_key", default=config.MAT_FRAMES_KEY,
                        help="Key in .mat file that holds the thermal frame data.")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frames per second of original recording.")

    # Slope Analysis Params
    parser.add_argument("--focus_duration_sec", type=float,
                        default=5.0, help="Duration (s) for slope analysis.")
    parser.add_argument("--smooth_window", type=int, default=1,
                        help="Temporal smoothing window for slope fit (1=None).")
    parser.add_argument("--p_value_thresh", type=float, default=0.10,
                        help="P-value threshold for slope significance filter.")
    parser.add_argument("--envir_para", type=int, default=1,
                        choices=[-1, 1], help="Environment parameter (-1=Winter, 1=Summer).")
    parser.add_argument("--augment", type=float, default=1.0,
                        help="Exponent for augmenting filtered slope magnitude.")

    # Pre-processing Args
    parser.add_argument("--normalizeT", type=int, default=0,
                        choices=[0, 1], help="Normalize temperature per frame? (0=No, 1=Yes).")
    parser.add_argument("--fuselevel", type=int, default=0,
                        help="Spatial fuse level (0=None).")
    parser.add_argument("--roi_border_percent", type=float, default=0.0,  # Defaulting to apply ROI
                        help="Percent border (0-0.5) to exclude for ROI. Set to large value like 0.99 to disable (effectively no ROI).")

    # Hotspot Extraction Args
    parser.add_argument("-q", "--quantile", type=float,
                        default=0.99, help="Quantile threshold (0-1).")
    parser.add_argument("--morph_op", default="none",
                        choices=['close', 'open_close', 'none'], help="Morphological operation.")
    parser.add_argument("--blur", action='store_true',
                        help="Apply Gaussian blur to activity map.")
    parser.add_argument("--blur_kernel", type=int, nargs=2,
                        default=[3, 3], metavar=('H', 'W'), help="Kernel size for Gaussian blur.")

    # Debugging
    parser.add_argument("--debug_pixel", type=int, nargs=2, default=None, metavar=('ROW', 'COL'),
                        help="Optional: (row, col) of a specific pixel to print detailed debug info for during slope calculation.")


    # Check if any arguments were passed, if not, print help.
    # The "output_dir" is positional, so len(sys.argv) will be at least 2 if it's provided.
    if len(sys.argv) < 2: # No arguments provided (script name is sys.argv[0])
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()


    # --- Argument Validation (Basic) ---
    if not (0.0 < args.quantile <= 1.0):
        print("Error: Quantile must be > 0.0 and <= 1.0."); sys.exit(1)
    if args.focus_duration_sec <= 0:
        print("Error: Focus duration must be positive."); sys.exit(1)
    if args.fps <= 0:
        print("Error: FPS must be positive."); sys.exit(1)
    if args.blur:
        kh, kw = args.blur_kernel
        if not (kh > 0 and kw > 0 and kh % 2 != 0 and kw % 2 != 0):
            print("Error: Blur kernel dimensions must be positive and odd."); sys.exit(1)
    if not (0.0 <= args.roi_border_percent < 0.5): # Allow 0 for full frame
        print("Error: ROI border percent must be >= 0.0 and < 0.5."); sys.exit(1)
    if args.fuselevel < 0:
        print("Error: Fuse level cannot be negative."); sys.exit(1)
    if args.smooth_window < 1:
        print("Error: Smooth window must be >= 1."); sys.exit(1)
    if not (0.0 < args.p_value_thresh < 1.0):
        print("Error: P-value threshold must be > 0.0 and < 1.0."); sys.exit(1)
    if args.augment <= 0:
        print("Error: Augment exponent must be positive."); sys.exit(1)

    # Get Input Dir
    input_dataset_dir_to_use = args.dataset_folder
    if not input_dataset_dir_to_use or not os.path.isdir(input_dataset_dir_to_use):
        print(f"Error: Dataset folder not found: {input_dataset_dir_to_use}")
        sys.exit(1)

    # Set Output Dir
    output_mask_dir_to_use = args.output_dir
    os.makedirs(output_mask_dir_to_use, exist_ok=True)

    # Run processing
    generate_masks_for_dataset(
        dataset_root_dir=input_dataset_dir_to_use, output_base_dir=output_mask_dir_to_use,
        mat_key=args.mat_key, fps=args.fps,
        focus_duration_sec=args.focus_duration_sec,
        quantile=args.quantile, morphology_op=args.morph_op,
        apply_blur=args.blur, blur_kernel_size=tuple(args.blur_kernel),
        envir_para=args.envir_para, augment=args.augment,
        smooth_window=args.smooth_window,
        p_value_threshold=args.p_value_thresh,
        fuselevel=args.fuselevel, normalizeT=bool(args.normalizeT), # Ensure bool
        roi_border_percent=args.roi_border_percent,
        debug_pixel_coords=args.debug_pixel
    )

    print("Mask generation script finished.")
