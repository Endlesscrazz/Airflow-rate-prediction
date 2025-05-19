# hotspot_mask_generation.py
"""
Pre-processing script to generate and save hotspot masks for thermal .mat files.
Uses parameters from config.py as defaults, which can be overridden by CLI args.
"""

import numpy as np
import cv2
import os
import scipy.io
import argparse
import sys
from scipy import stats as sp_stats
from tqdm import tqdm
import fnmatch
import matplotlib.pyplot as plt # For saving debug activity map

# --- Import from data_utils and config ---
try:
    import data_utils # Not strictly used in this script but good practice if it becomes needed
    import config
except ImportError:
    print("Error: Failed to import data_utils.py or config.py.")
    print("Please ensure these files are in the same directory or accessible in PYTHONPATH.")
    sys.exit(1)

# --- Helper Functions ---
def apply_preprocessing(frames, normalizeT, fuselevel):
    frames_proc = frames.copy()
    if normalizeT:
        frame_means = np.mean(frames_proc, axis=(0, 1), keepdims=True)
        frame_means[frame_means == 0] = 1e-9 # Avoid division by zero
        frames_proc = frames_proc / frame_means
    if fuselevel > 0:
        kernel_size = 2 * fuselevel + 1
        try:
            num_f = frames_proc.shape[2]
            frames_fused = np.zeros_like(frames_proc)
            for i in range(num_f):
                frames_fused[:, :, i] = cv2.boxFilter(
                    frames_proc[:, :, i], -1, (kernel_size, kernel_size))
            frames_proc = frames_fused
        except Exception as e:
            print(f"Warning: Spatial fuse failed: {e}")
    return frames_proc

# --- Activity Map Calculation with P-value Filter ---
def calculate_filtered_slope_activity_map(frames_focus, fps, smooth_window,
                                          envir_para, augment, p_value_threshold,
                                          roi_mask=None):
    H, W, T_focus = frames_focus.shape
    final_activity_map = np.zeros((H, W), dtype=np.float64)
    passed_filter_count = 0

    if T_focus < 2:
        print("  Error in slope calc: Less than 2 time points.")
        return final_activity_map, 0 # Return empty map and count

    t_base = (np.arange(T_focus) / fps).astype(np.float64)
    frames_for_slope = frames_focus.copy() # Work on a copy

    if smooth_window > 1 and T_focus >= smooth_window:
        frames_smoothed_temp = np.zeros_like(frames_for_slope)
        for r_idx in range(H):
            for c_idx in range(W):
                smoothed_series = np.convolve(frames_for_slope[r_idx, c_idx, :],
                                            np.ones(smooth_window)/smooth_window, mode='valid')
                pad_len_before = (T_focus - len(smoothed_series)) // 2
                pad_len_after = T_focus - len(smoothed_series) - pad_len_before
                frames_smoothed_temp[r_idx, c_idx, :] = np.pad(
                    smoothed_series, (pad_len_before, pad_len_after), mode='edge')
        frames_for_slope = frames_smoothed_temp
    
    # Determine iteration indices
    pixel_iterator_indices = []
    if roi_mask is not None and roi_mask.dtype == bool and roi_mask.shape == (H,W) and np.any(roi_mask):
        pixel_iterator_indices = np.argwhere(roi_mask) # List of (r,c) tuples
        desc = "Calculating Slope (ROI)"
    else:
        pixel_iterator_indices = [(r, c) for r in range(H) for c in range(W)] # List of all (r,c)
        desc = "Calculating Slope (Full Frame)"

    for r, c in tqdm(pixel_iterator_indices, desc=desc, leave=False, ncols=100):
        y_pixel_series = frames_for_slope[r, c, :]
        valid_mask_y = np.isfinite(y_pixel_series)
        y_valid = y_pixel_series[valid_mask_y]
        t_valid = t_base[valid_mask_y]

        if len(y_valid) < 2:
            continue
        try:
            res = sp_stats.linregress(t_valid, y_valid)
            if np.isfinite(res.slope) and np.isfinite(res.pvalue):
                if (res.slope * envir_para > 0) and (res.pvalue < p_value_threshold):
                    activity_value = np.abs(res.slope)
                    final_activity_map[r, c] = np.power(
                        activity_value, augment) if augment != 1.0 else activity_value
                    passed_filter_count +=1
        except ValueError:
            pass # Raised by linregress for bad inputs
        except Exception as e_lr:
            # This should be rare if input data is clean
            print(f"\n  Warning: linregress/filter failed at ({r},{c}): {e_lr}")
            pass
    return final_activity_map, passed_filter_count

# --- Extract Hotspot from Activity Map ---
def extract_hotspot_from_map(activity_map, threshold_quantile=0.95, morphology_op='close',
                             apply_blur=False, blur_kernel_size=(3, 3),
                             roi_mask=None): # roi_mask here is for processing the activity_map
    if activity_map is None or activity_map.size == 0:
        print("Warning: Activity map is None or empty in extract_hotspot_from_map.")
        return np.array([]), 0.0

    map_to_process = activity_map.copy()
    if roi_mask is not None:
        if roi_mask.shape == activity_map.shape:
            if roi_mask.dtype != bool:
                roi_mask = roi_mask.astype(bool)
            map_to_process[~roi_mask] = np.nan
        else:
            print("Warning: ROI mask shape mismatch in extract_hotspot. Ignoring ROI for extraction.")

    if apply_blur:
        if np.all(np.isnan(map_to_process)):
            print("Warning: Activity map is all NaN after ROI application, skipping blur.")
        else:
            try:
                map_for_blur_no_nan = np.nan_to_num(map_to_process.astype(np.float32), nan=0.0)
                k_h = blur_kernel_size[0] + (1 - blur_kernel_size[0] % 2)
                k_w = blur_kernel_size[1] + (1 - blur_kernel_size[1] % 2)
                blurred_map = cv2.GaussianBlur(map_for_blur_no_nan, (k_w, k_h), 0)
                if roi_mask is not None and roi_mask.shape == map_to_process.shape:
                    blurred_map[~roi_mask] = np.nan
                map_to_process = blurred_map
            except Exception as e:
                print(f"Warning: GaussianBlur failed: {e}. Using unblurred map (post-ROI).")

    # --- Find the pixel with the maximum activity in the (potentially ROI'd and blurred) map ---
    # This will be our anchor point.
    max_activity_coords = None
    if not np.all(np.isnan(map_to_process)):
        try:
            # nanargmax finds the index of the max value, ignoring NaNs.
            max_activity_flat_idx = np.nanargmax(map_to_process)
            max_activity_coords = np.unravel_index(max_activity_flat_idx, map_to_process.shape)
            #print(f"  Debug: Max activity in map_to_process at {max_activity_coords} with value {map_to_process[max_activity_coords]:.4e}")
        except ValueError: # nanargmax raises error if all are NaN
             print("  Debug: All values in map_to_process are NaN before nanargmax.")
    else:
        print("  Debug: map_to_process is all NaN before finding max activity.")


    non_nan_pixels_before_thresh = map_to_process[~np.isnan(map_to_process)]
    if non_nan_pixels_before_thresh.size > 0:
        min_act = np.min(non_nan_pixels_before_thresh)
        max_act = np.max(non_nan_pixels_before_thresh)
        non_zero_count = np.sum(non_nan_pixels_before_thresh > 1e-9)
        #print(f"Debug: Activity map before thresholding - Min: {min_act:.2e}, Max: {max_act:.2e}, Non-zero count: {non_zero_count}")
    else:
        print("Debug: Activity map before thresholding is all NaN or empty.")

    nan_mask = np.isnan(map_to_process)
    if np.all(nan_mask):
        print("Warning: Activity map all NaNs after ROI/blur in extract_hotspot_from_map.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    valid_pixels_for_threshold = map_to_process[~nan_mask]
    if valid_pixels_for_threshold.size == 0:
        print("Warning: No valid pixels found in activity map for thresholding.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    map_std_val = np.std(valid_pixels_for_threshold)
    map_max_val = np.max(valid_pixels_for_threshold)

    if map_max_val < 1e-9 or (map_std_val < 1e-9 and valid_pixels_for_threshold.size > 1):
        print("Warning: Max activity is near zero or no variation in activity map.")
        return np.zeros_like(activity_map, dtype=bool), 0.0
    
    try:
        threshold_value = np.percentile(valid_pixels_for_threshold, threshold_quantile * 100)
    except IndexError:
        threshold_value = map_max_val
    
    print(f"  Threshold value for activity map: {threshold_value:.4e} (Quantile: {threshold_quantile*100}%)")

    if threshold_value <= 1e-9 and map_max_val > 1e-9:
        threshold_value = min(1e-9, map_max_val / 2.0)

    binary_mask = (np.nan_to_num(map_to_process, nan=-np.inf) >= threshold_value).astype(np.uint8)

    if not np.any(binary_mask):
        print("Warning: No pixels passed the activity threshold. Empty binary mask.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    mask_processed = binary_mask.copy()
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
        mask_processed = binary_mask

    try:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_processed, connectivity=8)
    except cv2.error as e:
        print(f"Error in connectedComponentsWithStats: {e}. Returning empty mask.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    hotspot_mask_final = np.zeros_like(activity_map, dtype=bool)
    hotspot_area = 0.0

    if num_labels > 1: # Found at least one component beyond background
        # --- MODIFIED LOGIC: Anchor to max_activity_coords ---
        if max_activity_coords is not None and labels[max_activity_coords] != 0:
            # If the max activity pixel is part of a valid component (not background)
            label_at_max_activity = labels[max_activity_coords]
            hotspot_mask_final = (labels == label_at_max_activity)
            hotspot_area = np.sum(hotspot_mask_final)
            #print(f"  Selected component containing max activity pixel. Label: {label_at_max_activity}, Area: {hotspot_area}")
        elif stats.shape[0] > 1: # Fallback to largest if max_activity_coords not useful
            print("  Warning: Max activity pixel not in a valid component or not found. Falling back to largest component.")
            areas = stats[1:, cv2.CC_STAT_AREA]
            if areas.size > 0:
                largest_component_idx = np.argmax(areas)
                largest_label = largest_component_idx + 1
                hotspot_mask_final = (labels == largest_label)
                hotspot_area = np.sum(hotspot_mask_final)
                print(f"  Fallback: Selected largest component. Label: {largest_label}, Area: {hotspot_area}")
            else:
                print("  Warning: No foreground components found after morphology (fallback).")
        else:
            print("  Warning: Only background component found by connectedComponentsWithStats (fallback).")
    else:
        print("  Warning: No components found beyond background.")
        
    return hotspot_mask_final, hotspot_area


# --- Main Orchestration Function ---
def run_mask_generation_on_dataset(
    dataset_dir, output_dir, mat_key, fps, focus_sec, smooth_win, p_thresh,
    envir, augment_s, norm_t, fuse_lvl, roi_perc,
    act_quant, morph_op, apply_act_blur, blur_kern
):
    print(f"--- Mask Generation Run Parameters ---")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output Masks: {output_dir}")
    print(f"  MAT Key: {mat_key}, FPS: {fps}")
    print(f"  Slope Calc: Focus={focus_sec}s, SmoothWin={smooth_win}, P-Thresh={p_thresh}, Envir={envir}, Augment={augment_s}")
    print(f"  Frame Preproc: NormalizeT={norm_t}, FuseLevel={fuse_lvl}")
    print(f"  ROI Border: {roi_perc*100 if roi_perc is not None else 'None'}%")
    print(f"  Extraction: Quantile={act_quant:.3f}, MorphOp={morph_op}, BlurActivityMap={apply_act_blur} (Kernel: {blur_kern})")
    print("-" * 30)

    processed_files_count = 0
    error_files_count = 0
    total_mat_files = 0
    skipped_cooling_folders = 0

    for root, dirs, files in os.walk(dataset_dir):
        if "cooling" in dirs:
            dirs.remove("cooling") # This prevents os.walk from descending into 'cooling'
            skipped_cooling_folders +=1
            print(f"  Skipping 'cooling' subfolder found in: {root}")

        mat_files_in_current_dir = fnmatch.filter(files, '*.mat')
        total_mat_files += len(mat_files_in_current_dir)

        if not mat_files_in_current_dir and "cooling" not in os.path.basename(root).lower(): # Avoid printing for parent of cooling
            # Only print if it's not a directory that *only* contained 'cooling'
            parent_dir_name = os.path.basename(root)

        for mat_filename in mat_files_in_current_dir:
            mat_filepath = os.path.join(root, mat_filename)
            relative_dir_path = os.path.relpath(root, dataset_dir)
            if relative_dir_path == ".": relative_dir_path = "" # Handle case where dataset_dir is current dir

            print(f"\nProcessing: {os.path.join(relative_dir_path, mat_filename)}")

            try:
                mat_data = scipy.io.loadmat(mat_filepath)
                frames_raw_data = mat_data[mat_key].astype(np.float64)
                if frames_raw_data.ndim != 3 or frames_raw_data.shape[2] < 2:
                    raise ValueError(f"Invalid frame data shape: {frames_raw_data.shape}")
                H_img, W_img, num_total_frames_img = frames_raw_data.shape
            except Exception as e:
                print(f"  Error loading data from {mat_filepath}: {e}. Skipping.")
                error_files_count += 1
                continue

            frames_preprocessed = apply_preprocessing(frames_raw_data, norm_t, fuse_lvl)
            
            focus_frames_count = max(2, min(int(focus_sec * fps), num_total_frames_img))
            frames_for_slope_calc = frames_preprocessed[:, :, :focus_frames_count]

            current_roi_definition_mask = None
            if roi_perc is not None and 0.0 <= roi_perc < 0.5:
                border_h_roi = int(H_img * roi_perc)
                border_w_roi = int(W_img * roi_perc)
                if H_img - 2 * border_h_roi > 0 and W_img - 2 * border_w_roi > 0:
                    current_roi_definition_mask = np.zeros((H_img, W_img), dtype=bool)
                    current_roi_definition_mask[border_h_roi : H_img - border_h_roi, border_w_roi : W_img - border_w_roi] = True
                else:
                    print(f"  Warning: ROI border {roi_perc*100:.0f}% too large. Processing full frame.")
            
            activity_map_gen, num_passed_slope = calculate_filtered_slope_activity_map(
                frames_for_slope_calc, fps, smooth_win, envir, augment_s, p_thresh,
                roi_mask=current_roi_definition_mask
            )
            print(f"  Activity map from slope: Non-zero count={np.count_nonzero(np.nan_to_num(activity_map_gen))}, Pixels passed slope filter={num_passed_slope}")
            
            # Save debug activity map image
            debug_map_filename = f"debug_activity_{os.path.splitext(mat_filename)[0]}.png"
            debug_map_output_dir = os.path.join(output_dir, relative_dir_path, "debug_activity_maps")
            os.makedirs(debug_map_output_dir, exist_ok=True)
            plt.figure()
            plt.imshow(activity_map_gen, cmap='hot') # Or 'viridis' or another perceptually uniform map
            plt.colorbar(label="Activity (abs slope)")
            plt.title(f"Activity Map: {mat_filename}\n(Passed Slope Filter: {num_passed_slope})")
            plt.savefig(os.path.join(debug_map_output_dir, debug_map_filename))
            plt.close()


            if activity_map_gen is None or num_passed_slope == 0: # if map is None or all zeros effectively
                print(f"  Warning: Activity map is empty or no pixels passed slope filters. Resulting mask will be empty.")
                # Create an empty mask to save
                final_mask_generated = np.zeros((H_img, W_img), dtype=bool)
                final_mask_area = 0
            else:
                final_mask_generated, final_mask_area = extract_hotspot_from_map(
                    activity_map_gen, act_quant, morph_op, apply_act_blur, blur_kern,
                    roi_mask=current_roi_definition_mask # Use same ROI for consistency
                )

            if final_mask_generated.size == 0: # Error from extract_hotspot
                print(f"  Error: Mask extraction failed for {mat_filename}.")
                error_files_count += 1
                continue
            elif final_mask_area == 0:
                print(f"  Warning: Zero area mask generated for {mat_filename}.")
            else:
                print(f"  Generated final mask area: {final_mask_area:.0f} pixels.")

            try:
                out_mask_filename = os.path.splitext(mat_filename)[0] + '_mask.npy'
                final_mask_output_subdir = os.path.join(output_dir, relative_dir_path)
                os.makedirs(final_mask_output_subdir, exist_ok=True)
                np.save(os.path.join(final_mask_output_subdir, out_mask_filename), final_mask_generated)
                processed_files_count += 1
            except Exception as e:
                print(f"  Error saving final mask for {mat_filename}: {e}")
                error_files_count += 1
    
    if total_mat_files == 0:
        print(f"No .mat files found in dataset: {dataset_dir}")
    print("-" * 30)
    print(f"Mask generation run complete. Processed: {processed_files_count}. Errors: {error_files_count}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate hotspot masks using P-Value Filtered Slope Analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # IO
    parser.add_argument("output_dir",
                        help="Root directory where generated masks will be saved (subfolders mirroring dataset structure will be created).")
    parser.add_argument("--dataset_folder", default=config.DATASET_FOLDER,
                        help="Path to the root dataset folder containing .mat files.")
    parser.add_argument("--mat_key", default=config.MAT_FRAMES_KEY,
                        help="Key in .mat files for the thermal frame data.")
    parser.add_argument("--fps", type=float, default=config.MASK_FPS,
                        help="Frames per second of the original thermal video recordings.")

    # Slope Calculation Parameters
    parser.add_argument("--focus_duration_sec", type=float, default=config.MASK_FOCUS_DURATION_SEC,
                        help="Duration (seconds) of the initial video segment to analyze for slopes.")
    parser.add_argument("--smooth_window", type=int, default=config.MASK_SMOOTH_WINDOW,
                        help="Size of the moving average window for temporal smoothing (1 = no smoothing).")
    parser.add_argument("--p_value_thresh", type=float, default=config.MASK_P_VALUE_THRESHOLD,
                        help="P-value threshold for statistical significance of the slope.")
    parser.add_argument("--envir_para", type=int, default=config.MASK_ENVIR_PARA, choices=[-1, 1],
                        help="Environmental parameter: -1 for Winter/cooling expected, 1 for Summer/heating expected.")
    parser.add_argument("--augment_slope", type=float, default=config.MASK_AUGMENT_SLOPE,
                        help="Exponent to augment the magnitude of valid slopes.")

    # Frame Preprocessing Parameters
    parser.add_argument("--normalize_temp", type=lambda x: (str(x).lower() == 'true'), # Handles bool from CLI
                        default=config.MASK_NORMALIZE_TEMP_FRAMES,
                        help="Normalize temperature frames before slope calculation? (True/False)")
    parser.add_argument("--fuse_level", type=int, default=config.MASK_FUSE_LEVEL,
                        help="Spatial fuse level for frame preprocessing (0 = no fusing).")

    # ROI Parameter
    parser.add_argument("--roi_border_percent", type=float, default=config.MASK_ROI_BORDER_PERCENT,
                        help="Percentage of border to exclude for ROI (0.0 to <0.5). 0.0 means no ROI.")

    # Hotspot Extraction Parameters
    parser.add_argument("--activity_quantile", type=float, default=config.MASK_ACTIVITY_QUANTILE,
                        help="Quantile to threshold the activity map for hotspot extraction (0-1).")
    parser.add_argument("--morphology_op", default=config.MASK_MORPHOLOGY_OP, choices=['close', 'open_close', 'none'],
                        help="Morphological operation to apply to the binary mask.")
    parser.add_argument("--apply_blur_activity", type=lambda x: (str(x).lower() == 'true'),
                        default=config.MASK_APPLY_BLUR_TO_ACTIVITY_MAP,
                        help="Apply Gaussian blur to the activity map before thresholding? (True/False)")
    parser.add_argument("--blur_kernel_size", type=int, nargs=2, default=config.MASK_BLUR_KERNEL_SIZE,
                        metavar=('H', 'W'), help="Kernel size (height width) for Gaussian blur.")

    if len(sys.argv) < 2: # Need at least output_dir
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # --- Basic Validations for critical parameters ---
    if not (0.0 < args.activity_quantile <= 1.0):
        print("Error: Activity quantile must be > 0.0 and <= 1.0."); sys.exit(1)
    if args.focus_duration_sec <= 0 or args.fps <= 0:
        print("Error: Focus duration and FPS must be positive."); sys.exit(1)
    if not (0.0 <= args.roi_border_percent < 0.5):
        print("Error: ROI border percent must be >= 0.0 and < 0.5."); sys.exit(1)
    if not (0.0 < args.p_value_thresh < 1.0):
        print("Error: P-value threshold must be > 0.0 and < 1.0."); sys.exit(1)
    if args.apply_blur_activity:
        kh, kw = args.blur_kernel_size
        if not (kh > 0 and kw > 0 and kh % 2 != 0 and kw % 2 != 0):
            print("Error: Blur kernel dimensions must be positive and odd."); sys.exit(1)


    if not os.path.isdir(args.dataset_folder):
        print(f"Error: Dataset folder not found: {args.dataset_folder}")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    run_mask_generation_on_dataset(
        dataset_dir=args.dataset_folder,
        output_dir=args.output_dir,
        mat_key=args.mat_key,
        fps=args.fps,
        focus_sec=args.focus_duration_sec,
        smooth_win=args.smooth_window,
        p_thresh=args.p_value_thresh,
        envir=args.envir_para,
        augment_s=args.augment_slope,
        norm_t=args.normalize_temp,
        fuse_lvl=args.fuse_level,
        roi_perc=args.roi_border_percent,
        act_quant=args.activity_quantile,
        morph_op=args.morphology_op,
        apply_act_blur=args.apply_blur_activity,
        blur_kern=tuple(args.blur_kernel_size)
    )

    print("\nMask generation script finished.")