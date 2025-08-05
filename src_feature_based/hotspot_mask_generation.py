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

import data_preprocessing

# --- Import from data_utils and config ---
try:
    import data_utils 
    import config
except ImportError:
    print("Error: Failed to import data_utils.py or config.py.")
    print("Please ensure these files are in the same directory or accessible in PYTHONPATH.")
    sys.exit(1)

# --- Helper Functions ---
def apply_general_preprocessing(frames, normalizeT_frame_wise, fuselevel_box_filter):

    frames_proc = frames.copy()
    if normalizeT_frame_wise: # Frame-wise normalization by its own spatial mean
        frame_means = np.mean(frames_proc, axis=(0, 1), keepdims=True)
        frame_means[frame_means == 0] = 1e-9
        frames_proc = frames_proc / frame_means
    if fuselevel_box_filter > 0: # Your existing spatial box filter
        kernel_size = 2 * fuselevel_box_filter + 1
        try:
            for i in range(frames_proc.shape[2]):
                frames_proc[:, :, i] = cv2.boxFilter(frames_proc[:, :, i], -1, (kernel_size, kernel_size))
        except Exception as e: print(f"Warning: Fuselevel box filter failed: {e}")
    return frames_proc

# --- Activity Map Calculation with P-value Filter ---
def calculate_filtered_slope_activity_map(frames_for_slope_input, fps, temporal_smooth_window,
                                          envir_para, augment_slope_val, p_val_thresh,
                                          roi_mask=None):

    H, W, T_focus = frames_for_slope_input.shape
    final_activity_map = np.zeros((H, W), dtype=np.float64)
    passed_filters_count = 0

    if T_focus < 2:
        print("  Error in slope calc: Less than 2 time points."); return final_activity_map, 0
    
    t_base_slope = (np.arange(T_focus) / fps).astype(np.float64)
    frames_to_fit = frames_for_slope_input.copy() # Already prepped

    if temporal_smooth_window > 1 and T_focus >= temporal_smooth_window:
        # ... (Temporal smoothing logic - apply to frames_to_fit)
        frames_smoothed_temp_slope = np.zeros_like(frames_to_fit)
        for r_idx in range(H):
            for c_idx in range(W):
                smoothed_series_slope = np.convolve(frames_to_fit[r_idx, c_idx, :],
                                            np.ones(temporal_smooth_window)/temporal_smooth_window, mode='valid')
                pad_len_before_slope = (T_focus - len(smoothed_series_slope)) // 2
                pad_len_after_slope = T_focus - len(smoothed_series_slope) - pad_len_before_slope
                if len(smoothed_series_slope) > 0:
                    frames_smoothed_temp_slope[r_idx, c_idx, :] = np.pad(
                        smoothed_series_slope, (pad_len_before_slope, pad_len_after_slope), mode='edge')
                else: # Should not happen if T_focus >= smooth_window
                     frames_smoothed_temp_slope[r_idx, c_idx, :] = frames_to_fit[r_idx, c_idx, :]
        frames_to_fit = frames_smoothed_temp_slope 
    
    pixel_iterator_indices = []
    desc_iter_slope = "Calculating Slope"
    if roi_mask is not None and roi_mask.dtype == bool and roi_mask.shape == (H,W) and np.any(roi_mask):
        pixel_iterator_indices = np.argwhere(roi_mask); desc_iter_slope += " (ROI)"
    else:
        pixel_iterator_indices = [(r, c) for r in range(H) for c in range(W)]; desc_iter_slope += " (Full Frame)"

    for r_s, c_s in tqdm(pixel_iterator_indices, desc=desc_iter_slope, leave=False, ncols=100):
        y_pixel_s = frames_to_fit[r_s, c_s, :]; valid_y_mask_s = np.isfinite(y_pixel_s)
        y_valid_s = y_pixel_s[valid_y_mask_s]; t_valid_s = t_base_slope[valid_y_mask_s]
        if len(y_valid_s) < 2: 
            continue
        try:
            res_s = sp_stats.linregress(t_valid_s, y_valid_s)
            if np.isfinite(res_s.slope) and np.isfinite(res_s.pvalue):
                if (res_s.slope * envir_para > 0) and (res_s.pvalue < p_val_thresh):
                    act_val_s = np.abs(res_s.slope)
                    final_activity_map[r_s, c_s] = np.power(act_val_s, augment_slope_val) if augment_slope_val != 1.0 else act_val_s
                    passed_filters_count +=1
        except ValueError: pass
        except Exception as e_lr: 
            print(f"\n  Warning: linregress/filter failed at ({r_s},{c_s}): {e_lr}") 

    return final_activity_map, passed_filters_count

# --- Extract Hotspot from Activity Map ---
def extract_hotspot_from_map(activity_map, threshold_quantile=0.95, morphology_op='close',
                             apply_blur=False, blur_kernel_size=(3, 3),
                             roi_mask=None): 
    
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
    dataset_dir, output_dir, mat_key, 
    # Slope calc params
    fps_slope, focus_sec_slope, 
    spatial_blur_ksize_slope, subtract_pixel_mean_slope, # NEW PARAMS
    temporal_smooth_win_slope, p_thresh_slope, envir_slope, augment_val_slope,
    # General frame preproc (from your original apply_preprocessing)
    norm_t_frame_wise, fuse_lvl_box_filter, 
    # ROI
    roi_perc,
    # Extraction params
    act_quant_extract, morph_op_extract, apply_act_blur_extract, blur_kern_extract
):
    print(f"--- Mask Generation Run Parameters ---")
    print(f"  Dataset: {dataset_dir}, Output Masks: {output_dir}, MAT Key: {mat_key}")
    print(f"  Slope Input Preproc: SpatialBlurKsize={spatial_blur_ksize_slope}, SubtractPixelMean={subtract_pixel_mean_slope}")
    print(f"  Slope Calc: FPS={fps_slope}, Focus={focus_sec_slope}s, TemporalSmoothWin={temporal_smooth_win_slope}, P-Thresh={p_thresh_slope:.3f}, Envir={envir_slope}, Augment={augment_val_slope}")
    print(f"  General Frame Preproc: NormalizeT(FrameWise)={norm_t_frame_wise}, FuseLevel(BoxFilter)={fuse_lvl_box_filter}")
    print(f"  ROI Border: {roi_perc*100 if roi_perc is not None else 'None'}%")
    print(f"  Extraction: Quantile={act_quant_extract:.3f}, MorphOp={morph_op_extract}, BlurActivityMap={apply_act_blur_extract} (Kernel: {blur_kern_extract})")
    print("-" * 30)

    processed_files_count = 0; error_files_count = 0; total_mat_files = 0; skipped_cooling_count = 0

    for root, dirs, files in os.walk(dataset_dir):
        if "cooling" in dirs: dirs.remove("cooling"); skipped_cooling_count += 1
        mat_files_in_dir = fnmatch.filter(files, '*.mat')
        total_mat_files += len(mat_files_in_dir)

        for mat_filename in mat_files_in_dir:
            mat_filepath = os.path.join(root, mat_filename)
            relative_dir_path = os.path.relpath(root, dataset_dir)
            if relative_dir_path == ".": relative_dir_path = ""
            print(f"\nProcessing: {os.path.join(relative_dir_path, mat_filename)}")

            try:
                frames_raw = scipy.io.loadmat(mat_filepath)[mat_key].astype(np.float64)
                H_img, W_img, num_total_frames = frames_raw.shape
                if num_total_frames < 2: raise ValueError("Not enough frames.")
            except Exception as e: print(f"  Error loading: {e}. Skip."); error_files_count += 1; continue

            # 1. Apply general frame preprocessing (like your original `apply_preprocessing`)
            # This is for features like temp_max_overall_initial that need less aggressive processing
            frames_generally_preprocessed = apply_general_preprocessing(frames_raw, norm_t_frame_wise, fuse_lvl_box_filter)
            
            # 2. Prepare frames specifically for slope calculation
            focus_frames_count_slope = max(2, min(int(focus_sec_slope * fps_slope), num_total_frames))
            frames_focus_raw_for_slope = frames_raw[:, :, :focus_frames_count_slope].copy() # Start with raw for focus window

            # --- Apply NEW Step 1: Spatial Blur to focus window for slope calc ---
            frames_focus_spatially_blurred = data_preprocessing.spatial_blur_video_frames(
                frames_focus_raw_for_slope, ksize=spatial_blur_ksize_slope
            )

            # --- Apply NEW Step 2: Subtract Per-Pixel Mean from spatially blurred focus window ---
            frames_ready_for_slope_calc = frames_focus_spatially_blurred # Default if mean subtraction is off
            if subtract_pixel_mean_slope:
                if frames_focus_spatially_blurred.shape[2] > 0: # Ensure T_focus > 0
                    pixel_wise_mean = np.nanmean(frames_focus_spatially_blurred, axis=2, keepdims=True)
                    # Broadcast subtraction. Handle cases where pixel_wise_mean might be all NaN for a pixel.
                    frames_ready_for_slope_calc = np.where(np.isnan(pixel_wise_mean), 
                                                           frames_focus_spatially_blurred, 
                                                           frames_focus_spatially_blurred - pixel_wise_mean)
                else:
                    print("  Warning: Focus window for slope has 0 frames after some processing, cannot subtract pixel mean.")

            current_roi_definition_mask = None
            if roi_perc is not None and 0.0 <= roi_perc < 0.5:
                border_h_roi = int(H_img * roi_perc)
                border_w_roi = int(W_img * roi_perc)
                if H_img - 2 * border_h_roi > 0 and W_img - 2 * border_w_roi > 0:
                    current_roi_definition_mask = np.zeros((H_img, W_img), dtype=bool)
                    current_roi_definition_mask[border_h_roi : H_img - border_h_roi, border_w_roi : W_img - border_w_roi] = True
                else:
                    print(f"  Warning: ROI border {roi_perc*100:.0f}% too large. Processing full frame.")
            
            activity_map, passed_slope = calculate_filtered_slope_activity_map(
                frames_ready_for_slope_calc, fps_slope, temporal_smooth_win_slope, 
                envir_slope, augment_val_slope, p_thresh_slope,
                roi_mask=current_roi_definition_mask
            )
            print(f"  Activity map: Non-zero={np.count_nonzero(np.nan_to_num(activity_map))}, PassedSlopeFilter={passed_slope}")
            # Save debug activity map image
            debug_map_filename = f"debug_activity_{os.path.splitext(mat_filename)[0]}.png"
            debug_map_output_dir = os.path.join(output_dir, relative_dir_path, "debug_activity_maps")
            os.makedirs(debug_map_output_dir, exist_ok=True)
            plt.figure()
            plt.imshow(activity_map, cmap='hot') # Or 'viridis' or another perceptually uniform map
            plt.colorbar(label="Activity (abs slope)")
            plt.title(f"Activity Map: {mat_filename}\n(Passed Slope Filter: {passed_slope})")
            plt.savefig(os.path.join(debug_map_output_dir, debug_map_filename))
            plt.close()


            if activity_map is None or passed_slope == 0: # if map is None or all zeros effectively
                print(f"  Warning: Activity map is empty or no pixels passed slope filters. Resulting mask will be empty.")
                # Create an empty mask to save
                final_mask_generated = np.zeros((H_img, W_img), dtype=bool)
                final_mask_area = 0
            else:
                final_mask_generated, final_mask_area = extract_hotspot_from_map(
                    activity_map, act_quant_extract, morph_op_extract, 
                    apply_act_blur_extract, blur_kern_extract,
                    roi_mask=current_roi_definition_mask 
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
    parser.add_argument("--dataset_folder", default=config.DATASET_PARENT_DIR,
                        help="Path to the root dataset folder containing .mat files.")
    parser.add_argument("--mat_key", default=config.MAT_FRAMES_KEY,
                        help="Key in .mat files for the thermal frame data.")

    # Slope Calculation Group
    slope_group = parser.add_argument_group('Slope Calculation Parameters')
    slope_group.add_argument("--fps", type=float, default=config.MASK_FPS, help="Video FPS.")
    slope_group.add_argument("--focus_duration_sec", type=float, default=config.MASK_FOCUS_DURATION_SEC, help="Focus duration (s) for slope.")
    slope_group.add_argument("--spatial_blur_ksize", type=int, default=config.MASK_SPATIAL_BLUR_KSIZE, help="Kernel size for spatial blur before slope (0/1 to disable).")
    slope_group.add_argument("--subtract_pixel_mean", type=lambda x: (str(x).lower() == 'true'), default=config.MASK_SUBTRACT_PIXEL_MEAN_FOR_SLOPE, help="Subtract per-pixel mean from focus window? (True/False)")
    slope_group.add_argument("--temporal_smooth_window", type=int, default=config.MASK_SMOOTH_WINDOW, help="Temporal smoothing window (1=none).")
    slope_group.add_argument("--p_value_thresh", type=float, default=config.MASK_P_VALUE_THRESHOLD, help="P-value threshold.")
    slope_group.add_argument("--envir_para", type=int, default=config.MASK_ENVIR_PARA, choices=[-1, 1], help="Environment: -1 Cool, 1 Heat.")
    slope_group.add_argument("--augment_slope", type=float, default=config.MASK_AUGMENT_SLOPE, help="Slope augmentation exponent.")

    # General Frame Preprocessing Group (applied before focus window selection for slope if flags used)
    gen_prep_group = parser.add_argument_group('General Frame Preprocessing (e.g., for feature engineering consistency)')
    gen_prep_group.add_argument("--normalizeT_frame_wise", type=lambda x: (str(x).lower() == 'true'), default=config.MASK_NORMALIZE_TEMP_FRAMES, help="Frame-wise T normalization? (True/False)")
    gen_prep_group.add_argument("--fuse_level_box_filter", type=int, default=config.MASK_FUSE_LEVEL, help="Spatial box filter fuse level (0=none).")

    # ROI Group
    roi_group = parser.add_argument_group('ROI Parameters')
    roi_group.add_argument("--roi_border", type=float, default=config.MASK_ROI_BORDER_PERCENT, help="ROI border % (0.0 to <0.5).")
    
    # Extraction Group
    extract_group = parser.add_argument_group('Hotspot Extraction Parameters')
    extract_group.add_argument("--activity_quantile", type=float, default=config.MASK_ACTIVITY_QUANTILE, help="Activity map quantile threshold.")
    extract_group.add_argument("--morphology_op", default=config.MASK_MORPHOLOGY_OP, choices=['close', 'open_close', 'none'], help="Morphology operation.")
    extract_group.add_argument("--apply_blur_activity", type=lambda x: (str(x).lower() == 'true'), default=config.MASK_APPLY_BLUR_TO_ACTIVITY_MAP, help="Blur activity map? (True/False)")
    extract_group.add_argument("--blur_kernel_size", type=int, nargs=2, default=config.MASK_BLUR_KERNEL_SIZE, metavar=('H', 'W'), help="Activity map blur kernel.")

    if len(sys.argv) < 2: parser.print_help(); sys.exit(1)
    args = parser.parse_args()

    # Validations (simplified, add more as needed)
    if not (0.0 <= args.roi_border < 0.5): print("Error: ROI invalid."); sys.exit(1)
    if not (0.0 < args.p_value_thresh < 1.0): print("Error: P-value invalid."); sys.exit(1)
    # ... more validations ...

    if not os.path.isdir(args.dataset_folder): print(f"Error: Dataset folder missing: {args.dataset_folder}"); sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    run_mask_generation_on_dataset(
        dataset_dir=args.dataset_folder, output_dir=args.output_dir, mat_key=args.mat_key,
        # Slope calc params
        fps_slope=args.fps, focus_sec_slope=args.focus_duration_sec,
        spatial_blur_ksize_slope=args.spatial_blur_ksize, # NEW
        subtract_pixel_mean_slope=args.subtract_pixel_mean, # NEW
        temporal_smooth_win_slope=args.temporal_smooth_window,
        p_thresh_slope=args.p_value_thresh, envir_slope=args.envir_para, augment_val_slope=args.augment_slope,
        # General frame preproc
        norm_t_frame_wise=args.normalizeT_frame_wise, fuse_lvl_box_filter=args.fuse_level_box_filter,
        # ROI
        roi_perc=args.roi_border,
        # Extraction
        act_quant_extract=args.activity_quantile, morph_op_extract=args.morphology_op,
        apply_act_blur_extract=args.apply_blur_activity, blur_kern_extract=tuple(args.blur_kernel_size)
    )
    print("\nMask generation script finished.")