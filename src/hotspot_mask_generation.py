# hotspot_mask_generation.py
"""
Pre-processing script to generate and save hotspot masks for thermal .mat files.

Iterates through a dataset directory (defined in config), applies robust localization
based on focused, weighted linear slope analysis, and saves the resulting boolean masks
as .npy files in a structured output directory. Optionally applies an ROI mask.
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
    import data_utils
    import config
except ImportError:
    print("Error: Failed to import data_utils.py or config.py.")
    sys.exit(1)

# --- Helper Functions ---

def apply_preprocessing(frames, normalizeT, fuselevel):
    """Applies optional normalization and spatial fusing."""
    frames_proc = frames.copy()
    if normalizeT:
        frame_means = np.mean(frames_proc, axis=(0, 1), keepdims=True)
        frame_means[frame_means == 0] = 1.0
        frames_proc /= frame_means
    if fuselevel > 0:
        kernel_size = 2 * fuselevel + 1
        try:
            num_frames = frames_proc.shape[2]
            frames_fused = np.zeros_like(frames_proc)
            for i in range(num_frames):
                frames_fused[:, :, i] = cv2.boxFilter(frames_proc[:, :, i], -1, (kernel_size, kernel_size))
            frames_proc = frames_fused
        except Exception as e: print(f"Warning: Spatial fuse failed: {e}")
    return frames_proc

def calculate_weighted_slope_activity_map(frames_focus, fps, smooth_window,
                                           envir_para, errorweight, augment, roi_mask=None):
    """Calculates the filtered, weighted slope map.
       Slope calculation is vectorized, error estimation uses pixel loop.
       Optionally iterates only within ROI for error estimation.
    """
    H, W, T_focus = frames_focus.shape
    final_activity_map = np.zeros((H, W), dtype=np.float64)

    # --- Temporal Smoothing (Pixel loop still potentially slow) ---
    if smooth_window > 1:
        print(f"  Applying temporal smoothing (window={smooth_window})...")
        frames_focus_smoothed = np.zeros_like(frames_focus)
        # Pad edges for 'valid' mode equivalent length if desired, or use 'same'
        # Using 'same' is easier but less precise at edges
        for r in range(H):
            for c in range(W):
                frames_focus_smoothed[r, c, :] = np.convolve(frames_focus[r, c, :], np.ones(smooth_window)/smooth_window, mode='same')
        T_segment_smoothed = T_focus # Length remains the same with mode='same'
        t_smoothed = (np.arange(T_segment_smoothed) / fps).astype(np.float64)
    else:
        frames_focus_smoothed = frames_focus # No smoothing
        T_segment_smoothed = T_focus
        t_smoothed = (np.arange(T_segment_smoothed) / fps).astype(np.float64)

    if T_segment_smoothed < 2:
        print("Error: Less than 2 effective time points after potential smoothing.")
        return None

    # --- Vectorized Slope Calculation ---
    print("  Calculating slope map (vectorized)...")
    t_mean = t_smoothed.mean()
    var_t = ((t_smoothed - t_mean)**2).sum()
    if var_t < 1e-9:
         print("Warning: Time vector variance near zero.")
         # Return zero map or NaN map? Zero allows processing continuation.
         return np.zeros((H, W), dtype=np.float64)

    mean_pix = np.mean(frames_focus_smoothed, axis=2, keepdims=True)
    diff_f = frames_focus_smoothed - mean_pix
    diff_t = t_smoothed - t_mean
    cov = np.tensordot(diff_f, diff_t, axes=([2], [0]))
    slope_map = cov / var_t # Signed slopes calculated efficiently

    # --- Pixel Loop for Standard Error & Weighting ---
    # Determine indices to iterate over for SE calculation
    if roi_mask is not None:
        roi_indices = np.argwhere(roi_mask)
        iterator = tqdm(range(roi_indices.shape[0]), desc="Calculating SE & Weight (ROI)", leave=False, ncols=80)
        get_coords = lambda idx: roi_indices[idx]
    else:
        total_pixels = H * W
        iterator = tqdm(range(total_pixels), desc="Calculating SE & Weight (Full)", leave=False, ncols=80)
        get_coords = lambda idx: np.unravel_index(idx, (H, W))

    for idx in iterator:
        r, c = get_coords(idx)
        s = slope_map[r, c] # Get pre-calculated slope

        # Skip if slope is zero or has wrong direction already
        if abs(s) < 1e-9 or s * envir_para >= 0:
            continue

        # Need original (smoothed) data for SE calculation
        y = frames_focus_smoothed[r, c, :]
        # Simple SE estimation using slope (less accurate than linregress but faster)
        # Or call linregress just for stderr (still slow)
        # Let's call linregress for now for accuracy, accepting slowness here
        try:
            # Ensure we use the correct time vector length matching y
            current_t = t_smoothed[:len(y)]
            if len(current_t) < 2: continue

            res = sp_stats.linregress(current_t, y)
            if np.isfinite(res.stderr):
                se = res.stderr
                # Apply weighting using the calculated slope (s) and stderr (se)
                rel_err = abs(se / s) # s is non-zero here
                weight = np.exp(-rel_err * errorweight)
                final_activity_map[r, c] = (abs(s) * weight)**augment
        except ValueError: pass # Linregress failed
        except Exception as e_lr: print(f"\nWarning: linregress/SE failed at ({r},{c}): {e_lr}"); pass


    return final_activity_map

def extract_hotspot_from_map(activity_map, threshold_quantile=0.95, morphology_op='close',
                             apply_blur=False, blur_kernel_size=(3, 3),
                             roi_mask=None):
    # (Keep the function from the previous version with ROI logic)
    # ... (exact same implementation as in the previous answer) ...
    if activity_map is None or activity_map.size == 0: return None, np.nan
    if roi_mask is not None:
        if roi_mask.shape != activity_map.shape: print("Warning: ROI mask shape mismatch."); map_to_process = activity_map.copy()
        else: map_to_process = activity_map.copy(); map_to_process[~roi_mask] = np.nan
    else: map_to_process = activity_map.copy() # No ROI mask applied here either
    if apply_blur:
        if np.all(np.isnan(map_to_process)): print("Warning: map all NaN after ROI, skipping blur.")
        else:
            try:
                k_h = blur_kernel_size[0] + (1 - blur_kernel_size[0] % 2); k_w = blur_kernel_size[1] + (1 - blur_kernel_size[1] % 2)
                map_to_process = cv2.GaussianBlur(map_to_process, (k_w, k_h), 0)
            except Exception as e: print(f"Warning: GaussianBlur failed: {e}.")
    proc_map = map_to_process; nan_mask = np.isnan(proc_map)
    if np.all(nan_mask): print("Warning: Activity map all NaNs after ROI/blur."); return None, np.nan
    valid_pixels = proc_map[~nan_mask]
    if valid_pixels.size == 0: print("Warning: No valid pixels found."); return np.zeros_like(activity_map, dtype=bool), 0.0
    map_std = np.std(valid_pixels); map_max = np.max(valid_pixels)
    if map_max < 1e-9 or map_std < 1e-9: return np.zeros_like(activity_map, dtype=bool), 0.0
    try: threshold_value = np.percentile(valid_pixels, threshold_quantile * 100)
    except IndexError: threshold_value = map_max
    if threshold_value <= 1e-9 and map_max > 1e-9: threshold_value = 1e-9
    binary_mask = (np.nan_to_num(proc_map, nan=-np.inf) >= threshold_value).astype(np.uint8)
    mask_processed = binary_mask.copy(); kernel_size_dim = max(min(3, activity_map.shape[0]//20, activity_map.shape[1]//20), 1)
    kernel_close = np.ones((kernel_size_dim, kernel_size_dim), np.uint8); kernel_open = np.ones((2, 2), np.uint8)
    try: # Apply morphology
        if morphology_op == 'close': mask_processed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
        elif morphology_op == 'open_close': mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open); mask_processed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
        elif morphology_op != 'none': print(f"Warning: Unknown morph_op '{morphology_op}'. Using 'none'."); mask_processed = binary_mask
    except cv2.error as e: print(f"Warning: Morphology error: {e}. Using raw binary mask."); mask_processed = binary_mask
    try: # Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_processed, connectivity=8)
    except cv2.error as e: print(f"Error in connectedComponents: {e}."); return None, np.nan
    hotspot_mask = np.zeros_like(activity_map, dtype=bool); hotspot_area = 0.0
    if num_labels > 1 and stats.shape[0] > 1:
        foreground_stats = stats[1:];
        if foreground_stats.size > 0:
            largest_label_idx = np.argmax(foreground_stats[:, cv2.CC_STAT_AREA]); largest_label = largest_label_idx + 1
            hotspot_mask = (labels == largest_label); hotspot_area = np.sum(hotspot_mask)
    return hotspot_mask, hotspot_area


# --- Main Processing Function ---
def generate_masks_for_dataset(dataset_root_dir, output_base_dir, mat_key, fps, focus_duration_sec,
                               quantile, morphology_op, apply_blur, blur_kernel_size,
                               envir_para, errorweight, augment, smooth_window,
                               fuselevel, normalizeT,
                               roi_border_percent=None): # Default to None for optional ROI
    """Finds .mat files, generates masks using weighted slope, and saves them."""
    print(f"Starting mask generation process...")
    print(f"Dataset Root: {dataset_root_dir}")
    print(f"Mask Output Base: {output_base_dir}")
    print(f"--- Localization Parameters (Using Weighted Slope) ---")
    # (Keep print statements for parameters)
    print(f"  Focus Duration: {focus_duration_sec}s | FPS: {fps}")
    print(f"  Quantile: {quantile:.3f} | Morphology: {morphology_op}")
    print(f"  Blur: {apply_blur} | Fuse Level: {fuselevel} | Temporal Smooth: {smooth_window}")
    print(f"  NormalizeT: {normalizeT} | Envir: {envir_para} | ErrWeight: {errorweight} | Augment: {augment}")
    # Print ROI status
    if roi_border_percent is not None:
        print(f"  ROI Border: {roi_border_percent*100:.0f}%")
    else:
        print("  ROI Border: None (processing full frame)")
    print("-" * 30)

    processed_count = 0
    error_count = 0

    for root, _, files in os.walk(dataset_root_dir):
        # Check if current directory matches expected FanPower structure 
        # folder_name = os.path.basename(root)
        # if not folder_name.lower().startswith("fanpower_"): continue # Skip if not FanPower dir

        for mat_file in fnmatch.filter(files, '*.mat'):
            mat_file_path = os.path.join(root, mat_file)
            relative_path_dir = os.path.relpath(root, dataset_root_dir)
            print(f"\nProcessing: {os.path.join(relative_path_dir, mat_file)}")

            # --- Load Data ---
            try:
                mat_data = scipy.io.loadmat(mat_file_path)
                if mat_key not in mat_data: raise KeyError(f"Key '{mat_key}' not found.")
                frames_raw = mat_data[mat_key].astype(np.float64)
                if frames_raw.ndim != 3 or frames_raw.shape[2] < 2: raise ValueError("Invalid frame data.")
                H, W, num_frames = frames_raw.shape
            except Exception as e:
                print(f"  Error loading data: {e}. Skipping file.")
                error_count += 1
                continue

            # --- Apply Pre-processing ---
            frames_proc = apply_preprocessing(frames_raw, normalizeT, fuselevel)

            # --- Calculate Focus Window ---
            focus_duration_frames = int(focus_duration_sec * fps)
            focus_duration_frames = max(2, min(focus_duration_frames, num_frames))
            frames_focus = frames_proc[:, :, :focus_duration_frames]

            # --- Define ROI Mask (Conditional) ---
            roi_mask = None # Default to no ROI
            if roi_border_percent is not None:
                border_h_roi = int(H * roi_border_percent)
                border_w_roi = int(W * roi_border_percent)
                roi_mask = np.zeros((H, W), dtype=bool)
                roi_mask[border_h_roi:-border_h_roi, border_w_roi:-border_w_roi] = True
                # print(f"  Defined ROI mask") # Reduce verbosity

            # --- Calculate Activity Map (Weighted Slope) ---
            # Pass the potentially None roi_mask
            activity_map = calculate_weighted_slope_activity_map(
                frames_focus, fps, smooth_window, envir_para, errorweight, augment, roi_mask
            )

            if activity_map is None or not np.any(np.isfinite(activity_map)):
                print("  Error: Failed to generate valid activity map. Skipping mask.")
                error_count += 1
                continue

            # --- Extract Hotspot Mask ---
            # Pass the potentially None roi_mask again for consistency in processing steps
            hotspot_mask, hotspot_area = extract_hotspot_from_map(
                activity_map, quantile, morphology_op, apply_blur, tuple(blur_kernel_size), roi_mask
            )

            if hotspot_mask is None:
                print("  Error: Failed to extract hotspot mask. Skipping save.")
                error_count += 1
                continue
            elif hotspot_area == 0:
                 print("  Warning: Generated mask has zero area.")

            # --- Save the Mask ---
            try:
                mask_filename = os.path.splitext(mat_file)[0] + '_mask.npy'
                output_subdir = os.path.join(output_base_dir, relative_path_dir)
                os.makedirs(output_subdir, exist_ok=True)
                mask_save_path = os.path.join(output_subdir, mask_filename)
                np.save(mask_save_path, hotspot_mask)
                # print(f"  Saved mask (Area: {hotspot_area:.0f} px)") # Reduce verbosity
                processed_count += 1
            except Exception as e:
                print(f"  Error saving mask: {e}")
                error_count += 1

    print("-" * 30)
    print(f"Mask generation finished.")
    print(f"Successfully generated masks for: {processed_count} files.")
    print(f"Errors/Skipped: {error_count} files.")


# --- Command-Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate hotspot masks from thermal .mat files using weighted slope analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input / Output
    parser.add_argument("output_dir", help="Path to the root directory where masks will be saved.")
    parser.add_argument("-k", "--key", default=config.MAT_FRAMES_KEY, help=f"Key in .mat file (default from config: {config.MAT_FRAMES_KEY}).")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second of the original recording.")

    # Focus Window & Localization Parameters
    parser.add_argument("--focus_duration_sec", type=float, default=10.0, help="Duration (s) from start for analysis.")
    parser.add_argument("-q", "--quantile", type=float, default=0.98, help="Quantile threshold (0-1) for hotspot extraction.")
    parser.add_argument("--morph_op", default="none", choices=['close', 'open_close', 'none'], help="Morphological operation on mask.")
    parser.add_argument("--blur", action='store_true', help="Apply Gaussian blur to activity map before thresholding.")
    parser.add_argument("--blur_kernel", type=int, nargs=2, default=[3, 3], metavar=('H', 'W'), help="Kernel size (odd) for Gaussian blur.")
    # --- MODIFIED ROI Argument ---
    parser.add_argument("--roi_border_percent", type=float, default=None,
                        help="Optional: Percent border (0-0.5) to exclude for ROI. If not provided, no ROI is used.")

    # Pre-processing
    parser.add_argument("--normalizeT", type=int, default=0, choices=[0, 1], help="Normalize temperature per frame? (0=No, 1=Yes).")
    parser.add_argument("--fuselevel", type=int, default=0, help="Spatial fuse level (box filter radius, 0=None).")
    parser.add_argument("--smooth_window", type=int, default=1, help="Temporal smoothing window size (moving avg before slope fit).")

    # Slope Weighting Parameters
    parser.add_argument("--envir_para", type=int, default=-1, choices=[-1, 1], help="Environment parameter (-1=Winter, 1=Summer).")
    parser.add_argument("--errorweight", type=float, default=0.5, help="Weighting factor for slope standard error.")
    parser.add_argument("--augment", type=float, default=1.0, help="Exponent for augmenting final weighted slope.")

    if len(sys.argv) < 2: parser.print_help(); sys.exit(1) # Need at least output_dir
    args = parser.parse_args()

    # --- Argument Validation ---
    if not 0.0 < args.quantile <= 1.0: print("Error: Quantile must be > 0 and <= 1."); sys.exit(1)
    if args.focus_duration_sec <= 0: print("Error: Focus duration must be positive."); sys.exit(1)
    if args.fps <= 0: print("Error: FPS must be positive."); sys.exit(1)
    if args.blur:
        blur_k_h, blur_k_w = args.blur_kernel
        if blur_k_h <= 0 or blur_k_w <= 0: print("Error: Blur kernel dimensions must be positive."); sys.exit(1)
        if blur_k_h % 2 == 0 or blur_k_w % 2 == 0: print("Warning: Blur kernel dimensions should ideally be odd.")
    # --- UPDATED ROI Validation ---
    if args.roi_border_percent is not None and not 0.0 <= args.roi_border_percent < 0.5:
        print("Error: ROI border percent must be >= 0 and < 0.5 (or not provided)."); sys.exit(1)
    # (Keep other validations: fuselevel, smooth_window, errorweight, augment)
    if args.fuselevel < 0: print("Error: Fuse level cannot be negative."); sys.exit(1)
    if args.smooth_window < 1: print("Error: Smooth window must be at least 1."); sys.exit(1)
    if args.errorweight < 0: print("Error: Error weight cannot be negative."); sys.exit(1)
    if args.augment <= 0: print("Error: Augment exponent must be positive."); sys.exit(1)


    # --- Get Input Directory from Config ---
    input_dataset_dir = config.DATASET_FOLDER
    if not input_dataset_dir or not os.path.isdir(input_dataset_dir):
         print(f"Error: Dataset folder specified in config.py not found: {input_dataset_dir}"); sys.exit(1)

    # --- Set Output Base Directory ---
    output_mask_dir = args.output_dir
    os.makedirs(output_mask_dir, exist_ok=True)

    # Run the processing
    generate_masks_for_dataset(
        dataset_root_dir=input_dataset_dir,
        output_base_dir=output_mask_dir,
        mat_key=args.key,
        fps=args.fps,
        focus_duration_sec=args.focus_duration_sec,
        quantile=args.quantile,
        morphology_op=args.morph_op,
        apply_blur=args.blur,
        blur_kernel_size=tuple(args.blur_kernel),
        envir_para=args.envir_para,
        errorweight=args.errorweight,
        augment=args.augment,
        smooth_window=args.smooth_window,
        fuselevel=args.fuselevel,
        normalizeT=args.normalizeT,
        roi_border_percent=args.roi_border_percent # Pass potentially None value
    )

    print("Mask generation script finished.")