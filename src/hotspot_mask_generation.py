# hotspot_mask_generation.py
"""
Pre-processing script to generate and save hotspot masks for thermal .mat files.

Iterates through dataset, applies localization based on focused,
DIRECTION-FILTERED and STATISTICALLY-FILTERED (p-value) linear slope analysis,
and saves boolean masks. Optionally applies ROI. SLOWER due to pixel loop.
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
    # (Keep function as before)
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
                frames_fused[:, :, i] = cv2.boxFilter(
                    frames_proc[:, :, i], -1, (kernel_size, kernel_size))
            frames_proc = frames_fused
        except Exception as e:
            print(f"Warning: Spatial fuse failed: {e}")
    return frames_proc

# --- Activity Map Calculation with P-value Filter ---
def calculate_filtered_slope_activity_map(frames_focus, fps, smooth_window,
                                          envir_para, augment, p_value_threshold,
                                          roi_mask=None):  # roi_mask is for iteration bounds
    """
    Calculates activity map based on direction-filtered AND p-value filtered slope magnitude.
    Uses pixel-wise loop for linregress to get p-value. Slower.
    """
    H, W, T_focus = frames_focus.shape
    final_activity_map = np.zeros((H, W), dtype=np.float64)

    if smooth_window > 1:
        frames_focus_smoothed = np.zeros_like(frames_focus)
        for r in range(H):
            for c in range(W):
                smoothed_series = np.convolve(frames_focus[r, c, :], np.ones(
                    smooth_window)/smooth_window, mode='valid')
                pad_len_before = (T_focus - len(smoothed_series)) // 2
                pad_len_after = T_focus - len(smoothed_series) - pad_len_before
                if len(smoothed_series) > 0:
                    frames_focus_smoothed[r, c, :] = np.pad(
                        smoothed_series, (pad_len_before, pad_len_after), mode='edge')
                else:
                    frames_focus_smoothed[r, c, :] = np.nan
        t_smoothed = (np.arange(T_focus) / fps).astype(np.float64)
    else:
        frames_focus_smoothed = frames_focus
        t_smoothed = (np.arange(T_focus) / fps).astype(np.float64)

    if T_focus < 2:
        print("Error: Less than 2 time points.")
        return None

    # Determine iteration indices
    if roi_mask is not None:
        roi_indices = np.argwhere(roi_mask)
        iterator = tqdm(range(
            roi_indices.shape[0]), desc="Calculating Filtered Slope (ROI)", leave=False, ncols=80)

        def get_coords(idx): return roi_indices[idx]
    else:
        total_pixels = H * W
        iterator = tqdm(range(
            total_pixels), desc="Calculating Filtered Slope (Full)", leave=False, ncols=80)

        def get_coords(idx): return np.unravel_index(idx, (H, W))

    for idx in iterator:
        r, c = get_coords(idx)
        y = frames_focus_smoothed[r, c, :]
        valid_mask_y = np.isfinite(y)
        y_valid = y[valid_mask_y]
        t_valid = t_smoothed[valid_mask_y]

        if len(y_valid) < 2:
            continue
        try:
            res = sp_stats.linregress(t_valid, y_valid)
            passes_filter = False
            if np.isfinite(res.slope) and np.isfinite(res.pvalue):
                if res.slope * envir_para < 0 and res.pvalue < p_value_threshold:
                    passes_filter = True
            if passes_filter:
                activity_value = np.abs(res.slope)
                if augment != 1.0:
                    activity_value = np.power(activity_value, augment)
                final_activity_map[r, c] = activity_value
        except ValueError:
            pass
        except Exception as e_lr:
            print(f"\nWarn: linregress/filter failed at ({r},{c}): {e_lr}")
            pass
    return final_activity_map


def extract_hotspot_from_map(activity_map, threshold_quantile=0.95, morphology_op='close',
                             apply_blur=False, blur_kernel_size=(3, 3),
                             roi_mask=None):
    if activity_map is None or activity_map.size == 0:
        return None, np.nan
    if roi_mask is not None:
        if roi_mask.shape != activity_map.shape:
            print("Warning: ROI mask shape mismatch.")
            map_to_process = activity_map.copy()
        else:
            map_to_process = activity_map.copy()
            map_to_process[~roi_mask] = np.nan
    else:
        map_to_process = activity_map.copy()
    if apply_blur:
        if np.all(np.isnan(map_to_process)):
            print("Warning: map all NaN after ROI, skipping blur.")
        else:
            try:
                k_h = blur_kernel_size[0] + (1 - blur_kernel_size[0] % 2)
                k_w = blur_kernel_size[1] + (1 - blur_kernel_size[1] % 2)
                map_to_process = cv2.GaussianBlur(
                    map_to_process, (k_w, k_h), 0)
            except Exception as e:
                print(f"Warning: GaussianBlur failed: {e}.")
    proc_map = map_to_process
    nan_mask = np.isnan(proc_map)
    if np.all(nan_mask):
        print("Warning: Activity map all NaNs after ROI/blur.")
        return None, np.nan

    valid_pixels = proc_map[~nan_mask]
    if valid_pixels.size == 0:
        print("Warning: No valid pixels found.")
        return np.zeros_like(activity_map, dtype=bool), 0.0

    map_std = np.std(valid_pixels)
    map_max = np.max(valid_pixels)
    if map_max < 1e-9 or map_std < 1e-9:
        return np.zeros_like(activity_map, dtype=bool), 0.0
    try:
        threshold_value = np.percentile(valid_pixels, threshold_quantile * 100)
    except IndexError:
        threshold_value = map_max

    if threshold_value <= 1e-9 and map_max > 1e-9:
        threshold_value = 1e-9

    binary_mask = (np.nan_to_num(proc_map, nan=-np.inf)
                   >= threshold_value).astype(np.uint8)
    if not np.any(binary_mask):
        return np.zeros_like(activity_map, dtype=bool), 0.0

    mask_processed = binary_mask.copy()
    kernel_size_dim = max(
        min(3, activity_map.shape[0]//20, activity_map.shape[1]//20), 1)
    kernel_close = np.ones((kernel_size_dim, kernel_size_dim), np.uint8)
    kernel_open = np.ones((2, 2), np.uint8)
    try:
        if morphology_op == 'close':
            mask_processed = cv2.morphologyEx(
                binary_mask, cv2.MORPH_CLOSE, kernel_close)
        elif morphology_op == 'open_close':
            mask_opened = cv2.morphologyEx(
                binary_mask, cv2.MORPH_OPEN, kernel_open)
            mask_processed = cv2.morphologyEx(
                mask_opened, cv2.MORPH_CLOSE, kernel_close)
        elif morphology_op != 'none':
            print(
                f"Warning: Unknown morph_op '{morphology_op}'. Using 'none'.")
    except cv2.error as e:
        print(f"Warning: Morphology error: {e}. Using raw binary mask.")
        mask_processed = binary_mask
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_processed, connectivity=8)
    except cv2.error as e:
        print(f"Error in connectedComponents: {e}.")
        return None, np.nan

    hotspot_mask = np.zeros_like(activity_map, dtype=bool)
    hotspot_area = 0.0
    if num_labels > 1 and stats.shape[0] > 1:
        foreground_stats = stats[1:]
        if foreground_stats.size > 0:
            largest_label_idx = np.argmax(
                foreground_stats[:, cv2.CC_STAT_AREA])
            largest_label = largest_label_idx + 1
            hotspot_mask = (labels == largest_label)
            hotspot_area = np.sum(hotspot_mask)
    return hotspot_mask, hotspot_area

# --- Main Processing Function ---
def generate_masks_for_dataset(dataset_root_dir, output_base_dir, mat_key, fps, focus_duration_sec,
                               quantile, morphology_op, apply_blur, blur_kernel_size,
                               envir_para, augment, smooth_window, p_value_threshold,
                               fuselevel, normalizeT,
                               roi_border_percent=None):  # Default to None for optional ROI
    """Finds .mat files, generates masks using p-value filtered slope, and saves them."""
    print(f"Starting mask generation process...")
    print(f"Dataset Root: {dataset_root_dir}")
    print(f"Mask Output Base: {output_base_dir}")
    print(f"--- Localization Parameters (Using P-Value Filtered Slope) ---")
    print(
        f"  Focus Duration: {focus_duration_sec}s | FPS: {fps} | Smooth: {smooth_window}")
    print(
        f"  P-Value Thresh: {p_value_threshold} | Quantile: {quantile:.3f} | Morphology: {morphology_op}")
    print(
        f"  Blur: {apply_blur} | Fuse Level: {fuselevel} | NormalizeT: {normalizeT}")
    print(f"  Envir: {envir_para} | Augment: {augment}")
    if roi_border_percent is not None:
        print(f"  ROI Border: {roi_border_percent*100:.0f}%")
    else:
        print("  ROI Border: None (slope on full, extraction uses ROI if defined)")
    print("-" * 30)

    processed_count = 0
    error_count = 0

    for root, _, files in os.walk(dataset_root_dir):
        for mat_file in fnmatch.filter(files, '*.mat'):
            mat_file_path = os.path.join(root, mat_file)
            relative_path_dir = os.path.relpath(root, dataset_root_dir)
            print(f"\nProcessing: {os.path.join(relative_path_dir, mat_file)}")

            # Load Data
            try:
                mat_data = scipy.io.loadmat(mat_file_path)
                frames_raw = mat_data[mat_key].astype(np.float64)
                if frames_raw.ndim != 3 or frames_raw.shape[2] < 2:
                    raise ValueError("Invalid frame data.")
                H, W, num_frames = frames_raw.shape
            except Exception as e:
                print(f"  Error loading data: {e}. Skip.")
                error_count += 1
                continue

            # Pre-processing
            frames_proc = apply_preprocessing(
                frames_raw, normalizeT, fuselevel)
            # Focus Window
            focus_duration_frames = int(focus_duration_sec * fps)
            focus_duration_frames = max(
                2, min(focus_duration_frames, num_frames))
            frames_focus = frames_proc[:, :, :focus_duration_frames]

            # --- Define ROI Mask ---
            # If roi_border_percent is None, roi_mask_for_slope_calc will be None.
            # extract_hotspot_from_map will then use its own internal roi_mask if roi_border_percent is not None.
            roi_mask_for_slope_calc = None
            if roi_border_percent is not None:
                border_h = int(H*roi_border_percent)
                border_w = int(W*roi_border_percent)
                roi_mask_for_slope_calc = np.zeros((H, W), dtype=bool)
                roi_mask_for_slope_calc[border_h:-
                                        border_h, border_w:-border_w] = True

            # --- Calculate Activity Map (P-Value Filtered Slope) ---
            activity_map = calculate_filtered_slope_activity_map(
                frames_focus, fps, smooth_window, envir_para, augment, p_value_threshold, roi_mask_for_slope_calc
            )

            if activity_map is None or not np.any(np.isfinite(activity_map)):
                print("Error: Failed valid activity map generation.")
                error_count += 1
                continue

            # --- Extract Hotspot Mask ---
            hotspot_mask, hotspot_area = extract_hotspot_from_map(
                activity_map, quantile, morphology_op, apply_blur, tuple(
                    blur_kernel_size), roi_mask_for_slope_calc
            )

            if hotspot_mask is None:
                print("Error: Failed mask extraction.")
                error_count += 1
                continue
            elif hotspot_area == 0:
                print("Warning: Zero area mask generated.")

            # --- Save Mask ---
            try:
                mask_filename = os.path.splitext(mat_file)[0] + '_mask.npy'
                output_subdir = os.path.join(
                    output_base_dir, relative_path_dir)
                os.makedirs(output_subdir, exist_ok=True)
                mask_save_path = os.path.join(output_subdir, mask_filename)
                np.save(mask_save_path, hotspot_mask)
                processed_count += 1
            except Exception as e:
                print(f"Error saving mask: {e}")
                error_count += 1

    print("-" * 30)
    print(
        f"Mask generation finished. Success: {processed_count}, Errors: {error_count}")


# --- Command-Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate hotspot masks using p-value filtered slope analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # IO Args
    parser.add_argument("output_dir", help="Root directory to save masks.")
    parser.add_argument("-k", "--key", default=config.MAT_FRAMES_KEY,
                        help=f"Key in .mat file (default from config).")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frames per second of original recording.")

    # Slope Analysis Params
    parser.add_argument("--focus_duration_sec", type=float,
                        default=10.0, help="Duration (s) for slope analysis.")
    parser.add_argument("--smooth_window", type=int, default=1,
                        help="Temporal smoothing window for slope fit (1=None).")
    parser.add_argument("--p_value_thresh", type=float, default=0.10,
                        help="P-value threshold for slope significance filter.")
    parser.add_argument("--envir_para", type=int, default=-1,
                        choices=[-1, 1], help="Environment parameter (-1=Winter, 1=Summer).")
    parser.add_argument("--augment", type=float, default=1.0,
                        help="Exponent for augmenting filtered slope magnitude.")

    # Pre-processing Args
    parser.add_argument("--normalizeT", type=int, default=0,
                        choices=[0, 1], help="Normalize temperature per frame? (0=No, 1=Yes).")
    parser.add_argument("--fuselevel", type=int, default=0,
                        help="Spatial fuse level (0=None).")
    parser.add_argument("--roi_border_percent", type=float, default=0.20,  # Defaulting to apply ROI
                        help="Percent border (0-0.5) to exclude for ROI. Set to large value like 0.99 to disable (effectively no ROI).")

    # Hotspot Extraction Args
    parser.add_argument("-q", "--quantile", type=float,
                        default=0.995, help="Quantile threshold (0-1).")
    parser.add_argument("--morph_op", default="none",
                        choices=['close', 'open_close', 'none'], help="Morphological operation.")
    parser.add_argument("--blur", action='store_true',
                        help="Apply Gaussian blur to activity map.")
    parser.add_argument("--blur_kernel", type=int, nargs=2,
                        default=[3, 3], metavar=('H', 'W'), help="Kernel size for Gaussian blur.")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # --- Argument Validation ---
    if not 0.0 < args.quantile <= 1.0:
        print("Error: Quantile invalid.")
        sys.exit(1)
    if args.focus_duration_sec <= 0:
        print("Error: Focus duration invalid.")
        sys.exit(1)
    if args.fps <= 0:
        print("Error: FPS invalid.")
        sys.exit(1)
    if args.blur:
        kh, kw = args.blur_kernel
        assert kh > 0 and kw > 0
        assert kh % 2 != 0 and kw % 2 != 0, "Blur kernel dims must be odd"
    if args.roi_border_percent is not None and not 0.0 <= args.roi_border_percent < 0.5:
        print("Error: ROI border percent must be >= 0 and < 0.5 if specified.")
        sys.exit(1)
    if args.fuselevel < 0:
        print("Error: Fuse level invalid.")
        sys.exit(1)
    if args.smooth_window < 1:
        print("Error: Smooth window invalid.")
        sys.exit(1)
    if not 0.0 < args.p_value_thresh < 1.0:
        print("Error: P-value threshold invalid.")
        sys.exit(1)
    if args.augment <= 0:
        print("Error: Augment invalid.")
        sys.exit(1)

    # Get Input Dir from Config
    input_dataset_dir = config.DATASET_FOLDER
    if not input_dataset_dir or not os.path.isdir(input_dataset_dir):
        print(
            f"Error: Dataset folder specified in config.py not found: {input_dataset_dir}")
        sys.exit(1)

    # Set Output Dir
    output_mask_dir = args.output_dir
    os.makedirs(output_mask_dir, exist_ok=True)

    # Run processing
    generate_masks_for_dataset(
        dataset_root_dir=input_dataset_dir, output_base_dir=output_mask_dir,
        mat_key=args.key, fps=args.fps,
        focus_duration_sec=args.focus_duration_sec,
        quantile=args.quantile, morphology_op=args.morph_op,
        apply_blur=args.blur, blur_kernel_size=tuple(args.blur_kernel),
        envir_para=args.envir_para, augment=args.augment,
        smooth_window=args.smooth_window,
        p_value_threshold=args.p_value_thresh,
        fuselevel=args.fuselevel, normalizeT=args.normalizeT,
        roi_border_percent=args.roi_border_percent
    )

    print("Mask generation script finished.")

