# debug_single_mask.py
"""
Standalone script to debug and visualize hotspot mask generation for a SINGLE
thermal .mat file. Uses the p-value filtered slope method.
Allows quick iteration on parameters for problematic files.
Saves generated plots to an output directory.
Includes printing of max activity coordinates and threshold value.
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
import matplotlib.pyplot as plt

try:
    import config
except ImportError:
    print("Warning: Failed to import config.py. MAT_FRAMES_KEY might use default.")
    class MockConfig:
        MAT_FRAMES_KEY = "TempFrames"
        DATASET_FOLDER = "./dataset_new" # Placeholder
    config = MockConfig()

# --- Helper Functions (Copied and potentially refined from hotspot_mask_generation.py) ---
def apply_preprocessing(frames, normalizeT, fuselevel):
    # (Identical to the one in your previous debug script)
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


def calculate_filtered_slope_activity_map(frames_focus, fps, smooth_window,
                                          envir_para, augment, p_value_threshold,
                                          roi_mask=None):
    # (Identical to the one in your previous debug script with corrected slope logic)
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
        print("Error: Less than 2 time points for slope calculation.")
        return None

    iterator_desc = "Calculating Filtered Slope"
    if roi_mask is not None:
        iterator_desc += " (ROI)"
        roi_indices = np.argwhere(roi_mask)
        iterator_range = range(roi_indices.shape[0])
        def get_coords(idx): return roi_indices[idx]
    else:
        iterator_desc += " (Full)"
        total_pixels = H * W
        iterator_range = range(total_pixels)
        def get_coords(idx): return np.unravel_index(idx, (H, W))

    for idx in tqdm(iterator_range, desc=iterator_desc, leave=False, ncols=80):
        r, c = get_coords(idx)
        y = frames_focus_smoothed[r, c, :]
        valid_mask_y = np.isfinite(y)
        y_valid = y[valid_mask_y]
        t_valid = t_smoothed[valid_mask_y]
        if len(y_valid) < 2:
            continue
        try:
            res = sp_stats.linregress(t_valid, y_valid)
            is_significant_and_correct_direction = False
            if np.isfinite(res.slope) and np.isfinite(res.pvalue):
                if (res.slope * envir_para > 0) and (res.pvalue < p_value_threshold): # Corrected logic
                    is_significant_and_correct_direction = True

            if is_significant_and_correct_direction:
                activity_value = np.abs(res.slope)
                final_activity_map[r, c] = np.power(
                    activity_value, augment) if augment != 1.0 else activity_value
                
            # if r == 250 and c == 311: # Or whatever the true coordinates are
            #     print(f"  DEBUG AT TRUE LEAK (250,311): Slope={res.slope:.4f}, P-Value={res.pvalue:.4f}, ActivityMapValue={final_activity_map[r,c]:.4f}")
        except ValueError:
            pass
        except Exception: # More general exception
            pass
    
    return final_activity_map


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
            print(f"  Debug: Max activity in map_to_process at {max_activity_coords} with value {map_to_process[max_activity_coords]:.4e}")
        except ValueError: # nanargmax raises error if all are NaN
             print("  Debug: All values in map_to_process are NaN before nanargmax.")
    else:
        print("  Debug: map_to_process is all NaN before finding max activity.")


    non_nan_pixels_before_thresh = map_to_process[~np.isnan(map_to_process)]
    if non_nan_pixels_before_thresh.size > 0:
        min_act = np.min(non_nan_pixels_before_thresh)
        max_act = np.max(non_nan_pixels_before_thresh)
        non_zero_count = np.sum(non_nan_pixels_before_thresh > 1e-9)
        print(f"Debug: Activity map before thresholding - Min: {min_act:.2e}, Max: {max_act:.2e}, Non-zero count: {non_zero_count}")
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
            print(f"  Selected component containing max activity pixel. Label: {label_at_max_activity}, Area: {hotspot_area}")
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

# Global variable for consistent colorbar in visualize_overlay
frames_raw_for_colorbar_scope = None

def visualize_overlay(base_image_raw, mask, title="Mask Overlay", colormap_name='inferno', save_path=None):
    # (Keep visualize_overlay as in the previous version, ensuring it uses frames_raw_for_colorbar_scope)
    if base_image_raw is None: print("Error: Base image is None."); return
    if mask is None: mask = np.zeros_like(base_image_raw, dtype=bool)

    try:
        display_img_norm = base_image_raw # Will be normalized if not uint8
        if base_image_raw.dtype != np.uint8:
            min_val, max_val = np.nanmin(base_image_raw), np.nanmax(base_image_raw)
            if max_val > min_val:
                 display_img_norm = cv2.normalize(base_image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else: 
                 display_img_norm = np.full_like(base_image_raw, 128, dtype=np.uint8)
        
        base_image_bgr = cv2.cvtColor(display_img_norm, cv2.COLOR_GRAY2BGR)
        overlay_color_layer = np.zeros_like(base_image_bgr)
        if mask.shape == base_image_bgr.shape[:2]:
            overlay_color_layer[mask] = [0, 255, 255]  # Yellow BGR
        
        blended_img = cv2.addWeighted(overlay_color_layer, 0.4, base_image_bgr, 0.6, 0)
        img_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10 * img_rgb.shape[0]/img_rgb.shape[1] if img_rgb.shape[1] > 0 else 10))
        ax.imshow(img_rgb)

        if frames_raw_for_colorbar_scope is not None: # Use the global raw frames for colorbar
            d_min, d_max = np.nanmin(frames_raw_for_colorbar_scope), np.nanmax(frames_raw_for_colorbar_scope)
            if d_max > d_min:
                cmap_obj = plt.get_cmap(colormap_name); norm_obj = plt.Normalize(vmin=d_min, vmax=d_max)
                sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj); sm.set_array([])
                fig.colorbar(sm, ax=ax, label='Orig Video Temp Scale (°C)', fraction=0.046, pad=0.04)

        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path); print(f"  Saved plot to: {save_path}")
            plt.show()
        else:
            plt.show()
        plt.close(fig) 
    except Exception as e:
        print(f"Error during visualization: {e}"); traceback.print_exc()


def debug_single_file(mat_file_path, output_plot_dir,
                      mat_key, fps, focus_duration_sec,
                      quantile, morphology_op, apply_blur, blur_kernel_size,
                      envir_para, augment, smooth_window, p_value_threshold,
                      fuselevel, normalizeT,
                      roi_border_percent=None,
                      display_frame_index=0, colormap_name='inferno',
                      true_leak_coords=None):
    global frames_raw_for_colorbar_scope

    base_filename = os.path.splitext(os.path.basename(mat_file_path))[0]
    print(f"--- Debugging: {base_filename} ---")
    param_summary = (
        f"FocusDur:{focus_duration_sec}s, SmoothWin:{smooth_window}, PVal:{p_value_threshold}, "
        f"Quantile:{quantile:.3f}, Morph:{morphology_op}, ROI:{roi_border_percent*100 if roi_border_percent is not None else 'None'}%, "
        f"Envir:{envir_para}, Blur:{apply_blur}"
    )
    print(param_summary)
    print("-" * 30)

    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        frames_raw_for_colorbar_scope = mat_data[mat_key].astype(np.float64)
        frames_raw = frames_raw_for_colorbar_scope.copy()
        if frames_raw.ndim != 3 or frames_raw.shape[2] < 2: raise ValueError("Invalid frame data.")
        H, W, num_frames = frames_raw.shape
    except Exception as e: print(f"Error loading data: {e}. Exiting."); return

    frames_proc = apply_preprocessing(frames_raw, normalizeT, fuselevel)
    focus_duration_frames = int(focus_duration_sec * fps)
    focus_duration_frames = max(2, min(focus_duration_frames, num_frames))
    frames_focus = frames_proc[:, :, :focus_duration_frames]

    current_roi_mask = None
    if roi_border_percent is not None and 0.0 <= roi_border_percent < 0.5:
        border_h = int(H * roi_border_percent); border_w = int(W * roi_border_percent)
        if H - 2 * border_h > 0 and W - 2 * border_w > 0:
            current_roi_mask = np.zeros((H, W), dtype=bool)
            current_roi_mask[border_h:H-border_h, border_w:W-border_w] = True

    if true_leak_coords:
        print("\\n--- Plotting Pixel Time Series ---")
        # ... (Keep the time series plotting logic from previous version) ...
        time_vector_full = (np.arange(num_frames) / fps).astype(np.float64)
        time_vector_focus = (np.arange(focus_duration_frames) / fps).astype(np.float64)
        plt.figure(figsize=(14, 10))
        r_tl, c_tl = true_leak_coords
        r_tl = min(r_tl, H-1); c_tl = min(c_tl, W-1)

        ax1 = plt.subplot(2,1,1)
        ax1.plot(time_vector_full, frames_raw[r_tl, c_tl, :], label=f'True Leak Loc ({r_tl},{c_tl}) Raw Full')
        ax1.set_xlabel("Time (s) - Full"); ax1.set_ylabel("Raw Temp (°C)"); ax1.set_title(f"Raw Full @ ({r_tl},{c_tl})"); ax1.legend(); ax1.grid(True)

        ax2 = plt.subplot(2,1,2)
        ax2.plot(time_vector_focus, frames_focus[r_tl, c_tl, :], label=f'True Leak ({r_tl},{c_tl}) Focus PostProc')
        if smooth_window > 1 and len(frames_focus[r_tl,c_tl,:]) >= smooth_window:
            smoothed_series = np.convolve(frames_focus[r_tl,c_tl,:], np.ones(smooth_window)/smooth_window, mode='valid')
            pad_len_total = focus_duration_frames - len(smoothed_series)
            pad_len_before = pad_len_total // 2; pad_len_after = pad_len_total - pad_len_before
            if len(smoothed_series)>0 and pad_len_before >= 0 and pad_len_after >=0:
                 ax2.plot(time_vector_focus, np.pad(smoothed_series, (pad_len_before, pad_len_after), mode='edge'), 
                          label=f'True Leak ({r_tl},{c_tl}) Smoothed (win={smooth_window})', linestyle='--')
        ax2.set_xlabel(f"Time (s) - Focus ({focus_duration_sec}s)"); ax2.set_ylabel("Temp (°C) PostProc"); ax2.set_title(f"Focus @ ({r_tl},{c_tl}) (Smoothed if win>1)"); ax2.legend(); ax2.grid(True)
        
        plt.suptitle(f"Pixel Time Series for {base_filename}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        ts_plot_save_path = os.path.join(output_plot_dir, f"{base_filename}_pixel_timeseries.png")
        plt.savefig(ts_plot_save_path); print(f"  Saved time series plot to: {ts_plot_save_path}")
        plt.close()


    print("Calculating activity map...")
    activity_map = calculate_filtered_slope_activity_map(
        frames_focus, fps, smooth_window, envir_para, augment, p_value_threshold, current_roi_mask
    )

    if activity_map is None or not np.any(np.isfinite(activity_map[current_roi_mask if current_roi_mask is not None else ...])):
        print("Error: Failed valid activity map. Cannot proceed."); return

    activity_map_plot_save_path = os.path.join(output_plot_dir, f"{base_filename}_activity_map.png")
    plt.figure(figsize=(8,7)); plt.imshow(activity_map, cmap='hot'); plt.colorbar(label="Activity (abs_slope)")
    plt.title(f"Activity Map - {base_filename}\nParams: {param_summary}", fontsize=9) # Shorter title
    plt.savefig(activity_map_plot_save_path); print(f"  Saved activity map plot to: {activity_map_plot_save_path}")
    plt.close()
    
    print("Extracting hotspot mask...")
    hotspot_mask, hotspot_area = extract_hotspot_from_map(
        activity_map, quantile, morphology_op, apply_blur, tuple(blur_kernel_size), current_roi_mask
    )

    if hotspot_mask is None: print("Error: Failed mask extraction."); return
    print(f"Generated mask area: {hotspot_area:.0f} pixels")

    mask_plot_save_path = os.path.join(output_plot_dir, f"{base_filename}_mask_overlay.png")
    frame_to_display_raw = frames_raw[:, :, min(display_frame_index, num_frames-1)]
    
    viz_title = (f"{base_filename} | Fr{min(display_frame_index, num_frames-1)} | Mask Area: {hotspot_area:.0f}px\n"
                 f"{param_summary}") # Use param_summary for title
    visualize_overlay(frame_to_display_raw, hotspot_mask, title=viz_title, colormap_name=colormap_name, save_path=mask_plot_save_path)
    
    try:
        mask_filename_npy = f"{base_filename}_DEBUG_mask.npy"
        mask_save_path_npy = os.path.join(output_plot_dir, mask_filename_npy)
        np.save(mask_save_path_npy, hotspot_mask)
        print(f"  Saved debug mask to: {mask_save_path_npy}")
    except Exception as e:
        print(f"  Error saving debug mask .npy: {e}")

    print("-" * 30 + f"\nDebug run for {base_filename} finished." + "-"*30 )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug hotspot mask generation for a SINGLE .mat file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("mat_file", help="Path to the specific .mat file.")
    parser.add_argument("output_plot_dir", help="Directory to save plots and debug mask.")

    parser.add_argument("-k", "--key", default=config.MAT_FRAMES_KEY if hasattr(config, 'MAT_FRAMES_KEY') else 'TempFrames', help="Key in .mat file.")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second.")
    parser.add_argument("--focus_duration_sec", type=float, default=5.0, help="Duration (s) for slope analysis.")
    parser.add_argument("--smooth_window", type=int, default=3, help="Temporal smoothing window (1=None).")
    parser.add_argument("--p_value_thresh", type=float, default=0.10, help="P-value for slope significance.") # Relaxed for debug
    parser.add_argument("--envir_para", type=int, default=1, choices=[-1, 1], help="Environment (-1=Cool, 1=Heat).") # Default to -1
    parser.add_argument("--augment", type=float, default=1.0, help="Exponent for slope magnitude.")
    parser.add_argument("--normalizeT", type=int, default=0, choices=[0, 1], help="Normalize T per frame?")
    parser.add_argument("--fuselevel", type=int, default=0, help="Spatial fuse level.")
    parser.add_argument("--roi_border_percent", type=float, default=0.0, help="ROI border (0.0-0.49).")
    parser.add_argument("-q", "--quantile", type=float, default=0.98, help="Quantile threshold.") # Lowered for debug
    parser.add_argument("--morph_op", default="open_close", choices=['close', 'open_close', 'none'], help="Morphology op.")
    parser.add_argument("--blur", action='store_true', help="Apply Gaussian blur.")
    parser.add_argument("--blur_kernel", type=int, nargs=2, default=[3, 3], metavar=('H', 'W'), help="Blur kernel (odd nums).")
    parser.add_argument("--display_frame", type=int, default=0, help="Frame index for overlay.")
    parser.add_argument("--colormap", default='inferno', choices=plt.colormaps(), help="Colormap.")
    parser.add_argument("--true_leak_coords", type=int, nargs=2, default=None, metavar=('ROW', 'COL'), help="Optional: (row, col) of true leak for time series plot.")

    if len(sys.argv) < 3: parser.print_help(); sys.exit(1)
    args = parser.parse_args()

    if not os.path.isfile(args.mat_file): print(f"Error: .mat file not found: {args.mat_file}"); sys.exit(1)
    if args.focus_duration_sec <= 0 : print("Error: focus_duration_sec must be positive."); sys.exit(1)
    if args.fps <= 0 : print("Error: fps must be positive."); sys.exit(1)
    if args.roi_border_percent is not None and not (0.0 <= args.roi_border_percent < 0.5):
        print("Error: ROI border percent must be between 0.0 and 0.49."); sys.exit(1)
    if args.blur and not (args.blur_kernel[0] > 0 and args.blur_kernel[1] > 0 and args.blur_kernel[0] % 2 != 0 and args.blur_kernel[1] % 2 != 0):
        print("Error: Blur kernel dimensions must be positive and odd."); sys.exit(1)


    os.makedirs(args.output_plot_dir, exist_ok=True)

    debug_single_file(
        mat_file_path=args.mat_file,
        output_plot_dir=args.output_plot_dir,
        mat_key=args.key, fps=args.fps,
        focus_duration_sec=args.focus_duration_sec,
        quantile=args.quantile, morphology_op=args.morph_op,
        apply_blur=args.blur, blur_kernel_size=tuple(args.blur_kernel),
        envir_para=args.envir_para, augment=args.augment,
        smooth_window=args.smooth_window,
        p_value_threshold=args.p_value_thresh,
        fuselevel=args.fuselevel, normalizeT=args.normalizeT,
        roi_border_percent=args.roi_border_percent,
        display_frame_index=args.display_frame,
        colormap_name=args.colormap,
        true_leak_coords=args.true_leak_coords
    )
