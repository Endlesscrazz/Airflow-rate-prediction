# debug_single_mask.py
"""
Standalone script to debug and visualize hotspot mask generation for a SINGLE
thermal .mat file. Uses the p-value filtered slope method.
Allows quick iteration on parameters for problematic files.
Saves generated plots to an output directory.
Includes printing of max activity coordinates and threshold value.
Highlights true_leak_coords if provided and prints its stats.
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
from matplotlib.patches import Circle # For highlighting true leak

try:
    import config
except ImportError:
    print("Warning: Failed to import config.py. MAT_FRAMES_KEY might use default.")
    class MockConfig:
        MAT_FRAMES_KEY = "TempFrames"
    config = MockConfig()

# --- Helper Functions ---
def apply_preprocessing(frames, normalizeT, fuselevel):
    # (Identical to previous)
    frames_proc = frames.copy()
    if normalizeT:
        frame_means = np.mean(frames_proc, axis=(0, 1), keepdims=True)
        frame_means[frame_means == 0] = 1.0 # or 1e-9 to avoid division by zero strictly
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
                                          roi_mask=None, true_leak_coords_for_debug=None): # Added true_leak_coords_for_debug
    H, W, T_focus = frames_focus.shape
    final_activity_map = np.zeros((H, W), dtype=np.float64)
    # Store stats for the true_leak_coords if provided
    true_leak_stats = {'slope': np.nan, 'p_value': np.nan, 'activity': np.nan, 'passed_filters': False}


    if smooth_window > 1:
        frames_focus_smoothed = np.zeros_like(frames_focus)
        for r_idx in range(H): # Renamed r to r_idx to avoid conflict
            for c_idx in range(W): # Renamed c to c_idx
                # Ensure there are enough points for convolution mode 'valid'
                if T_focus >= smooth_window:
                    smoothed_series = np.convolve(frames_focus[r_idx, c_idx, :], np.ones(
                        smooth_window)/smooth_window, mode='valid')
                    pad_len_before = (T_focus - len(smoothed_series)) // 2
                    pad_len_after = T_focus - len(smoothed_series) - pad_len_before
                    if len(smoothed_series) > 0: 
                        frames_focus_smoothed[r_idx, c_idx, :] = np.pad(
                            smoothed_series, (pad_len_before, pad_len_after), mode='edge')
                    else: 
                        frames_focus_smoothed[r_idx, c_idx, :] = frames_focus[r_idx, c_idx, :] 
                else: 
                    frames_focus_smoothed[r_idx, c_idx, :] = frames_focus[r_idx, c_idx, :]
        t_smoothed = (np.arange(T_focus) / fps).astype(np.float64)

    else:
        frames_focus_smoothed = frames_focus.copy() # Use a copy
        t_smoothed = (np.arange(T_focus) / fps).astype(np.float64)


    if T_focus < 2:
        print("Error: Less than 2 time points for slope calculation.")
        return None, true_leak_stats # Return None for map, and initial stats

    iterator_desc = "Calculating Filtered Slope"
    # Determine iteration indices (Logic for ROI or Full remains the same)
    if roi_mask is not None and roi_mask.dtype == bool and roi_mask.shape == (H,W) and np.any(roi_mask):
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
        r, c = get_coords(idx) # Current pixel coordinates
        y = frames_focus_smoothed[r, c, :]
        valid_mask_y = np.isfinite(y)
        y_valid = y[valid_mask_y]
        t_valid = t_smoothed[valid_mask_y]

        if len(y_valid) < 2:
            if true_leak_coords_for_debug and r == true_leak_coords_for_debug[0] and c == true_leak_coords_for_debug[1]:
                 print(f"  DEBUG AT TRUE LEAK ({r},{c}): Skipped - Not enough valid data points ({len(y_valid)}) after NaN filter.")
            continue
        try:
            res = sp_stats.linregress(t_valid, y_valid)
            is_significant_and_correct_direction = False
            
            slope_ok_filter = False
            p_value_ok_filter = False

            if np.isfinite(res.slope):
                if (res.slope * envir_para > 0):
                    slope_ok_filter = True
            if np.isfinite(res.pvalue):
                if (res.pvalue < p_value_threshold):
                    p_value_ok_filter = True
            
            is_significant_and_correct_direction = slope_ok_filter and p_value_ok_filter

            current_pixel_activity = 0.0
            if is_significant_and_correct_direction:
                activity_value = np.abs(res.slope)
                current_pixel_activity = np.power(activity_value, augment) if augment != 1.0 else activity_value
                final_activity_map[r, c] = current_pixel_activity
            
            # --- DEBUG PRINT FOR TRUE LEAK COORDS ---
            if true_leak_coords_for_debug and r == true_leak_coords_for_debug[0] and c == true_leak_coords_for_debug[1]:
                true_leak_stats['slope'] = res.slope
                true_leak_stats['p_value'] = res.pvalue
                true_leak_stats['activity'] = final_activity_map[r, c] # Will be 0 if filters not passed
                true_leak_stats['passed_filters'] = is_significant_and_correct_direction
                print(f"\n  --- STATS AT TRUE LEAK COORDS ({r},{c}) ---")
                print(f"    Raw Slope: {res.slope:.6f}")
                print(f"    P-Value: {res.pvalue:.6f} (Threshold: < {p_value_threshold}) -> P-Value OK? {p_value_ok_filter}")
                print(f"    Direction OK (Slope * envir_para ({envir_para}) > 0)? {slope_ok_filter}")
                print(f"    Passed Both Filters? {is_significant_and_correct_direction}")
                print(f"    Activity Map Value: {final_activity_map[r, c]:.6f}")
                print(f"  --------------------------------------------")
            # --- END DEBUG PRINT ---

        except ValueError: # From linregress
            if true_leak_coords_for_debug and r == true_leak_coords_for_debug[0] and c == true_leak_coords_for_debug[1]:
                print(f"  DEBUG AT TRUE LEAK ({r},{c}): Linregress ValueError.")
            pass
        except Exception as e_lr_calc: # Other errors
            if true_leak_coords_for_debug and r == true_leak_coords_for_debug[0] and c == true_leak_coords_for_debug[1]:
                 print(f"  DEBUG AT TRUE LEAK ({r},{c}): Unexpected error in linregress/filter: {e_lr_calc}")
            # print(f"\nWarn: linregress/filter failed at ({r},{c}): {e_lr_calc}") # Optional: general warning
            pass
    
    return final_activity_map, true_leak_stats


def extract_hotspot_from_map(activity_map, threshold_quantile=0.95, morphology_op='close',
                             apply_blur=False, blur_kernel_size=(3, 3),
                             roi_mask=None):
    # (This function remains largely the same as your previous version)
    # ... (ensure it returns np.array([]), 0.0 on early exits for empty/None activity_map) ...
    if activity_map is None or activity_map.size == 0:
        print("Warning: Activity map is None or empty in extract_hotspot_from_map.")
        return np.array([]), 0.0 

    map_to_process = activity_map.copy()
    if roi_mask is not None: # Apply ROI if provided
        if roi_mask.shape == activity_map.shape:
            if roi_mask.dtype != bool: roi_mask = roi_mask.astype(bool) # Ensure boolean
            map_to_process[~roi_mask] = np.nan 
        else: print("Warning: ROI mask shape mismatch in extract_hotspot. Ignoring ROI for extraction.")

    if apply_blur: # Blur if requested
        if np.all(np.isnan(map_to_process)): print("Warning: Activity map is all NaN after ROI, skipping blur.")
        else:
            try:
                map_for_blur_no_nan = np.nan_to_num(map_to_process.astype(np.float32), nan=0.0)
                k_h = blur_kernel_size[0] + (1 - blur_kernel_size[0] % 2); k_w = blur_kernel_size[1] + (1 - blur_kernel_size[1] % 2)
                blurred_map = cv2.GaussianBlur(map_for_blur_no_nan, (k_w, k_h), 0)
                if roi_mask is not None and roi_mask.shape == map_to_process.shape: blurred_map[~roi_mask] = np.nan # Restore NaNs
                map_to_process = blurred_map
            except Exception as e: print(f"Warning: GaussianBlur failed: {e}.")

    max_activity_coords = None # Find max activity pixel
    if not np.all(np.isnan(map_to_process)):
        try: max_activity_coords = np.unravel_index(np.nanargmax(map_to_process), map_to_process.shape)
        except ValueError: pass # All NaN
    # print(f"  Debug: Max activity in map_to_process at {max_activity_coords} with value {map_to_process[max_activity_coords] if max_activity_coords else 'N/A':.4e}")


    non_nan_pixels = map_to_process[~np.isnan(map_to_process)] # Thresholding
    if non_nan_pixels.size == 0: print("Warning: No valid pixels for thresholding."); return np.zeros_like(activity_map, dtype=bool), 0.0
    
    map_max_val = np.max(non_nan_pixels)
    if map_max_val < 1e-9 : print("Warning: Max activity near zero."); return np.zeros_like(activity_map, dtype=bool), 0.0
    
    try: threshold_val = np.percentile(non_nan_pixels, threshold_quantile * 100)
    except IndexError: threshold_val = map_max_val
    # print(f"  Threshold value for activity map: {threshold_val:.4e} (Quantile: {threshold_quantile*100}%)")
    if threshold_val <= 1e-9 and map_max_val > 1e-9: threshold_val = min(1e-9, map_max_val * 0.1) # Adjusted floor

    binary_mask = (np.nan_to_num(map_to_process, nan=-np.inf) >= threshold_val).astype(np.uint8)
    if not np.any(binary_mask): print("Warning: No pixels passed activity threshold."); return np.zeros_like(activity_map, dtype=bool), 0.0

    mask_processed = binary_mask.copy() # Morphological operations
    if morphology_op != 'none':
        kernel_size_morph = 3 # Simplified
        kernel_morph = np.ones((kernel_size_morph, kernel_size_morph), np.uint8)
        try:
            if morphology_op == 'close': mask_processed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_morph)
            elif morphology_op == 'open_close':
                opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_morph)
                mask_processed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_morph)
        except cv2.error as e: print(f"Warning: Morphology error: {e}.")
    
    try: num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_processed, connectivity=8) # Connected components
    except cv2.error as e: print(f"Error in CC: {e}."); return np.zeros_like(activity_map, dtype=bool), 0.0

    hotspot_final = np.zeros_like(activity_map, dtype=bool); area = 0.0
    if num_labels > 1:
        if max_activity_coords and labels[max_activity_coords] != 0: # Anchor to max activity
            label_at_max = labels[max_activity_coords]
            hotspot_final = (labels == label_at_max); area = np.sum(hotspot_final)
            # print(f"  Selected component with max activity. Label: {label_at_max}, Area: {area}")
        elif stats.shape[0] > 1 : # Fallback to largest if max_activity_coords not useful
            areas = stats[1:, cv2.CC_STAT_AREA]
            if areas.size > 0:
                largest_idx = np.argmax(areas) + 1 # +1 due to slicing off background
                hotspot_final = (labels == largest_idx); area = np.sum(hotspot_final)
                # print(f"  Fallback: Selected largest component. Label: {largest_idx}, Area: {area}")
    return hotspot_final, area


# Global variable for consistent colorbar in visualize_overlay
frames_raw_for_colorbar_scope = None

def visualize_overlay(base_image_raw, mask, title="Mask Overlay", colormap_name='inferno', save_path=None, 
                      true_leak_coords_vis=None, highlight_color=(0,0,255), highlight_radius=5): # Added true_leak_coords_vis
    global frames_raw_for_colorbar_scope
    if base_image_raw is None: print("Error: Base image is None for visualize_overlay."); return
    
    # Normalize base_image_raw to 0-255 CV_8U for display
    display_img_norm = base_image_raw
    if base_image_raw.dtype != np.uint8:
        min_val, max_val = np.nanmin(base_image_raw), np.nanmax(base_image_raw)
        if max_val > min_val:
            display_img_norm = cv2.normalize(base_image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else: # Handle flat image
            display_img_norm = np.full_like(base_image_raw, 128, dtype=np.uint8) # Mid-gray
    
    base_image_bgr = cv2.cvtColor(display_img_norm, cv2.COLOR_GRAY2BGR)

    # Overlay mask (e.g., yellow)
    if mask is not None and mask.shape == base_image_bgr.shape[:2] and np.any(mask):
        overlay_color_layer = np.zeros_like(base_image_bgr)
        overlay_color_layer[mask.astype(bool)] = [0, 255, 255]  # Yellow (BGR) for mask
        blended_img = cv2.addWeighted(overlay_color_layer, 0.4, base_image_bgr, 0.6, 0)
    else:
        blended_img = base_image_bgr # No mask to overlay or mask invalid

    # --- HIGHLIGHT TRUE LEAK COORDS ---
    if true_leak_coords_vis:
        r_tl_vis, c_tl_vis = true_leak_coords_vis
        # Ensure coords are within image bounds for drawing
        if 0 <= r_tl_vis < blended_img.shape[0] and 0 <= c_tl_vis < blended_img.shape[1]:
            cv2.circle(blended_img, (c_tl_vis, r_tl_vis), radius=highlight_radius, 
                       color=highlight_color, thickness=2) # Red circle (BGR)
    # --- END HIGHLIGHT ---

    img_rgb_to_plot = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * img_rgb_to_plot.shape[0]/img_rgb_to_plot.shape[1] if img_rgb_to_plot.shape[1] > 0 else 10))
    ax.imshow(img_rgb_to_plot)

    # Consistent colorbar using globally set raw frames
    if frames_raw_for_colorbar_scope is not None:
        d_min_cb, d_max_cb = np.nanmin(frames_raw_for_colorbar_scope), np.nanmax(frames_raw_for_colorbar_scope)
        if d_max_cb > d_min_cb:
            cmap_obj_cb = plt.get_cmap(colormap_name); norm_obj_cb = plt.Normalize(vmin=d_min_cb, vmax=d_max_cb)
            sm_cb = plt.cm.ScalarMappable(cmap=cmap_obj_cb, norm=norm_obj_cb); sm_cb.set_array([])
            fig.colorbar(sm_cb, ax=ax, label='Orig Video Temp Scale (°C)', fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path); print(f"  Saved plot to: {save_path}")
    else:
        plt.show() # Show interactively if no save path
    plt.close(fig)


def debug_single_file(mat_file_path, output_plot_dir,
                      mat_key, fps, focus_duration_sec,
                      quantile, morphology_op, apply_blur, blur_kernel_size,
                      envir_para, augment, smooth_window, p_value_threshold,
                      fuselevel, normalizeT,
                      roi_border_percent=None,
                      display_frame_index=0, colormap_name='inferno',
                      true_leak_coords=None): # Keep true_leak_coords
    global frames_raw_for_colorbar_scope # To ensure visualize_overlay uses the correct raw frames

    base_filename = os.path.splitext(os.path.basename(mat_file_path))[0]
    print(f"--- Debugging: {base_filename} ---")
    param_summary = ( # Create a summary of parameters used
        f"FocusDur:{focus_duration_sec}s, SmoothWin:{smooth_window}, PVal:{p_value_threshold:.3f}, " # Use .3f for p_value
        f"Quantile:{quantile:.3f}, Morph:{morphology_op}, ROI:{roi_border_percent*100 if roi_border_percent is not None else 'None'}%, "
        f"Envir:{envir_para}, Blur:{apply_blur}"
    )
    print(param_summary)
    print("-" * 30)

    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        frames_raw_for_colorbar_scope = mat_data[mat_key].astype(np.float64) # Set global for colorbar
        frames_raw = frames_raw_for_colorbar_scope.copy() # Work with a copy
        if frames_raw.ndim != 3 or frames_raw.shape[2] < 2: raise ValueError("Invalid frame data.")
        H, W, num_frames = frames_raw.shape
    except Exception as e: print(f"Error loading data from {mat_file_path}: {e}. Exiting."); return

    frames_proc = apply_preprocessing(frames_raw, normalizeT, fuselevel)
    focus_duration_frames = int(focus_duration_sec * fps)
    focus_duration_frames = max(2, min(focus_duration_frames, num_frames))
    frames_focus = frames_proc[:, :, :focus_duration_frames]

    current_roi_mask = None
    if roi_border_percent is not None and 0.0 <= roi_border_percent < 0.5:
        border_h = int(H * roi_border_percent); border_w = int(W * roi_border_percent)
        if H - 2 * border_h > 0 and W - 2 * border_w > 0: # Ensure ROI is valid
            current_roi_mask = np.zeros((H, W), dtype=bool)
            current_roi_mask[border_h:H-border_h, border_w:W-border_w] = True
        else: print("Warning: ROI border percent too large for image dimensions. Processing full frame.")


    # Time Series Plot for true_leak_coords (if provided)
    if true_leak_coords:
        # ... (Your existing time series plot logic - seems fine) ...
        # Ensure it uses 'frames_raw' for raw data and 'frames_focus' for processed focus data
        print("\n--- Plotting Pixel Time Series for True Leak Coords ---")
        time_vector_full = (np.arange(num_frames) / fps).astype(np.float64)
        time_vector_focus = (np.arange(focus_duration_frames) / fps).astype(np.float64)
        plt.figure(figsize=(14, 10))
        r_tl, c_tl = true_leak_coords[0], true_leak_coords[1] # Unpack
        r_tl = min(max(0, r_tl), H-1); c_tl = min(max(0, c_tl), W-1) # Clamp to bounds

        ax1 = plt.subplot(2,1,1)
        ax1.plot(time_vector_full, frames_raw[r_tl, c_tl, :], label=f'True Leak Loc ({r_tl},{c_tl}) Raw Full')
        ax1.set_xlabel("Time (s) - Full"); ax1.set_ylabel("Raw Temp (°C)"); ax1.set_title(f"Raw Full @ ({r_tl},{c_tl})"); ax1.legend(); ax1.grid(True)

        ax2 = plt.subplot(2,1,2)
        ax2.plot(time_vector_focus, frames_focus[r_tl, c_tl, :], label=f'True Leak ({r_tl},{c_tl}) Focus PostProc') # frames_focus is preprocessed
        if smooth_window > 1 and len(frames_focus[r_tl,c_tl,:]) >= smooth_window:
            smoothed_focus = np.convolve(frames_focus[r_tl,c_tl,:], np.ones(smooth_window)/smooth_window, mode='valid')
            pad_before = (focus_duration_frames - len(smoothed_focus)) // 2
            pad_after = focus_duration_frames - len(smoothed_focus) - pad_before
            if len(smoothed_focus)>0 and pad_before >= 0 and pad_after >=0 :
                 ax2.plot(time_vector_focus, np.pad(smoothed_focus, (pad_before, pad_after), mode='edge'), 
                          label=f'Smoothed (win={smooth_window})', linestyle='--')
        ax2.set_xlabel(f"Time (s) - Focus ({focus_duration_sec}s)"); ax2.set_ylabel("Temp (°C) PostProc"); ax2.set_title(f"Focus @ ({r_tl},{c_tl}) (Smoothed if win>1)"); ax2.legend(); ax2.grid(True)
        
        plt.suptitle(f"Pixel Time Series for {base_filename}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        ts_plot_save_path = os.path.join(output_plot_dir, f"{base_filename}_pixel_timeseries_leak_({r_tl}-{c_tl}).png")
        plt.savefig(ts_plot_save_path); print(f"  Saved time series plot to: {ts_plot_save_path}")
        plt.close()


    print("\nCalculating activity map...")
    # Pass true_leak_coords to the slope calculation function for debugging its specific stats
    activity_map, reported_leak_stats = calculate_filtered_slope_activity_map(
        frames_focus, fps, smooth_window, envir_para, augment, p_value_threshold, 
        roi_mask=current_roi_mask, 
        true_leak_coords_for_debug=true_leak_coords # Pass it here
    )

    # Print stats if true_leak_coords were provided and stats were collected
    if true_leak_coords:
        print(f"\n--- Stats at Specified True Leak Coords ({true_leak_coords[0]},{true_leak_coords[1]}) ---")
        print(f"  Slope: {reported_leak_stats.get('slope', 'N/A'):.6f}")
        print(f"  P-Value: {reported_leak_stats.get('p_value', 'N/A'):.6f}")
        print(f"  Activity Map Value at Coords: {reported_leak_stats.get('activity', 'N/A'):.6f}")
        print(f"  Passed Slope/P-Value Filters? {reported_leak_stats.get('passed_filters', 'N/A')}")
        print("  ------------------------------------------")


    if activity_map is None: # Check if activity_map calculation failed critically
        print("Error: Activity map calculation returned None. Cannot proceed."); return

    # Check if the activity map (within ROI if applicable) is all NaN or zeros
    map_for_check = activity_map[current_roi_mask] if current_roi_mask is not None else activity_map
    if not np.any(np.isfinite(map_for_check)) or np.nanmax(map_for_check) == 0 :
        print("Warning: Activity map is effectively empty (all NaNs or zeros within ROI/Full). "
              "This means no pixels passed the slope filters. Mask will be empty.")
    
    # --- Visualize Activity Map ---
    activity_map_plot_save_path = os.path.join(output_plot_dir, f"{base_filename}_activity_map_plot.png")
    fig_act, ax_act = plt.subplots(figsize=(8,7))
    im_act = ax_act.imshow(activity_map, cmap='hot') # 'hot' is good for activity
    fig_act.colorbar(im_act, ax=ax_act, label="Activity (abs_slope)")
    title_act = f"Activity Map - {base_filename}\n{param_summary}"
    if true_leak_coords: # Add circle to activity map
        ax_act.add_patch(Circle((true_leak_coords[1], true_leak_coords[0]), radius=7, 
                                edgecolor='lime', facecolor='none', lw=1.5, linestyle='--'))
        title_act += f"\nTrueLeak @({true_leak_coords[0]},{true_leak_coords[1]})"
    ax_act.set_title(title_act, fontsize=9)
    plt.savefig(activity_map_plot_save_path); print(f"  Saved activity map plot to: {activity_map_plot_save_path}")
    plt.close(fig_act)
    
    print("\nExtracting hotspot mask...")
    hotspot_mask, hotspot_area = extract_hotspot_from_map(
        activity_map, quantile, morphology_op, apply_blur, tuple(blur_kernel_size), 
        roi_mask=current_roi_mask # Pass ROI here too for consistency in processing
    )

    if hotspot_mask.size == 0: print("Error: Failed mask extraction (empty array returned)."); return # Check for empty array
    print(f"Generated mask area: {hotspot_area:.0f} pixels")

    # --- Visualize Final Mask Overlay ---
    mask_plot_save_path = os.path.join(output_plot_dir, f"{base_filename}_final_mask_overlay.png")
    # Ensure display_frame_index is valid
    valid_display_frame_idx = min(max(0, display_frame_index), num_frames-1)
    frame_to_display_raw = frames_raw[:, :, valid_display_frame_idx]
    
    viz_title_overlay = (f"{base_filename} | Frame {valid_display_frame_idx} | Mask Area: {hotspot_area:.0f}px\n"
                         f"{param_summary}")
    if true_leak_coords: viz_title_overlay += f"\nTrueLeak @({true_leak_coords[0]},{true_leak_coords[1]})"
    
    visualize_overlay(frame_to_display_raw, hotspot_mask, title=viz_title_overlay, 
                      colormap_name=colormap_name, save_path=mask_plot_save_path,
                      true_leak_coords_vis=true_leak_coords, highlight_radius=7, highlight_color=(0,0,255)) # BGR red for CV2
    
    # Save the generated mask as .npy
    try:
        mask_filename_npy = f"{base_filename}_DEBUG_mask.npy"
        mask_save_path_npy = os.path.join(output_plot_dir, mask_filename_npy)
        np.save(mask_save_path_npy, hotspot_mask)
        print(f"  Saved debug mask .npy to: {mask_save_path_npy}")
    except Exception as e:
        print(f"  Error saving debug mask .npy: {e}")

    print("-" * 30 + f"\nDebug run for {base_filename} finished." + "-"*30 )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug hotspot mask generation for a SINGLE .mat file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Positional Arguments
    parser.add_argument("mat_file", help="Path to the specific .mat file.")
    parser.add_argument("output_plot_dir", help="Directory to save plots and debug mask.")

    # Optional Arguments (with defaults from config where applicable)
    # IO and Data
    parser.add_argument("-k", "--key", default=getattr(config, 'MAT_FRAMES_KEY', 'TempFrames'), help="Key in .mat file for frame data.")
    parser.add_argument("--fps", type=float, default=getattr(config, 'MASK_FPS', 5.0), help="Frames per second of original video.")
    
    # Slope Calculation Parameters
    parser.add_argument("--focus_duration_sec", type=float, default=getattr(config, 'MASK_FOCUS_DURATION_SEC', 5.0), help="Duration (s) for slope analysis.")
    parser.add_argument("--smooth_window", type=int, default=getattr(config, 'MASK_SMOOTH_WINDOW', 1), help="Temporal smoothing window (1=None).")
    parser.add_argument("--p_value_thresh", type=float, default=getattr(config, 'MASK_P_VALUE_THRESHOLD', 0.10), help="P-value for slope significance.")
    parser.add_argument("--envir_para", type=int, default=getattr(config, 'MASK_ENVIR_PARA', 1), choices=[-1, 1], help="Environment (-1=Cooling expected, 1=Heating expected).")
    parser.add_argument("--augment", type=float, default=getattr(config, 'MASK_AUGMENT_SLOPE', 1.0), help="Exponent for slope magnitude.")

    # Frame Preprocessing
    parser.add_argument("--normalizeT", type=lambda x: (str(x).lower() == 'true'), default=getattr(config, 'MASK_NORMALIZE_TEMP_FRAMES', False), help="Normalize T per frame? (True/False)")
    parser.add_argument("--fuselevel", type=int, default=getattr(config, 'MASK_FUSE_LEVEL', 0), help="Spatial fuse level (0=None).")
    
    # ROI
    parser.add_argument("--roi_border_percent", type=float, default=getattr(config, 'MASK_ROI_BORDER_PERCENT', 0.0), help="ROI border (0.0 to <0.5). 0.0 means no ROI.")
    
    # Hotspot Extraction
    parser.add_argument("-q", "--quantile", type=float, default=getattr(config, 'MASK_ACTIVITY_QUANTILE', 0.99), help="Quantile threshold for activity map.")
    parser.add_argument("--morph_op", default=getattr(config, 'MASK_MORPHOLOGY_OP', 'none'), choices=['close', 'open_close', 'none'], help="Morphology op.")
    parser.add_argument("--blur", type=lambda x: (str(x).lower() == 'true'), default=getattr(config, 'MASK_APPLY_BLUR_TO_ACTIVITY_MAP', False), help="Apply Gaussian blur to activity map? (True/False)")
    parser.add_argument("--blur_kernel", type=int, nargs=2, default=getattr(config, 'MASK_BLUR_KERNEL_SIZE', (3,3)), metavar=('H', 'W'), help="Blur kernel (odd nums H W).")
    
    # Visualization & Debug
    parser.add_argument("--display_frame", type=int, default=0, help="Frame index for final mask overlay visualization.")
    parser.add_argument("--colormap", default='inferno', choices=plt.colormaps(), help="Colormap for thermal visualizations.")
    parser.add_argument("--true_leak_coords", type=int, nargs=2, default=None, metavar=('ROW', 'COL'), 
                        help="Optional: (ROW, COL) of known/true leak. If provided, time series for this pixel will be plotted, "
                             "and its slope/p-value/activity will be printed, and it will be highlighted on plots.")


    if len(sys.argv) < 3: parser.print_help(); sys.exit(1) # mat_file and output_plot_dir are positional
    args = parser.parse_args()

    # --- Argument Validations ---
    if not os.path.isfile(args.mat_file): print(f"Error: .mat file not found: {args.mat_file}"); sys.exit(1)
    if args.focus_duration_sec <= 0 or args.fps <= 0 : print("Error: focus_duration_sec and fps must be positive."); sys.exit(1)
    if not (0.0 <= args.roi_border_percent < 0.5): print("Error: ROI border percent must be between 0.0 and 0.49 (inclusive of 0.0 for no ROI)."); sys.exit(1)
    if args.blur and not (args.blur_kernel[0] > 0 and args.blur_kernel[1] > 0 and args.blur_kernel[0] % 2 != 0 and args.blur_kernel[1] % 2 != 0):
        print("Error: Blur kernel dimensions must be positive and odd."); sys.exit(1)
    if not (0.0 < args.p_value_thresh < 1.0): print("Error: P-value threshold must be >0 and <1."); sys.exit(1)
    if not (0.0 < args.quantile <= 1.0): print("Error: Quantile must be >0 and <=1."); sys.exit(1)


    os.makedirs(args.output_plot_dir, exist_ok=True)

    debug_single_file(
        mat_file_path=args.mat_file,
        output_plot_dir=args.output_plot_dir,
        mat_key=args.key, fps=args.fps,
        focus_duration_sec=args.focus_duration_sec,
        quantile=args.quantile, morphology_op=args.morph_op,
        apply_blur=args.blur, blur_kernel_size=tuple(args.blur_kernel), # Ensure tuple
        envir_para=args.envir_para, augment=args.augment,
        smooth_window=args.smooth_window,
        p_value_threshold=args.p_value_thresh,
        fuselevel=args.fuselevel, normalizeT=args.normalizeT, # Already bool from lambda
        roi_border_percent=args.roi_border_percent,
        display_frame_index=args.display_frame,
        colormap_name=args.colormap,
        true_leak_coords=args.true_leak_coords # Pass it through
    )