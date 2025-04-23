# visualization_utils.py
"""
Utility functions for saving visualizations related to thermal image processing,
focusing on dynamic hotspot identification via gradient maps.
"""
import os
import cv2
import numpy as np

def save_dynamic_hotspot_visualizations(
    original_frame,
    mean_frame,          # For comparison
    interval_gradient_maps, # List of interval gradient maps
    combined_gradient_map, # The combined map used for hotspot detection
    hotspot_mask,        # The final mask derived from combined_gradient_map
    index,
    save_dir,
    orig_colormap=cv2.COLORMAP_HOT,
    grad_colormap=cv2.COLORMAP_INFERNO
):
    """
    Saves diagnostic visualization images for dynamic hotspot identification.

    Args:
        original_frame (np.ndarray): First frame for context.
        mean_frame (np.ndarray): Mean temperature frame.
        interval_gradient_maps (list): List of np.ndarray interval gradient maps.
        combined_gradient_map (np.ndarray): Combined gradient map used for detection.
        hotspot_mask (np.ndarray): Final boolean hotspot mask.
        index (int): Sample index for filenames.
        save_dir (str): Specific directory to save images.
        orig_colormap (int): OpenCV colormap enum for original/mean frames.
        grad_colormap (int): OpenCV colormap enum for gradient maps.
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created visualization subdirectory: {save_dir}")

        num_saved = 0

        # --- Helper for normalizing and saving ---
        def _normalize_save(img_data, filename_base, description, colormap=None):
            nonlocal num_saved
            if img_data is None:
                print(f"  Skipping save: {description} data is None.")
                return
            try:
                d_min, d_max = np.min(img_data), np.max(img_data)
                if d_max > d_min: # Avoid normalizing flat images
                    norm_img = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                elif d_max == d_min: # Flat image
                    norm_img = np.full(img_data.shape[:2], int(np.clip(d_min, 0, 255)), dtype=cv2.CV_8U) # Use constant value if possible
                else: # Should not happen if not None, but safety
                     norm_img = np.zeros(img_data.shape[:2], dtype=cv2.CV_8U)

                path_gray = os.path.join(save_dir, f"{filename_base}_gray.png")
                cv2.imwrite(path_gray, norm_img)
                num_saved += 1

                if colormap is not None:
                    try:
                        color_img = cv2.applyColorMap(norm_img, colormap)
                        path_color = os.path.join(save_dir, f"{filename_base}_color.png")
                        cv2.imwrite(path_color, color_img)
                        num_saved += 1
                    except Exception as cmap_e:
                        print(f"  Warning: Failed applying colormap for {description}: {cmap_e}")
            except Exception as e:
                print(f"  Error processing/saving {description}: {e}")

        # --- Save individual images ---
        base_fn = f"sample_{index}"
        # 1. Original Frame
        _normalize_save(original_frame, f"{base_fn}_1_original", "Original Frame", orig_colormap)
        # 2. Mean Frame
        _normalize_save(mean_frame, f"{base_fn}_2_mean_frame", "Mean Frame", orig_colormap)
        # 3. Interval Gradient Maps
        if interval_gradient_maps:
            for i, grad_map in enumerate(interval_gradient_maps):
                _normalize_save(grad_map, f"{base_fn}_3_interval_{i}_grad", f"Interval {i} Gradient", grad_colormap)
        # 4. Combined Gradient Map
        _normalize_save(combined_gradient_map, f"{base_fn}_4_combined_grad", "Combined Gradient", grad_colormap)
        # 5. Hotspot Mask
        if hotspot_mask is not None:
            try:
                mask_img = (hotspot_mask.astype(np.uint8) * 255)
                mask_path = os.path.join(save_dir, f"{base_fn}_5_hotspot_mask.png")
                cv2.imwrite(mask_path, mask_img)
                num_saved += 1
            except Exception as e: print(f"  Error saving hotspot mask: {e}")
        # 6. Hotspot Overlay
        if hotspot_mask is not None and original_frame is not None:
            try:
                vis_orig_norm = cv2.normalize(original_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                vis_orig_bgr = cv2.cvtColor(vis_orig_norm, cv2.COLOR_GRAY2BGR)
                overlay_img = vis_orig_bgr.copy()
                overlay_color = [0, 255, 255] # Yellow
                overlay_img[hotspot_mask] = overlay_color
                alpha = 0.4
                blended_img = cv2.addWeighted(overlay_img, alpha, vis_orig_bgr, 1 - alpha, 0)
                overlay_path = os.path.join(save_dir, f"{base_fn}_6_hotspot_overlay.png")
                cv2.imwrite(overlay_path, blended_img)
                num_saved += 1
            except Exception as e: print(f"  Error saving overlay: {e}")

        print(f"Saved {num_saved} visualization images for sample {index} to {save_dir}")

    except Exception as e:
        print(f"!!! Error during visualization saving for sample {index}: {e}")
        import traceback
        traceback.print_exc()