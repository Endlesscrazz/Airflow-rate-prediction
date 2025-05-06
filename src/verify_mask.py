# verify_masks.py
"""
Script to visualize pre-generated hotspot masks overlaid on their corresponding
raw thermal images FOR ALL SAMPLES IN A DATASET.

Iterates through the dataset directory, finds corresponding .mat and .npy mask
files, and saves overlay images to a structured output directory.
"""

import numpy as np
import cv2
import os
import scipy.io
import argparse
import sys
import traceback
import matplotlib.pyplot as plt
import fnmatch # For finding .mat files

# --- Import data utils and config ---
try:
    import data_utils # To use parsing functions if needed, or just for directory structure
    import config     # To get dataset root and MAT key
except ImportError:
    print("Error: Failed to import data_utils.py or config.py.")
    sys.exit(1)

# --- Configuration ---
# Get MAT key from config
DEFAULT_MAT_KEY = config.MAT_FRAMES_KEY
# Default colormap for displaying the thermal image
DEFAULT_COLORMAP = 'inferno' # 'hot', 'inferno', 'gray', 'viridis' etc.
# Default overlay color (BGR format for OpenCV)
DEFAULT_OVERLAY_COLOR = [0, 255, 255] # Yellow
# Default overlay transparency
DEFAULT_ALPHA = 0.4
# Frame index to visualize for each video
DEFAULT_FRAME_INDEX = 0

# --- Helper Functions (Keep load_frames, load_mask, create_overlay as before) ---

def load_frames(mat_file_path, mat_key):
    """Loads frames from a .mat file."""
    if not os.path.exists(mat_file_path): return None
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        if mat_key not in mat_data: return None
        frames = mat_data[mat_key].astype(np.float64)
        if frames.ndim != 3 or frames.shape[2] < 1: return None
        return frames
    except Exception as e: print(f"Error loading .mat {os.path.basename(mat_file_path)}: {e}"); return None

def load_mask(mask_file_path):
    """Loads a boolean mask from a .npy file."""
    if not os.path.exists(mask_file_path): return None
    try:
        mask = np.load(mask_file_path)
        if mask.dtype != bool: mask = mask.astype(bool)
        if mask.ndim != 2: return None
        return mask
    except Exception as e: print(f"Error loading mask {os.path.basename(mask_file_path)}: {e}"); return None

def create_overlay(base_image_gray, mask, color_bgr, alpha):
    """Creates a blended overlay image."""
    if base_image_gray is None or mask is None or base_image_gray.shape != mask.shape: return None
    try:
        base_image_bgr = cv2.cvtColor(base_image_gray, cv2.COLOR_GRAY2BGR)
        overlay_color_layer = np.zeros_like(base_image_bgr)
        overlay_color_layer[mask] = color_bgr
        blended_img = cv2.addWeighted(overlay_color_layer, alpha, base_image_bgr, 1 - alpha, 0)
        return blended_img
    except Exception as e: print(f"Error creating overlay: {e}"); return None

# --- Main Processing Function for the Dataset ---
def verify_all_masks(dataset_root_dir, mask_base_dir, output_base_dir, mat_key,
                       display_frame_index=DEFAULT_FRAME_INDEX,
                       colormap_name=DEFAULT_COLORMAP,
                       overlay_color=DEFAULT_OVERLAY_COLOR,
                       alpha=DEFAULT_ALPHA):
    """
    Iterates through dataset, loads mat/mask pairs, saves overlay visualization.
    """
    print(f"Starting mask verification process...")
    print(f"Dataset Root: {dataset_root_dir}")
    print(f"Input Mask Base Directory: {mask_base_dir}")
    print(f"Output Visualization Directory: {output_base_dir}")
    print(f"Visualizing Frame Index: {display_frame_index}")
    print("-" * 30)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Iterate through the dataset directory structure
    for root, _, files in os.walk(dataset_root_dir):
        # Determine relative path for constructing output/mask paths
        try:
            relative_dir = os.path.relpath(root, dataset_root_dir)
        except ValueError: # Handle case where root might not be under dataset_root_dir (e.g., '.' )
            if root == dataset_root_dir: relative_dir = '.'
            else: continue # Skip if path is unexpected

        # Skip the root directory itself if necessary, depending on structure
        # if relative_dir == '.': continue

        print(f"Scanning directory: {relative_dir}")
        found_in_dir = 0

        for mat_filename in fnmatch.filter(files, '*.mat'):
            mat_file_path = os.path.join(root, mat_filename)
            mat_filename_no_ext = os.path.splitext(mat_filename)[0]

            # --- Construct Expected Mask Path ---
            mask_filename = mat_filename_no_ext + '_mask.npy'
            expected_mask_path = os.path.join(mask_base_dir, relative_dir, mask_filename)
            # Add fallback check if masks are stored differently (e.g., in subfolder)
            if not os.path.exists(expected_mask_path):
                 mask_path_alt = os.path.join(mask_base_dir, relative_dir, mat_filename_no_ext, mask_filename)
                 if os.path.exists(mask_path_alt): expected_mask_path = mask_path_alt
                 else:
                     # print(f"  Skipping {mat_filename}: Mask not found at primary or alternate path.")
                     skipped_count += 1
                     continue # Skip if mask not found

            # --- Load Frames and Mask ---
            frames = load_frames(mat_file_path, mat_key)
            mask = load_mask(expected_mask_path)

            if frames is None or mask is None:
                print(f"  Skipping {mat_filename}: Failed to load frames or mask.")
                error_count += 1
                continue

            # Validate frame index and shapes
            num_frames = frames.shape[2]
            current_frame_index = min(display_frame_index, num_frames - 1) # Use valid index
            display_frame = frames[:, :, current_frame_index]
            if display_frame.shape != mask.shape:
                 print(f"  Skipping {mat_filename}: Frame/mask shape mismatch ({display_frame.shape} vs {mask.shape}).")
                 error_count += 1
                 continue

            # --- Prepare Base Image for Overlay ---
            try:
                base_image_normalized = cv2.normalize(display_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            except Exception as e:
                print(f"  Skipping {mat_filename}: Error normalizing frame: {e}")
                error_count += 1
                continue

            # --- Create Overlay ---
            overlay_image = create_overlay(base_image_normalized, mask, overlay_color, alpha)
            if overlay_image is None:
                print(f"  Skipping {mat_filename}: Failed to create overlay.")
                error_count += 1
                continue

            # --- Save Overlay Image ---
            try:
                # Construct output path, mirroring input structure
                vis_filename = mat_filename_no_ext + f'_frame{current_frame_index}_overlay.png'
                output_subdir = os.path.join(output_base_dir, relative_dir)
                os.makedirs(output_subdir, exist_ok=True)
                save_path = os.path.join(output_subdir, vis_filename)

                # Save using matplotlib for colormap/colorbar
                img_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
                fig, ax = plt.subplots(1, 1, figsize=(8, 8 * img_rgb.shape[0]/img_rgb.shape[1]))
                im = ax.imshow(img_rgb)

                d_min, d_max = np.nanmin(display_frame), np.nanmax(display_frame)
                if d_max > d_min:
                     cmap = plt.get_cmap(colormap_name); norm = plt.Normalize(vmin=d_min, vmax=d_max)
                     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
                     fig.colorbar(sm, ax=ax, label='Orig Temp Scale (Approx)')

                ax.set_title(f"{relative_dir}/{mat_filename}\nFrame {current_frame_index} w/ Mask Overlay", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                plt.tight_layout()
                plt.savefig(save_path, dpi=100) # Lower dpi for faster saving of many files
                plt.close(fig)
                processed_count += 1
                found_in_dir +=1

            except Exception as e:
                print(f"  Error saving image for {mat_filename}: {e}")
                # traceback.print_exc() # Uncomment for full traceback
                error_count += 1

        if found_in_dir > 0: print(f"  Processed {found_in_dir} files in this directory.")


    print("-" * 30)
    print(f"Mask verification finished.")
    print(f"Successfully generated visualizations for: {processed_count} files.")
    print(f"Skipped (missing mask/data): {skipped_count} files.")
    print(f"Errors during processing/saving: {error_count} files.")

# --- Command-Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate overlay visualizations for all pre-computed masks in a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Removed mat_file argument
    parser.add_argument("mask_dir", help="Path to the BASE directory containing the pre-generated mask files (e.g., 'output_hotspot_mask').")
    parser.add_argument("output_dir", help="Path to the BASE directory where overlay images will be saved (e.g., 'verified_masks').")
    parser.add_argument("-k", "--key", default=DEFAULT_MAT_KEY, help="Key in the .mat file for frame data.")
    parser.add_argument("-f", "--frame_index", type=int, default=DEFAULT_FRAME_INDEX, help="Index of the frame from each .mat file to display (0-based).")
    parser.add_argument("-c", "--colormap", default=DEFAULT_COLORMAP, choices=plt.colormaps(), help="Matplotlib colormap for displaying the base image.")
    # Removed --save argument, saving is default now

    # Check if enough arguments provided (Now needs 2 positional args)
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # --- Get Input Dataset Directory from Config ---
    input_dataset_dir = config.DATASET_FOLDER
    if not input_dataset_dir or not os.path.isdir(input_dataset_dir):
         print(f"Error: Dataset folder specified in config.py not found: {input_dataset_dir}")
         sys.exit(1)

    # Basic validation
    if not os.path.isdir(args.mask_dir):
        print(f"Error: Mask directory not found: {args.mask_dir}")
        sys.exit(1)
    if args.frame_index < 0:
        print("Warning: Frame index cannot be negative. Using frame 0.")
        args.frame_index = 0

    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run verification for the whole dataset
    verify_all_masks(
        dataset_root_dir=input_dataset_dir,
        mask_base_dir=args.mask_dir,
        output_base_dir=args.output_dir,
        mat_key=args.key,
        display_frame_index=args.frame_index,
        colormap_name=args.colormap,
        # Using default overlay color and alpha
    )

    print("Verification script finished.")