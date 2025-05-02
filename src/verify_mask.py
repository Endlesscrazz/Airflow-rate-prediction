"""
Script to visualize pre-generated hotspot masks overlaid on their corresponding
raw thermal images. Helps verify the output of hotspot_mask_generation.py.

Loads a .mat file, finds its corresponding .npy mask file, and displays
(or saves) an overlay image.
"""

import numpy as np
import cv2
import os
import scipy.io
import argparse
import sys
import traceback
import matplotlib.pyplot as plt # For displaying the image

# --- Configuration ---
# Default key for frame data in .mat files (can be overridden)
DEFAULT_MAT_KEY = "TempFrames"
# Default colormap for displaying the thermal image
# Use standard matplotlib colormaps: https://matplotlib.org/stable/users/explain/colors/colormaps.html
DEFAULT_COLORMAP = 'inferno' # 'hot', 'inferno', 'gray', 'viridis' etc.
# Default overlay color (BGR format for OpenCV)
DEFAULT_OVERLAY_COLOR = [0, 255, 255] # Yellow
# Default overlay transparency
DEFAULT_ALPHA = 0.4

# --- Helper Functions ---

def load_frames(mat_file_path, mat_key):
    """Loads frames from a .mat file."""
    if not os.path.exists(mat_file_path):
        print(f"Error: .mat file not found: {mat_file_path}")
        return None
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        if mat_key not in mat_data:
            print(f"Error: Key '{mat_key}' not found in {mat_file_path}. Available keys: {list(mat_data.keys())}")
            return None
        frames = mat_data[mat_key].astype(np.float64)
        if frames.ndim != 3 or frames.shape[2] < 1:
            print(f"Error: Invalid frame data shape {frames.shape} in {mat_file_path}")
            return None
        return frames
    except Exception as e:
        print(f"Error loading .mat file {mat_file_path}: {e}")
        traceback.print_exc()
        return None

def load_mask(mask_file_path):
    """Loads a boolean mask from a .npy file."""
    if not os.path.exists(mask_file_path):
        print(f"Error: Mask file not found: {mask_file_path}")
        return None
    try:
        mask = np.load(mask_file_path)
        if mask.dtype != bool:
            print(f"Warning: Mask dtype is {mask.dtype}, converting to bool.")
            mask = mask.astype(bool)
        if mask.ndim != 2:
            print(f"Error: Invalid mask dimension {mask.ndim}, expected 2.")
            return None
        return mask
    except Exception as e:
        print(f"Error loading mask file {mask_file_path}: {e}")
        traceback.print_exc()
        return None

def create_overlay(base_image_gray, mask, color_bgr, alpha):
    """Creates a blended overlay image."""
    if base_image_gray is None or mask is None:
        return None
    try:
        if base_image_gray.shape != mask.shape:
            print("Error: Base image shape and mask shape do not match.")
            return None

        # Convert grayscale base image to BGR for color overlay
        base_image_bgr = cv2.cvtColor(base_image_gray, cv2.COLOR_GRAY2BGR)

        # Create colored overlay where mask is True
        overlay_color_layer = np.zeros_like(base_image_bgr)
        overlay_color_layer[mask] = color_bgr

        # Blend
        blended_img = cv2.addWeighted(overlay_color_layer, alpha, base_image_bgr, 1 - alpha, 0)
        return blended_img
    except Exception as e:
        print(f"Error creating overlay: {e}")
        traceback.print_exc()
        return None

# --- Main Function ---
def visualize_mask(mat_file_path, mask_dir, mat_key, display_frame_index=0,
                     colormap_name=DEFAULT_COLORMAP, overlay_color=DEFAULT_OVERLAY_COLOR,
                     alpha=DEFAULT_ALPHA, save_path=None):
    """Loads data, finds mask, creates overlay, and displays or saves."""

    print(f"Visualizing mask for: {mat_file_path}")
    print(f"Looking for mask in base directory: {mask_dir}")

    # --- Construct Mask Path ---
    try:
        mat_filename = os.path.basename(mat_file_path)
        mat_filename_no_ext = os.path.splitext(mat_filename)[0]
        # Try to determine relative path from a common ancestor if possible,
        # otherwise assume mask structure mirrors .mat relative path from mask_dir root.
        # This part might need adjustment depending on your exact structure.
        # Simplest approach: assume mask is in mask_dir/[same_relative_path]/[mask_name]
        # We need the relative path of the mat file *within its dataset folder*
        # This assumes mask_dir mirrors the dataset structure.
        dataset_dir = os.path.dirname(os.path.dirname(mat_file_path)) # Go up two levels (e.g., from dataset/FanPower/file.mat to dataset) - ADJUST IF NEEDED
        relative_dir = os.path.relpath(os.path.dirname(mat_file_path), dataset_dir)

        mask_filename = mat_filename_no_ext + '_mask.npy'
        expected_mask_path = os.path.join(mask_dir, relative_dir, mask_filename)
        print(f"Expected mask path: {expected_mask_path}")

    except Exception as e:
        print(f"Error determining mask path: {e}")
        return

    # --- Load Frames and Mask ---
    frames = load_frames(mat_file_path, mat_key)
    mask = load_mask(expected_mask_path)

    if frames is None or mask is None:
        print("Failed to load frames or mask. Cannot visualize.")
        return

    # Validate frame index
    num_frames = frames.shape[2]
    if not 0 <= display_frame_index < num_frames:
        print(f"Warning: Frame index {display_frame_index} out of bounds (0-{num_frames-1}). Using frame 0.")
        display_frame_index = 0

    # Select frame and ensure correct shape
    display_frame = frames[:, :, display_frame_index]
    if display_frame.shape != mask.shape:
         print(f"Error: Frame shape {display_frame.shape} and mask shape {mask.shape} mismatch.")
         return

    # --- Prepare Base Image for Overlay ---
    # Normalize the single frame to 0-255 uint8 for visualization
    try:
        base_image_normalized = cv2.normalize(display_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    except Exception as e:
        print(f"Error normalizing display frame: {e}")
        return

    # --- Create Overlay ---
    overlay_image = create_overlay(base_image_normalized, mask, overlay_color, alpha)

    if overlay_image is None:
        print("Failed to create overlay image.")
        return

    # --- Display or Save ---
    title = f"Mask Overlay ({os.path.basename(expected_mask_path)})\non Frame {display_frame_index} of {os.path.basename(mat_file_path)}"

    if save_path:
        try:
            # Use a colormap for the saved image if desired (more informative than raw gray)
            # Apply colormap to the overlay image which is already BGR
            # To apply matplotlib colormap correctly, convert overlay BGR to RGB first
            img_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)

            # Create figure and axes for proper colormap application
            fig, ax = plt.subplots(1, 1, figsize=(8, 8 * img_rgb.shape[0] / img_rgb.shape[1])) # Maintain aspect ratio
            im = ax.imshow(img_rgb) # Display the RGB overlay directly

            # Get the raw frame data for the colorbar limits
            d_min, d_max = np.nanmin(display_frame), np.nanmax(display_frame)
            if d_max > d_min:
                # Apply colormap to the raw data to get colors for colorbar
                 cmap = plt.get_cmap(colormap_name)
                 norm = plt.Normalize(vmin=d_min, vmax=d_max)
                 sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                 sm.set_array([]) # Needed for colorbar
                 fig.colorbar(sm, ax=ax, label='Original Temperature Scale (Approx)')

            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close(fig) # Close the figure to free memory
            print(f"Saved overlay image with colormap to: {save_path}")
        except Exception as e:
            print(f"Error saving image: {e}")
            traceback.print_exc()
            # Fallback: save the BGR overlay directly if matplotlib fails
            try: cv2.imwrite(save_path, overlay_image); print(f"Saved raw BGR overlay to: {save_path}")
            except: print("Fallback save failed.")

    else:
        # Display using matplotlib for better window handling and colormaps
        try:
            # Similar to saving, convert BGR overlay to RGB for display
            img_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(img_rgb) # Display the RGB overlay

            # Add colorbar based on raw frame data range
            d_min, d_max = np.nanmin(display_frame), np.nanmax(display_frame)
            if d_max > d_min:
                 cmap = plt.get_cmap(colormap_name)
                 norm = plt.Normalize(vmin=d_min, vmax=d_max)
                 sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                 sm.set_array([])
                 fig.colorbar(sm, ax=ax, label='Original Temperature Scale (Approx)')

            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.show() # Display the plot window
        except ImportError:
             print("Matplotlib not found. Displaying with OpenCV (no colormap/colorbar).")
             # Fallback display using OpenCV
             cv2.imshow(title, overlay_image)
             cv2.waitKey(0)
             cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error displaying image: {e}")
            traceback.print_exc()


# --- Command-Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a pre-generated hotspot mask overlaid on a raw thermal image frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("mat_file", help="Path to the specific input .mat file to visualize.")
    parser.add_argument("mask_dir", help="Path to the BASE directory containing the pre-generated mask files (e.g., 'output_hotspot_mask').")
    parser.add_argument("-k", "--key", default=DEFAULT_MAT_KEY, help="Key in the .mat file for frame data.")
    parser.add_argument("-f", "--frame_index", type=int, default=0, help="Index of the frame from the .mat file to display (0-based).")
    parser.add_argument("-c", "--colormap", default=DEFAULT_COLORMAP, choices=plt.colormaps(), help="Matplotlib colormap for displaying the base image.")
    parser.add_argument("-s", "--save", metavar="SAVE_PATH", default=None, help="Optional: Path to save the output image instead of displaying it.")

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Basic validation
    if not os.path.isfile(args.mat_file):
        print(f"Error: Input .mat file not found: {args.mat_file}")
        sys.exit(1)
    if not os.path.isdir(args.mask_dir):
        print(f"Error: Mask directory not found: {args.mask_dir}")
        sys.exit(1)
    if args.frame_index < 0:
        print("Warning: Frame index cannot be negative. Using frame 0.")
        args.frame_index = 0

    # Run visualization
    visualize_mask(
        mat_file_path=args.mat_file,
        mask_dir=args.mask_dir,
        mat_key=args.key,
        display_frame_index=args.frame_index,
        colormap_name=args.colormap,
        save_path=args.save
        # Using default overlay color and alpha for now
    )