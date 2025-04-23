# extract_thermal_images.py

import scipy.io
import numpy as np
import cv2
import os
import argparse
import sys

def extract_and_save_frames(mat_file_path, base_output_dir, colormap_name='jet', img_format='png'):
    """
    Loads thermal video data from a .mat file, creates a structured output
    directory, normalizes each frame, applies a chosen colormap, and saves
    them as individual image files.

    Args:
        mat_file_path (str): Path to the input .mat file.
        base_output_dir (str): Directory where the 'thermal_images' folder
                               and subsequent subfolders will be created.
        colormap_name (str): Name of the OpenCV colormap to apply
                             (e.g., 'jet', 'hot', 'inferno', 'gray').
                             'gray' preserves the original grayscale.
        img_format (str): The desired output image format ('png', 'jpg', 'tif').
                          Defaults to 'png'.
    """
    print(f"--- Starting Frame Extraction with Pseudocoloring ---")
    print(f"Input MAT file: {mat_file_path}")
    print(f"Base output directory: {base_output_dir}")

    mat_key = 'TempFrames' # Hardcoded key

    # --- Map colormap names to OpenCV constants ---
    all_maps = {
    'jet':       'COLORMAP_JET',
    'hot':       'COLORMAP_HOT',
    'inferno':   'COLORMAP_INFERNO',
    # …
    'coolwarm':  'COLORMAP_COOLWARM',
    'gray':      None  # special case
    }

    colormap_map = {}
    for name, attr in all_maps.items():
        if name == 'gray':
            colormap_map[name] = -1
        elif hasattr(cv2, attr):
            colormap_map[name] = getattr(cv2, attr)

    if colormap_name.lower() not in colormap_map:
        print(f"Error: Invalid colormap name '{colormap_name}'.")
        print(f"Available choices: {list(colormap_map.keys())}")
        sys.exit(1)

    selected_colormap = colormap_map[colormap_name.lower()]
    print(f"Applying Colormap: {colormap_name.upper()}")

    # --- 1. Derive and Create Output Directory Structure ---
    try:
        base_name = os.path.basename(mat_file_path)
        subfolder_name, _ = os.path.splitext(base_name)
        if not subfolder_name: raise ValueError("Invalid subfolder name derived.")

        main_thermal_dir = os.path.join(base_output_dir, "thermal_images")
        # Add colormap name to the final directory for clarity
        final_output_dir = os.path.join(main_thermal_dir, subfolder_name + f"_{colormap_name.lower()}")

        os.makedirs(final_output_dir, exist_ok=True)
        print(f"Output directory ensured: {final_output_dir}")

    except ValueError as e:
        print(f"Error deriving output path: {e}")
        sys.exit(1)
    except OSError as e:
        print(f"Error creating output directory '{final_output_dir}': {e}")
        sys.exit(1)

    # --- 2. Load the .mat File ---
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        print("MAT file loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Input MAT file not found at '{mat_file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading MAT file '{mat_file_path}': {e}")
        sys.exit(1)

    # --- 3. Access Frame Data ---
    if mat_key not in mat_data:
        print(f"Error: Key '{mat_key}' not found. Available keys: {list(mat_data.keys())}")
        sys.exit(1)
    try:
        frames_data = mat_data[mat_key]
        if not isinstance(frames_data, np.ndarray): raise TypeError("Data is not NumPy array.")
        print(f"Successfully accessed data with key '{mat_key}'.")
    except Exception as e:
         print(f"Error accessing data: {e}"); sys.exit(1)

    # --- 4. Validate Data Shape ---
    if frames_data.ndim != 3:
        print(f"Error: Expected 3D array, got {frames_data.ndim}D shape {frames_data.shape}")
        sys.exit(1)
    height, width, num_frames = frames_data.shape
    print(f"Data shape: Height={height}, Width={width}, Frames={num_frames}")
    if num_frames == 0: print("Error: No frames found."); sys.exit(1)

    # --- 5. Normalize, Apply Colormap, and Save Frames ---
    print("Calculating global min/max temperature for normalization...")
    min_temp = np.nanmin(frames_data); max_temp = np.nanmax(frames_data)
    if np.isnan(min_temp) or np.isnan(max_temp): print("Warning: NaNs detected.")
    if min_temp == max_temp: print("Warning: Constant temperature data.")
    print(f"Global Temperature Range: Min={min_temp:.2f}, Max={max_temp:.2f}")
    num_digits = len(str(num_frames - 1))
    print(f"Saving {num_frames} frames as .{img_format}...")
    saved_count = 0

    for i in range(num_frames):
        frame = frames_data[:, :, i].astype(np.float64)

        # --- Normalize to 0-255 Grayscale ---
        if min_temp == max_temp:
            normalized_frame_gray = np.full((height, width), 128, dtype=np.uint8)
        else:
            try:
                normalized_frame_gray = cv2.normalize(frame, None, alpha=0, beta=255,
                                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            except Exception as e:
                print(f"Error normalizing frame {i}: {e}. Skipping frame.")
                continue

        # --- Apply Colormap (if not 'gray') ---
        if selected_colormap != -1: # -1 is our flag for 'gray'
             try:
                 output_frame = cv2.applyColorMap(normalized_frame_gray, selected_colormap)
             except Exception as e:
                 print(f"Error applying colormap to frame {i}: {e}. Skipping frame.")
                 continue
        else:
            output_frame = normalized_frame_gray # Keep grayscale

        # --- Save Frame ---
        base_filename = f"frame_{str(i).zfill(num_digits)}.{img_format}"
        output_path = os.path.join(final_output_dir, base_filename)

        try:
            success = cv2.imwrite(output_path, output_frame)
            if not success:
                 print(f"Warning: Failed to save frame {i} to {output_path}")
            else:
                 saved_count += 1
                 if (i + 1) % 100 == 0: print(f"  Saved frame {i+1}/{num_frames}...")
        except Exception as e:
            print(f"Error writing image file '{output_path}' for frame {i}: {e}")

    print(f"--- Frame Extraction Complete ---")
    print(f"Successfully saved {saved_count} out of {num_frames} frames to '{final_output_dir}'.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Get list of available colormap names for help text
    core_maps = {
    'jet':    cv2.COLORMAP_JET,
    'hot':    cv2.COLORMAP_HOT,
    'inferno': getattr(cv2, 'COLORMAP_INFERNO', None),
    'magma':   getattr(cv2, 'COLORMAP_MAGMA',   None),
    'plasma':  getattr(cv2, 'COLORMAP_PLASMA',  None),
    'viridis': getattr(cv2, 'COLORMAP_VIRIDIS', None),
    'cool':    getattr(cv2, 'COLORMAP_COOL',    None),
    'hsv':     getattr(cv2, 'COLORMAP_HSV',     None),
    'pink':    getattr(cv2, 'COLORMAP_PINK',    None),
    'bone':    getattr(cv2, 'COLORMAP_BONE',    None),
    'autumn':  getattr(cv2, 'COLORMAP_AUTUMN',  None),
    'winter':  getattr(cv2, 'COLORMAP_WINTER',  None),
    'rainbow': getattr(cv2, 'COLORMAP_RAINBOW', None),
    'ocean':   getattr(cv2, 'COLORMAP_OCEAN',   None),
    'summer':  getattr(cv2, 'COLORMAP_SUMMER',  None),
    'spring':  getattr(cv2, 'COLORMAP_SPRING',  None),
    # coolwarm WILL be skipped if not present
    }

    available_colormaps = [name for name, val in core_maps.items() if val is not None] + ['gray']

    parser = argparse.ArgumentParser(description="Extract and optionally pseudocolor thermal frames from a .mat file.")

    parser.add_argument("mat_file", help="Path to the input .mat file.")
    parser.add_argument("output_dir", help="Path to the BASE directory where 'thermal_images/<matfile_name>_<colormap>/' structure will be created.")
    parser.add_argument("-c", "--colormap", default="jet", choices=available_colormaps,
                        help="Colormap to apply (default: jet). Use 'gray' for no colormap.")
    parser.add_argument("-f", "--format", default="png", choices=["png", "jpg", "tif", "bmp"],
                        help="Output image format (default: png).")

    if len(sys.argv) < 3:
         parser.print_help(); sys.exit(1)

    args = parser.parse_args()

    extract_and_save_frames(args.mat_file, args.output_dir, args.colormap, args.format)