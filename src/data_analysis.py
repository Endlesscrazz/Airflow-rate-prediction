# interactive_ir_visualizer.py
"""
A script to load an IR video from a .mat file and display the last frame
interactively. It shows a grayscale version and a heatmap, and displays
the (x, y) coordinates of the pixel under the mouse cursor.

How to Run:
1. Save this file as `interactive_ir_visualizer.py`.
2. Open your terminal or command prompt.
3. Run the script by providing the path to your .mat file, for example:
   python interactive_ir_visualizer.py "/path/to/your/thermal_video.mat"
"""

import os
import sys
import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# --- Main Functions ---

def load_last_frame(mat_filepath, mat_key='TempFrames'):
    """
    Loads a .mat file and extracts the last frame from the thermal video data.

    Args:
        mat_filepath (str): The full path to the .mat file.
        mat_key (str): The key in the .mat file that contains the video frames.
                       Defaults to 'TempFrames' based on your project structure.

    Returns:
        np.ndarray: A 2D NumPy array representing the last frame.
                    Returns None if an error occurs.

    python3 src/data_analysis.py datasets/dataset_two_holes/T1.4V_2.2Pa_2025-6-16-16-44-18_20_30_10_.mat
    """
    # --- 1. Validate File Path ---
    # Check if the provided file path actually exists before trying to open it.
    if not os.path.exists(mat_filepath):
        print(f"Error: File not found at '{mat_filepath}'")
        return None

    try:
        # --- 2. Load the .mat File ---
        # scipy.io.loadmat reads the MATLAB file into a Python dictionary.
        print(f"Loading data from: {os.path.basename(mat_filepath)}...")
        mat_data = scipy.io.loadmat(mat_filepath)

        # --- 3. Extract the Frame Data ---
        # Check if the specified key exists in the loaded data.
        if mat_key not in mat_data:
            print(f"Error: The key '{mat_key}' was not found in the .mat file.")
            print(f"Available keys are: {list(mat_data.keys())}")
            return None

        # Get the video frames. The data is expected to be in a (Height, Width, NumFrames) format.
        frames = mat_data[mat_key]

        # --- 4. Validate Frame Data ---
        # Ensure the data is a 3D array and has at least one frame.
        if not isinstance(frames, np.ndarray) or frames.ndim != 3:
            print(f"Error: Data under key '{mat_key}' is not a valid 3D video array.")
            return None

        # --- 5. Select and Return the Last Frame ---
        # Array indexing in Python is 0-based. `:, :, -1` selects all rows, all columns,
        # and the very last item along the third axis (the frames axis).
        last_frame = frames[:, :, -1].astype(np.float64)
        print(f"Successfully loaded frame. Dimensions (H, W): {last_frame.shape}")
        return last_frame

    except Exception as e:
        # Catch any other potential errors during file loading or processing.
        print(f"An unexpected error occurred: {e}")
        return None


def visualize_frame_interactively(frame, source_filename):
    """
    Creates an interactive plot with two views of the frame (grayscale and heatmap)
    and displays mouse coordinates on hover.

    Args:
        frame (np.ndarray): The 2D numpy array of the thermal frame to display.
        source_filename (str): The name of the source file for use in titles.
    """
    if frame is None:
        print("Cannot visualize an empty frame.")
        return

    # --- 1. Create the Figure and Subplots ---
    # `plt.subplots(1, 2, ...)` creates a figure containing a grid of 1 row and 2 columns of plots.
    # `fig` is the entire window, and `axes` is a list containing the two individual plot objects.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 2. Grayscale (RGB-like) Visualization ---
    # The first subplot is for the grayscale image.
    ax1 = axes[0]
    # `imshow` is the function to display data as an image.
    # `cmap='gray'` tells matplotlib to use a grayscale colormap.
    ax1.imshow(frame, cmap='gray')
    ax1.set_title("Grayscale View")
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")

    # --- 3. Heatmap Visualization ---
    # The second subplot is for the heatmap.
    ax2 = axes[1]
    # Here, we use `cmap='hot'`. This maps low values to black/red and high values to yellow/white.
    # It's a very common and intuitive colormap for thermal data.
    im = ax2.imshow(frame, cmap='hot')
    ax2.set_title("Heatmap View")
    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")

    # Add a colorbar to the heatmap to show the mapping between colors and temperature values.
    # The `cax` argument ensures the colorbar has a nice layout.
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (Raw Sensor Units)')

    # --- 4. Set up Interactive Coordinate Display ---
    # This is the core of the interactivity. We define a function that matplotlib
    # will call whenever it needs to display coordinates in the status bar.
    def format_coord(x, y):
        # `x` and `y` are the floating-point coordinates from the plot's data space.
        # We round them to the nearest integer to get the pixel index.
        col = int(x + 0.5)
        row = int(y + 0.5)

        # We need to make sure the coordinates are within the bounds of the image.
        if 0 <= col < frame.shape[1] and 0 <= row < frame.shape[0]:
            # If they are valid, get the temperature value at that pixel.
            temp_value = frame[row, col]
            # Format the string for display.
            return f'x={col}, y={row}  |  Temp={temp_value:.2f}'
        else:
            # If the mouse is outside the image area, just show the coordinates.
            return f'x={col}, y={row}'

    # We assign our custom formatting function to both subplots.
    ax1.format_coord = format_coord
    ax2.format_coord = format_coord

    # --- 5. Final Touches and Display ---
    # Set an overall title for the entire figure window.
    fig.suptitle(f'Interactive Visualization for: {source_filename}', fontsize=16)
    # `plt.tight_layout()` automatically adjusts plot parameters for a tight layout.
    plt.tight_layout()
    # `plt.show()` displays the plot window. The script will pause here until you close the window.
    plt.show()


# --- Script Execution Block ---

# This block runs only when the script is executed directly from the command line.
if __name__ == "__main__":
    # --- 1. Set up Command-Line Argument Parser ---
    # `argparse` is the standard Python library for creating command-line interfaces.
    parser = argparse.ArgumentParser(
        description="Load and interactively visualize the last frame of an IR video from a .mat file.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting.
    )
    # We add one required argument: the path to the .mat file.
    parser.add_argument(
        "mat_file_path",
        type=str,
        help="The full path to the input .mat file containing the IR video."
    )
    # We add an optional argument for the key, in case it's different from 'TempFrames'.
    parser.add_argument(
        "--key",
        type=str,
        default='TempFrames',
        help="The key within the .mat file that holds the video data.\n(default: 'TempFrames')"
    )

    # If the script is run with no arguments, print the help message and exit.
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # --- 2. Run the Main Logic ---
    # Call our function to load the last frame from the specified file.
    last_frame_data = load_last_frame(args.mat_file_path, args.key)

    # If the frame was loaded successfully, call the visualization function.
    if last_frame_data is not None:
        visualize_frame_interactively(last_frame_data, os.path.basename(args.mat_file_path))
    else:
        print("\nExiting due to errors.")
        sys.exit(1)

