# data_anaylsis.py
"""
A simple utility script to inspect a thermal .mat file and display its
metadata, including the calculated Frames Per Second (FPS).

Usage:
  python get_video_info.py /path/to/your/video.mat
"""

import scipy.io
import numpy as np
import sys
import os

def analyze_mat_file(file_path):
    """
    Loads a .mat file, extracts video data, and prints its metadata.
    """
    if not os.path.isfile(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"--- Analyzing Metadata for: {os.path.basename(file_path)} ---")

    try:
        # 1. Load the .mat file
        mat_data = scipy.io.loadmat(file_path)
        print(f"Found variables in .mat file: {list(mat_data.keys())}")

        # 2. Find the video data array
        # Common keys are 'TempFrames', 'frames', etc. We'll search for them.
        video_data = None
        possible_keys = ['TempFrames', 'frames', 'video', 'data']
        
        for key in possible_keys:
            if key in mat_data and isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 3:
                video_data = mat_data[key]
                print(f"Found video data under key: '{key}'")
                break
        
        if video_data is None:
            print("\nError: Could not find a 3D video data array in the .mat file.")
            print("Please check the variable names inside your file.")
            return
            
        # 3. Extract and print metadata
        height, width, num_frames = video_data.shape
        data_type = video_data.dtype
        
        print("\n--- Video Metadata ---")
        print(f"  Dimensions (H x W): {height} x {width} pixels")
        print(f"  Number of Frames:   {num_frames}")
        print(f"  Data Type:          {data_type}")

        # 4. Calculate and print the FPS
        # This assumes a fixed video duration of 30 seconds, as per our dataset.
        assumed_duration_sec = 30.0
        if num_frames > 0:
            fps = num_frames / assumed_duration_sec
            print(f"  Calculated FPS:     {fps:.2f} (based on {num_frames} frames over {assumed_duration_sec}s)")
        else:
            print("  Calculated FPS:     N/A (no frames found)")
        
        print("-" * 25)

    except Exception as e:
        print(f"\nAn error occurred while processing the file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_video_info.py <path_to_mat_file>")
        sys.exit(1)
        
    mat_file_path = sys.argv[1]
    analyze_mat_file(mat_file_path)