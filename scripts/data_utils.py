# data_utils.py
"""Functions for loading and parsing raw data."""

import os
import re
import numpy as np
from scipy.io import loadmat
from archive.src_feature_based import old_cfg

def parse_airflow_rate(folder_name):
    """
    Extract the numeric airflow rate (fan voltage) from a folder name like 'FanPower_1.6V'.
    Returns float if found, otherwise raises ValueError.
    """
    match = re.search(r'FanPower_(\d+(\.\d+)?)V', folder_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not parse airflow rate from folder name: {folder_name}")

def parse_delta_T(mat_filename):
    """
    Extract delta T from the .mat file name.
    Looks for the last numeric token before '.mat'.
    """
    base_name = mat_filename.replace('.mat', '')
    tokens = base_name.split('_')
    if len(tokens) < 2:
        print(f"Warning: Could not parse delta T from {mat_filename}, too few tokens.")
        return None # Return None if parsing fails

    # Be slightly more robust: search for the last token that looks like a float
    for token in reversed(tokens):
        token = token.strip('_')
        try:
            return float(token)
        except ValueError:
            continue # Try the previous token

    print(f"Warning: Could not parse delta T from {mat_filename}, no float token found.")
    return None

def load_raw_data(dataset_folder):
    """
    Loads file paths, raw frames, delta_T, and airflow rate for each sample.

    Args:
        dataset_folder (str): Path to the root dataset folder.

    Returns:
        list: A list of dictionaries, each containing 'filepath', 'frames',
              'delta_T', and 'airflow_rate' for one sample.
              Returns an empty list if no data is found or critical errors occur.
    """
    raw_data = []
    if not os.path.isdir(dataset_folder):
        print(f"Error: Dataset folder not found at {dataset_folder}")
        return raw_data

    print(f"Starting data loading from: {dataset_folder}")
    for folder_name in sorted(os.listdir(dataset_folder)): # Sort for consistent order
        folder_path = os.path.join(dataset_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        try:
            airflow_rate = parse_airflow_rate(folder_name)
            print(f"Processing folder: {folder_name} (Airflow Rate: {airflow_rate})")
        except ValueError as e:
            print(f"Skipping folder '{folder_name}' - {e}")
            continue

        mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".mat")]) # Sort files
        if not mat_files:
            print(f"Warning: No .mat files found in {folder_name}")
            continue

        for mat_file in mat_files:
            mat_filepath = os.path.join(folder_path, mat_file)
            delta_T = parse_delta_T(mat_file)
            if delta_T is None:
                print(f"Skipping {mat_file} due to delta_T parsing error.")
                continue

            try:
                mat_data = loadmat(mat_filepath, squeeze_me=True)

                if old_cfg.MAT_FRAMES_KEY not in mat_data:
                     print(f"Error: '{old_cfg.MAT_FRAMES_KEY}' key not found in {mat_file}. Available keys: {list(mat_data.keys())}. Skipping file.")
                     continue

                frames = mat_data[old_cfg.MAT_FRAMES_KEY]

                # Handle potential inconsistencies in frame structure
                if frames.ndim == 1 or (frames.dtype == object):
                    # Try stacking assuming it's a list/array of 2D frames
                    try:
                         frames = np.stack([np.asarray(frame) for frame in frames])
                    except ValueError as stack_err:
                         print(f"Error stacking frames in {mat_file}: {stack_err}. Skipping.")
                         continue
                elif frames.ndim == 2: # Single frame video? Wrap it.
                    frames = frames[np.newaxis, :, :]
                elif frames.ndim == 3:
                     pass # Expected format (frames, height, width)
                else:
                    print(f"Warning: Unexpected frames shape {frames.shape} in {mat_file}. Skipping.")
                    continue

                if frames.shape[1] == 0 or frames.shape[2] == 0:
                    print(f"Warning: Frames have zero dimension size {frames.shape} in {mat_file}. Skipping.")
                    continue

                raw_data.append({
                    "filepath": mat_filepath,
                    "frames": frames.astype(np.float32), # Ensure consistent type
                    "delta_T": float(delta_T), # Ensure float
                    "airflow_rate": float(airflow_rate) # Ensure float
                })
            except Exception as e:
                print(f"Error loading or processing {mat_file}: {type(e).__name__} - {e}")
                continue
    print(f"Finished data loading. Found {len(raw_data)} samples.")
    return raw_data