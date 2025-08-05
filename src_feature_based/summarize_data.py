# src_feature_based/summarize_data.py
"""
A utility script to scan the raw datasets directory and print a summary
of all video files, including their material, hole configuration, airflow rate,
and delta_T.
"""
import os
import re
import pandas as pd
import fnmatch

# --- Configuration ---
# This points to the parent directory containing all your dataset subfolders.
# It assumes the script is run from the project root.
DATASETS_ROOT_DIR = "datasets"

# --- Parsing Functions (copied from data_utils.py for standalone use) ---

def parse_airflow_rate(folder_name):
    """Extracts fan voltage from a folder name like 'FanPower_1.6V'."""
    match = re.search(r'FanPower_(\d+(\.\d+)?)V', folder_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Add a check for the format in dataset_two_holes_brickcladding (e.g., T1.4V_2.2Pa...)
    match = re.search(r'T(\d+(\.\d+)?)V', folder_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def parse_delta_T(mat_filename):
    """Extracts delta T from the .mat file name."""
    base_name = mat_filename.replace('.mat', '')
    tokens = base_name.split('_')
    if len(tokens) < 2:
        return None
    for token in reversed(tokens):
        token = token.strip('_')
        try:
            return float(token)
        except (ValueError, TypeError):
            continue
    return None

def get_material_info(dataset_subfolder_name):
    """Infers material and hole count from the dataset folder name."""
    name = dataset_subfolder_name.lower()
    material = "unknown"
    holes = "unknown"

    if "gypsum" in name:
        material = "gypsum"
    elif "brickcladding" in name:
        material = "brick_cladding"

    if "two_holes" in name:
        holes = 2
    elif "single_hole" in name or material != "unknown": # Assume single if not specified
        holes = 1
        
    return material, holes

def main():
    """Main function to scan and summarize the data."""
    if not os.path.isdir(DATASETS_ROOT_DIR):
        print(f"Error: Datasets root directory not found at '{DATASETS_ROOT_DIR}'")
        print("Please run this script from the project's root directory.")
        return

    print(f"--- Scanning Raw Data in '{DATASETS_ROOT_DIR}' ---")
    all_video_data = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(DATASETS_ROOT_DIR):
        # We only care about directories that contain .mat files
        if not any(f.endswith('.mat') for f in files):
            continue

        try:
            # Get metadata from folder names
            relative_path = os.path.relpath(root, DATASETS_ROOT_DIR)
            path_parts = relative_path.split(os.sep)
            
            dataset_name = path_parts[0]
            airflow_folder = path_parts[-1]
            
            material, num_holes = get_material_info(dataset_name)
            airflow_rate = parse_airflow_rate(airflow_folder)

            if airflow_rate is None:
                # Handle cases where airflow is in the filename itself
                # (like in dataset_two_holes_brickcladding)
                airflow_rate = parse_airflow_rate(files[0]) if files else None

            if airflow_rate is None:
                print(f"Warning: Could not determine airflow rate for folder: {root}")
                continue

            # Process each .mat file in the directory
            for mat_file in fnmatch.filter(files, '*.mat'):
                delta_T = parse_delta_T(mat_file)
                if delta_T is not None:
                    all_video_data.append({
                        'Dataset': dataset_name,
                        'Material': material,
                        'Num_Holes': num_holes,
                        'Airflow_Rate': airflow_rate,
                        'Delta_T': delta_T,
                        'Filename': mat_file
                    })
        except Exception as e:
            print(f"Error processing directory {root}: {e}")

    if not all_video_data:
        print("\nNo valid video data found.")
        return

    # Create a pandas DataFrame for pretty printing
    df = pd.DataFrame(all_video_data)
    
    print(f"\n--- Found {len(df)} Total Video Samples ---")
    
    # Print a summary of the full dataset
    print("\n--- Full Data Listing ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df.sort_values(by=['Dataset', 'Airflow_Rate', 'Delta_T']))
        
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print("\nCounts per Dataset:")
    print(df['Dataset'].value_counts())
    
    print("\nCounts per Material:")
    print(df['Material'].value_counts())
    
    print("\nCounts per Airflow Rate:")
    print(df['Airflow_Rate'].value_counts().sort_index())


if __name__ == "__main__":
    main()