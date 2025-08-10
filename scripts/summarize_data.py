# scripts/summarize_data.py
"""
A utility script to scan the raw datasets directory, link videos to their
ground truth airflow rates from a CSV, and print a comprehensive summary.
"""
import os
import sys
import pandas as pd
import fnmatch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_feature_based import config as cfg
from src_feature_based import data_utils

def get_material_info(dataset_subfolder_name):
    """Infers material from the dataset folder name."""
    name = dataset_subfolder_name.lower()
    if "gypsum" in name:
        return "gypsum"
    elif "brickcladding" in name:
        return "brick_cladding"
    return "unknown"

def main():
    """Main function to scan and summarize the data."""
    if not os.path.isdir(cfg.RAW_DATA_ROOT):
        print(f"Error: Datasets root directory not found at '{cfg.RAW_DATA_ROOT}'")
        print("Please check the RAW_DATA_ROOT path in src_feature_based/config.py")
        return

    print(f"--- Scanning Raw Data in '{cfg.RAW_DATA_ROOT}' ---")
    
    # 1. Load the ground truth airflow data from the CSV
    try:
        airflow_map = data_utils.load_airflow_from_csv(cfg.GROUND_TRUTH_CSV_PATH)
        print(f"Successfully loaded {len(airflow_map)} ground truth entries from CSV.")
    except Exception as e:
        print(f"Could not load ground truth file. Aborting. Error: {e}")
        return

    all_video_data = []

    # 2. Walk through the configured dataset directories
    for d_key, d_conf in cfg.DATASET_CONFIGS.items():
        dataset_path = os.path.join(cfg.RAW_DATA_ROOT, d_conf["dataset_subfolder"])
        
        if not os.path.isdir(dataset_path):
            print(f"Warning: Directory not found for '{d_key}': {dataset_path}")
            continue

        for root, _, files in os.walk(dataset_path):
            for mat_file in fnmatch.filter(files, '*.mat'):
                try:
                    # Parse voltage from filename to link to ground truth
                    voltage = data_utils.parse_voltage_from_filename(mat_file)
                    if voltage is None or voltage not in airflow_map:
                        continue

                    airflow_rate = airflow_map[voltage]
                    delta_T = data_utils.parse_delta_T(mat_file)
                    
                    if delta_T is not None:
                        all_video_data.append({
                            'Dataset': d_conf["dataset_subfolder"],
                            'Material': d_conf["material"],
                            'Voltage': voltage,
                            'Airflow_Rate_L/min': airflow_rate,
                            'Delta_T': delta_T,
                            'Filename': mat_file
                        })
                except Exception as e:
                    print(f"Error processing file {mat_file}: {e}")

    if not all_video_data:
        print("\nNo valid video data found. Please check paths and file names.")
        return

    # 3. Create and display the summary DataFrame
    df = pd.DataFrame(all_video_data)
    
    print(f"\n--- Found {len(df)} Total Video Samples ---")
    
    print("\n--- Full Data Listing ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(df.sort_values(by=['Dataset', 'Voltage', 'Delta_T']))
        
    print("\n--- Summary Statistics ---")
    
    print("\nCounts per Airflow Rate (L/min):")
    print(df['Airflow_Rate_L/min'].value_counts().sort_index())

    print("\nCounts per Delta_T:")
    print(df['Delta_T'].value_counts().sort_index())
    
    print("\nCounts per Dataset:")
    print(df['Dataset'].value_counts())

if __name__ == "__main__":
    main()
# python -m scripts.summarize_data