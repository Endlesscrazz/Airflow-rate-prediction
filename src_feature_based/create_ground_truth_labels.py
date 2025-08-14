# create_combined_ground_truth.py
import os
import pandas as pd
import re

# --- Helper functions ---
def parse_voltage_from_filename(filename: str) -> float:
    """Extracts the fan voltage from a filename (e.g., 'T1.4V_...')."""
    match = re.search(r'T(\d+(\.\d+)?)V', filename)
    if match:
        return float(match.group(1))
    return None

def parse_delta_T(filename: str) -> float:
    """Extracts delta T from a filename (e.g., '..._22_30_8_.mat' -> 8.0)."""
    parts = filename.replace('.mat', '').split('_')
    try:
        if len(parts) > 3 and parts[-2].replace('.', '', 1).isdigit():
            return float(parts[-2])
    except (ValueError, IndexError):
        pass
    return None

# --- Main Script Logic ---

DATASETS_ROOT = "datasets"
OUTPUT_CSV_PATH = "airflow_ground_truth_combined.csv"

DATASET_FOLDERS = [
    "Fluke_Gypsum_07162025_noshutter",
    "Fluke_Gypsum_07252025_noshutter",
    "Fluke_Gypsum_07292025_noshutter",
    "Fluke_BrickCladding_2holes_0616_2025_noshutter",
    "Fluke_BrickCladding_2holes_0805_2025_noshutter",
    "Fluke_BrickCladding_2holes_0808_2025_noshutter"
]

all_records = []
total_videos_processed = 0

print("Starting to process dataset folders...")

for folder_name in DATASET_FOLDERS:
    folder_path = os.path.join(DATASETS_ROOT, folder_name)
    print(f"\nProcessing folder: {folder_name}")

    if not os.path.isdir(folder_path):
        print(f"  - WARNING: Folder not found. Skipping.")
        continue
        
    is_two_holes = "2holes" in folder_name.lower()
    material = "brick_cladding" if "brickcladding" in folder_name.lower() else "gypsum"
    
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    print(f"  - Found {len(video_files)} '.mat' files in this directory.")
    
    try:
        if is_two_holes:
            gt_path = os.path.join(folder_path, "flowrate data.txt")
            gt_df = pd.read_csv(gt_path, delim_whitespace=True, header=None, skiprows=1)
            gt_df.columns = ['V', 'Pa', 'rate_large', 'rate_small']
        else:
            # --- MODIFICATION START ---
            # Now reads the clean CSV for gypsum datasets
            gt_path = os.path.join(folder_path, "airflow_ground_truth.csv")
            # Read it as a standard CSV, pandas will handle the header correctly
            gt_df = pd.read_csv(gt_path)
            # Rename the 'L/min' column to be consistent with our internal variable name
            gt_df = gt_df.rename(columns={'L/min': 'rate_single'})
            # --- MODIFICATION END ---
            
        # Convert columns to numeric types after loading
        for col in gt_df.columns:
            gt_df[col] = pd.to_numeric(gt_df[col])

        print(f"  - Successfully loaded ground truth from: {gt_path}")
    except FileNotFoundError:
        print(f"  - ERROR: Ground truth file not found in '{folder_path}'. Skipping.")
        continue
    except Exception as e:
        print(f"  - ERROR: Could not read ground truth file. Error: {e}. Skipping.")
        continue

    records_from_this_folder = 0
    videos_processed_this_folder = 0

    for filename in video_files:
        video_id = filename.replace('.mat', '')
        voltage = parse_voltage_from_filename(filename)
        delta_T = parse_delta_T(filename)
        
        if voltage is None:
            print(f"  - WARNING: Could not parse voltage from '{filename}'. Skipping.")
            continue
            
        gt_row = gt_df[gt_df['V'] == voltage]
        if gt_row.empty:
            print(f"  - WARNING: No ground truth entry found for voltage {voltage} for file '{filename}'. Skipping.")
            continue
        
        pressure = gt_row.iloc[0]['Pa']
        videos_processed_this_folder += 1

        if is_two_holes:
            rate_large = gt_row.iloc[0]['rate_large']
            rate_small = gt_row.iloc[0]['rate_small']
            
            all_records.append({
                'video_id': video_id, 'hole_id': '1_largehole', 'airflow_rate': rate_large,
                'material': material, 'delta_T': delta_T, 'voltage': voltage, 'pressure_Pa': pressure
            })
            all_records.append({
                'video_id': video_id, 'hole_id': '2_smallhole', 'airflow_rate': rate_small,
                'material': material, 'delta_T': delta_T, 'voltage': voltage, 'pressure_Pa': pressure
            })
            records_from_this_folder += 2
        else:
            rate_single = gt_row.iloc[0]['rate_single']
            all_records.append({
                'video_id': video_id, 'hole_id': '1', 'airflow_rate': rate_single,
                'material': material, 'delta_T': delta_T, 'voltage': voltage, 'pressure_Pa': pressure
            })
            records_from_this_folder += 1

    print(f"  - Processed {videos_processed_this_folder} videos, generating {records_from_this_folder} records for the '{material}' material.")
    total_videos_processed += videos_processed_this_folder


if all_records:
    final_df = pd.DataFrame(all_records)
    final_df = final_df.sort_values(by=['material', 'voltage', 'delta_T']).reset_index(drop=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Successfully created combined ground truth file at: {OUTPUT_CSV_PATH}")
    print(f"Total videos processed across all folders: {total_videos_processed}")
    print(f"Total records (hole-level samples) created: {len(final_df)}")
    print("\nPreview of the first 5 rows:")
    print(final_df.head())
else:
    print("\n❌ No records were created. Please check folder paths and file contents.")