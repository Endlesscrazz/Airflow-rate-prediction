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

def parse_material_from_folder(folder_name: str) -> str:
    """Extracts the material name from the dataset folder name."""
    name = folder_name.lower()
    if "gypsum" in name:
        return "gypsum"
    elif "brickcladding" in name:
        return "brick_cladding"
    elif "hardyboard" in name:
        return "hardyboard"
    return "unknown"


# --- Main Script Logic ---

# IMPORTANT: Update this path to your external drive if needed
DATASETS_ROOT = "/Volumes/One_Touch/Airflow-rate-prediction/datasets" 
OUTPUT_CSV_PATH = "airflow_ground_truth_gypsum.csv"         # CHANGE THIS DEPEDNING ON MATERIAL TYPE WANT TO USE

# --- CONFIGURE YOUR DATASETS HERE ---
DATASET_FOLDERS = [
    "Fluke_Gypsum_07162025_noshutter",
    "Fluke_Gypsum_07252025_noshutter",
    "Fluke_Gypsum_07292025_noshutter",
    # "Fluke_BrickCladding_2holes_0616_2025_noshutter",
    # "Fluke_BrickCladding_2holes_0805_2025_noshutter",
    # "Fluke_BrickCladding_2holes_0808_2025_noshutter",
    #"Fluke_HardyBoard_08132025_2holes_noshutter"
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
    material = parse_material_from_folder(folder_name)
    
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mat') and not f.startswith('._')]
    print(f"  - Found {len(video_files)} '.mat' video files.")
    
    try:
        if is_two_holes:
            # --- MODIFIED: Robust parsing for 2-hole files ---
            gt_path = os.path.join(folder_path, "flow_rate_data.txt")
            # Read the file, skip the header, and provide our own clean column names
            gt_df = pd.read_csv(gt_path, delim_whitespace=True, header=None, skiprows=1,
                                names=['V', 'Pa', 'rate_center', 'rate_corner'])
        else:
            # --- MODIFIED: Robust parsing for 1-hole files ---
            gt_path = os.path.join(folder_path, "flow_rate.txt")
            # Read the file, skip the header, and provide our own clean column names
            gt_df = pd.read_csv(gt_path, delim_whitespace=True, header=None, skiprows=1,
                                names=['V', 'Pa', 'rate_single'])
            
        # Convert all loaded columns to numeric types
        for col in gt_df.columns:
            gt_df[col] = pd.to_numeric(gt_df[col])

        print(f"  - Successfully loaded ground truth from: {os.path.basename(gt_path)}")
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
        
        if voltage is None: continue
            
        gt_row = gt_df[gt_df['V'] == voltage]
        if gt_row.empty:
            print(f"  - WARNING: No ground truth for voltage {voltage} for file '{filename}'. Skipping.")
            continue
        
        pressure = gt_row.iloc[0]['Pa']
        videos_processed_this_folder += 1

        if is_two_holes:
            all_records.append({
                'video_id': video_id, 'hole_id': '1_centerhole', 'airflow_rate': gt_row.iloc[0]['rate_center'],
                'material': material, 'delta_T': delta_T, 'voltage': voltage, 'pressure_Pa': pressure
            })
            all_records.append({
                'video_id': video_id, 'hole_id': '2_cornerhole', 'airflow_rate': gt_row.iloc[0]['rate_corner'],
                'material': material, 'delta_T': delta_T, 'voltage': voltage, 'pressure_Pa': pressure
            })
            records_from_this_folder += 2
        else:
            all_records.append({
                'video_id': video_id, 'hole_id': '1', 'airflow_rate': gt_row.iloc[0]['rate_single'],
                'material': material, 'delta_T': delta_T, 'voltage': voltage, 'pressure_Pa': pressure
            })
            records_from_this_folder += 1

    print(f"  - Processed {videos_processed_this_folder} videos, generating {records_from_this_folder} records.")
    total_videos_processed += videos_processed_this_folder

if all_records:
    final_df = pd.DataFrame(all_records)
    final_df = final_df.sort_values(by=['material', 'voltage', 'delta_T']).reset_index(drop=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n Successfully created combined ground truth file at: {OUTPUT_CSV_PATH}")
    print(f"Total videos processed: {total_videos_processed}")
    print(f"Total records (hole-level samples) created: {len(final_df)}")
    print("\nPreview of the first 5 rows:")
    print(final_df.head())
else:
    print("\n No records were created. Please check folder paths and file contents.")

# python src_feature_based/create_ground_truth_labels.py