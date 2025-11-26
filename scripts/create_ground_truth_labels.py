# scripts/create_ground_truth_labels.py
import os
import pandas as pd
import numpy as np
import re

def parse_voltage_from_new_filename(filename: str) -> float:
    match = re.search(r'T(\d+(\.\d+)?)V', filename)
    return float(match.group(1)) if match else None

def parse_delta_T_from_new_filename(filename: str) -> float:
    parts = filename.replace('.mat', '').split('_')
    try:
        # Example: T1.4V_..._20_34_14_.mat -> delta_T is 14
        if len(parts) > 3 and parts[-2].replace('.', '', 1).isdigit():
            return float(parts[-2])
    except (ValueError, IndexError):
        return None
    return None

def parse_voltage_from_old_foldername(foldername: str) -> float:
    match = re.search(r'(\d+(\.\d+)?)V', foldername)
    return float(match.group(1)) if match else None

def parse_delta_T_from_old_filename(filename: str) -> float:
    parts = filename.replace('.mat', '').split('_')
    for part in reversed(parts[:-1]):
        if part.replace('.', '', 1).isdigit():
            try:
                return float(part)
            except ValueError:
                continue
    return None

def parse_pressure_from_multi_hole_filename(filename: str) -> int:
    match = re.search(r'T(\d+)P', filename)
    return int(match.group(1)) if match else None

def main():
    # Configure this for the machine you are running on
    # DATASETS_ROOT = "/scratch/general/vast/u1527145/datasets"

    ## LOCAL Macbook
    DATASETS_ROOT = "/Volumes/One_Touch/Airflow-rate-prediction/datasets"
    # The output file name should match what's in config_v2.py
    OUTPUT_CSV_PATH = "airflow_ground_truth_gypsum_all.csv"

    DATASET_CONFIGS = {
        # New Gypsum
        "gypsum_0716": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07162025_noshutter", "structure_type": "new", "gt_file": "flow_rate.txt", "session": "gypsum_new"},
        "gypsum_0725": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07252025_noshutter", "structure_type": "new", "gt_file": "flow_rate.txt", "session": "gypsum_new"},
        "gypsum_0729": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07292025_noshutter", "structure_type": "new", "gt_file": "flow_rate.txt", "session": "gypsum_new"},
        # Old Gypsum
        "gypsum_0307": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_03072025", "structure_type": "old", "gt_file": "flow_rates.xlsx", "session": "gypsum_old"},
    }

    all_records = []
    print("--- Starting Ground Truth CSV Creation ---")

    for config_key, config in DATASET_CONFIGS.items():
        folder_path = os.path.join(DATASETS_ROOT, config['dataset_subfolder'])
        print(f"\nProcessing folder: {config['dataset_subfolder']}")

        if not os.path.isdir(folder_path):
            print(f"  - WARNING: Folder not found. Skipping.")
            continue

        material, structure_type, gt_file, session = config['material'], config['structure_type'], config['gt_file'], config['session']
        gt_path = os.path.join(folder_path, gt_file)
        is_two_holes = "2holes" in config['dataset_subfolder'].lower()

        try:
            # --- MODIFICATION START: Handle .xlsx vs .txt/.csv ---
            if gt_file.endswith('.xlsx'):
                # Read Excel file
                # Assuming the Excel file has headers and the columns are Voltage, Pressure, Flow Rate
                gt_df = pd.read_excel(gt_path)
                
                # Standardize column names to match logic below
                # We assume the first 3 columns correspond to V, Pa, rate_single
                # Rename columns to standard internal names
                gt_df = gt_df.iloc[:, :3] # Ensure we only take relevant columns
                gt_df.columns = ['V', 'Pa', 'rate_single']
                
            else:
                # Read Text/CSV file (Existing Logic)
                if is_two_holes:
                    gt_df = pd.read_csv(gt_path, sep='\s+', header=None, skiprows=1, names=['V', 'Pa', 'rate_1', 'rate_2'])
                else: 
                    gt_df = pd.read_csv(gt_path, sep='\s+', header=None, skiprows=1, names=['V', 'Pa', 'rate_single'])
            # --- MODIFICATION END ---
            
            gt_df['V'] = gt_df['V'].astype(float)
            print(f"  - Successfully loaded ground truth from: {gt_file}")
        except Exception as e:
            print(f"  - ERROR: Could not read GT file '{gt_path}'. Error: {e}. Skipping.")
            continue

        processed_videos_count = 0
        for subfolder_path, _, files in os.walk(folder_path):
            video_files = [f for f in files if f.endswith('.mat') and not f.startswith('._')]
            if not video_files:
                continue

            for filename in video_files:
                video_id = filename.replace('.mat', '')
                
                # Try new naming convention first
                voltage = parse_voltage_from_new_filename(filename)
                delta_T = parse_delta_T_from_new_filename(filename)

                # Fallback to old naming convention if new fails
                if voltage is None:
                    voltage = parse_voltage_from_old_foldername(os.path.basename(subfolder_path))
                
                if delta_T is None:
                    delta_T = parse_delta_T_from_old_filename(filename)

                if voltage is None: 
                    print(f"  - WARNING: Could not parse voltage for '{filename}'. Skipping.")
                    continue
                
                # Find matching row in GT dataframe
                # Using tolerance for float comparison just in case
                gt_row = gt_df[np.isclose(gt_df['V'], voltage, atol=0.01)]
                
                if gt_row.empty: 
                    print(f"  - WARNING: No ground truth found for voltage {voltage} ('{filename}'). Skipping.")
                    continue

                base_record = {
                    'material': material, 'delta_T': delta_T, 'voltage': voltage, 
                    'pressure_Pa': gt_row.iloc[0]['Pa'], 'session': session
                }
                
                if is_two_holes:
                    all_records.append({'video_id': video_id, 'hole_id': '1', 'airflow_rate': gt_row.iloc[0]['rate_1'], **base_record})
                    all_records.append({'video_id': video_id, 'hole_id': '2', 'airflow_rate': gt_row.iloc[0]['rate_2'], **base_record})
                else:
                    all_records.append({'video_id': video_id, 'hole_id': '1', 'airflow_rate': gt_row.iloc[0]['rate_single'], **base_record})
                
                processed_videos_count += 1
        print(f"  - Processed {processed_videos_count} videos.")

    if all_records:
        final_df = pd.DataFrame(all_records)
        # Sort for cleanliness
        final_df = final_df.sort_values(by=['material', 'session', 'voltage', 'delta_T']).reset_index(drop=True)
        final_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSuccessfully created combined ground truth file at: {OUTPUT_CSV_PATH}")
        print(f"Total records created: {len(final_df)}")
    else:
        print("\nNo records were created.")

if __name__ == "__main__":
    main()

# python scripts/create_ground_truth_labels.py



