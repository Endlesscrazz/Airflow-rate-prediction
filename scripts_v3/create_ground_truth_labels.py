# scripts/create_ground_truth_labels.py
import os
import pandas as pd
import numpy as np
import re
import sys

# --- Parsing Helper Functions ---


def parse_voltage_from_new_filename(filename: str) -> float:
    """Extracts voltage (e.g., T1.4V) from filename."""
    match = re.search(r'T(\d+(\.\d+)?)V', filename)
    return float(match.group(1)) if match else None


def parse_delta_T_from_new_filename(filename: str) -> float:
    """Extracts delta_T from filename like ..._20_34_14_.mat (14 is deltaT)."""
    parts = filename.replace('.mat', '').split('_')
    try:
        # Check second to last part
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


def parse_delta_T_10hole(filename: str) -> float:
    """
    Parses delta_T for the 10-hole dataset.
    Format: temp_YYYY-MM-DD-HH-MM-SS_Outdoor_Indoor_DeltaT_.mat
    Example: ..._23_35_12_.mat -> returns 12.0
    """
    clean_name = filename.replace('.mat', '')
    if clean_name.endswith('_'):
        clean_name = clean_name[:-1]
    parts = clean_name.split('_')
    try:
        # Return the last number
        return float(parts[-1])
    except:
        return None


def main():
    # --- PATH CONFIGURATION ---
    # Update this to match your environment
   # DATASETS_ROOT = "/scratch/general/vast/u1527145/datasets"
    DATASETS_ROOT = "/Volumes/One_Touch/Airflow-rate-prediction/datasets" # Local debugging

    # This output file will be used by create_metadata_v3.py
    OUTPUT_CSV_PATH = "airflow_ground_truth_gypsum_combined.csv"

    DATASET_CONFIGS = {
    ## GYPSUM DATASET ALL
        # --- NEW 10-HOLE DATASET ---
        "gypsum_10holes_0903": {
            "material": "gypsum",
            "dataset_subfolder": "Fluke_Gypsum_09032025_10holes_noshutter_Sameem",
            "structure_type": "10_hole",
            "gt_file": "flow_rates.txt",
            "session": "gypsum_10hole"
        },
        # --- OLDER DATASETS ---
        "gypsum_0716": {
            "material": "gypsum", 
            "dataset_subfolder": "Fluke_Gypsum_07162025_noshutter", 
            "structure_type": "new", # Voltage based filenames
            "gt_file": "flow_rate.txt", 
            "session": "gypsum_new"
        },
        "gypsum_0725": {
            "material": "gypsum", 
            "dataset_subfolder": "Fluke_Gypsum_07252025_noshutter", 
            "structure_type": "new", 
            "gt_file": "flow_rate.txt", 
            "session": "gypsum_new"
        },
        "gypsum_0729": {
            "material": "gypsum", 
            "dataset_subfolder": "Fluke_Gypsum_07292025_noshutter", 
            "structure_type": "new", 
            "gt_file": "flow_rate.txt", 
            "session": "gypsum_new"
        },
        # Add any others (e.g. 0307) if you have them and they are Gypsum
    }

    all_records = []
    print(f"--- Starting Ground Truth Creation -> {OUTPUT_CSV_PATH} ---")

    for config_key, config in DATASET_CONFIGS.items():
        folder_path = os.path.join(DATASETS_ROOT, config['dataset_subfolder'])
        print(f"\nProcessing Dataset: {config_key}")

        if not os.path.isdir(folder_path):
            print(f"  - WARNING: Folder not found at {folder_path}. Skipping.")
            continue

        structure = config.get('structure_type', 'new')
        gt_file = config['gt_file']
        gt_path = os.path.join(folder_path, gt_file)

        try:
            # ==========================================
            # LOGIC FOR NEW 10-HOLE DATASET
            # ==========================================
            if structure == "10_hole":
                # 1. Read Flow Rates Text File
                # Format: Tab/Space separated. Col 0=Pressure, Cols 1-10=Rates for Hole 1-10
                if not os.path.exists(gt_path):
                    print(f"  - Error: GT file not found: {gt_path}")
                    continue

                # specific parser for the 10hole format
                gt_df = pd.read_csv(gt_path, sep=r'\s+',
                                    skiprows=1, header=None)

                # 2. Iterate through Pressure Subfolders (2P, 5P, etc.)
                subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(
                    os.path.join(folder_path, d))]

                count_for_dataset = 0
                for subdir in subdirs:
                    if not subdir.endswith('P'):
                        continue  # Skip unrelated folders

                    try:
                        pressure_val = int(subdir[:-1])  # "2P" -> 2
                    except ValueError:
                        continue

                    # 3. Find GT Row for this pressure
                    # Check column 0 for pressure value
                    gt_row = gt_df[gt_df.iloc[:, 0] == pressure_val]
                    if gt_row.empty:
                        print(
                            f"  - Warning: No flow rate data found for Pressure {pressure_val}P")
                        continue

                    # Extract rates for Holes 1-10 (Columns 1 through 10)
                    rates = gt_row.iloc[0, 1:11].values

                    # 4. Process all video files in this pressure folder
                    pressure_dir = os.path.join(folder_path, subdir)
                    mat_files = [f for f in os.listdir(pressure_dir) if f.endswith(
                        '.mat') and not f.startswith('._')]

                    for f in mat_files:
                        video_id = f.replace('.mat', '')
                        delta_T = parse_delta_T_10hole(f)

                        if delta_T is None:
                            print(
                                f"  - Warning: Could not parse delta_T from {f}")
                            continue

                        # Create 10 entries for this single video (one for each hole)
                        for i in range(10):
                            hole_num = i + 1
                            flow_rate = rates[i]

                            all_records.append({
                                'video_id': video_id,
                                'hole_id': str(hole_num),
                                'airflow_rate': float(flow_rate),
                                'delta_T': float(delta_T),
                                'pressure_Pa': float(pressure_val),
                                'material': config['material'],
                                'session': config['session'],
                                'source_dataset_key': config_key
                            })
                            count_for_dataset += 1

                print(
                    f"  - Added {count_for_dataset} records from {config_key}")

            # ==========================================
            # BACKWARD COMPATIBILITY FOR V2 SCRIPTS
            # ==========================================
            else:
                # Load GT file (Excel or CSV)
                if gt_file.endswith('.xlsx'):
                    gt_df = pd.read_excel(gt_path).iloc[:, :3]
                    gt_df.columns = ['V', 'Pa', 'rate_single']
                else:
                    # Assume standard CSV/Txt
                    is_two_holes = "2holes" in config['dataset_subfolder'].lower(
                    )
                    if is_two_holes:
                        gt_df = pd.read_csv(
                            gt_path, sep=r'\s+', header=None, skiprows=1, names=['V', 'Pa', 'rate_1', 'rate_2'])
                    else:
                        gt_df = pd.read_csv(
                            gt_path, sep=r'\s+', header=None, skiprows=1, names=['V', 'Pa', 'rate_single'])

                gt_df['V'] = gt_df['V'].astype(float)

                # Walk through folders
                count_for_dataset = 0
                for subfolder_path, _, files in os.walk(folder_path):
                    video_files = [f for f in files if f.endswith(
                        '.mat') and not f.startswith('._')]
                    for filename in video_files:
                        video_id = filename.replace('.mat', '')

                        # Parse Voltage/DT
                        voltage = parse_voltage_from_new_filename(filename)
                        if voltage is None:
                            voltage = parse_voltage_from_old_foldername(
                                os.path.basename(subfolder_path))

                        delta_T = parse_delta_T_from_new_filename(filename)
                        if delta_T is None:
                            delta_T = parse_delta_T_from_old_filename(filename)

                        if voltage is None:
                            continue

                        # Match GT
                        gt_row = gt_df[np.isclose(
                            gt_df['V'], voltage, atol=0.01)]
                        if gt_row.empty:
                            continue

                        base_rec = {
                            'video_id': video_id,
                            'delta_T': delta_T,
                            'pressure_Pa': gt_row.iloc[0]['Pa'],
                            'material': config['material'],
                            'session': config['session'],
                            'source_dataset_key': config_key
                        }

                        # Add records
                        if "2holes" in config['dataset_subfolder'].lower():
                            all_records.append(
                                {**base_rec, 'hole_id': '1', 'airflow_rate': gt_row.iloc[0]['rate_1']})
                            all_records.append(
                                {**base_rec, 'hole_id': '2', 'airflow_rate': gt_row.iloc[0]['rate_2']})
                            count_for_dataset += 2
                        else:
                            all_records.append(
                                {**base_rec, 'hole_id': '1', 'airflow_rate': gt_row.iloc[0]['rate_single']})
                            count_for_dataset += 1

                print(
                    f"  - Added {count_for_dataset} records from {config_key}")

        except Exception as e:
            print(f"  - ERROR processing {config_key}: {e}")
            import traceback
            traceback.print_exc()

    # --- SAVE OUTPUT ---
    if all_records:
        final_df = pd.DataFrame(all_records)
        # Sort for consistency
        cols_to_sort = ['session', 'pressure_Pa', 'delta_T', 'hole_id']
        # Filter strictly for columns that exist
        cols_to_sort = [c for c in cols_to_sort if c in final_df.columns]

        final_df = final_df.sort_values(by=cols_to_sort)
        final_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSUCCESS: Ground Truth file saved to: {OUTPUT_CSV_PATH}")
        print(f"Total Rows: {len(final_df)}")
        print("Sample Data:")
        print(final_df.head())
    else:
        print("\nFAILURE: No records were generated.")


if __name__ == "__main__":
    main()

# python scripts_v3/create_ground_truth_labels.py