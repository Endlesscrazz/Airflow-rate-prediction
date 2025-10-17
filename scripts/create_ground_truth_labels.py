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
        if len(parts) > 3 and parts[-2].replace('.', '', 1).isdigit():
            return float(parts[-2])
    except (ValueError, IndexError): return None

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

def parse_delta_T_from_multi_hole_filename(filename: str) -> float:
    # This logic is identical to the old parser, so we can reuse it
    return parse_delta_T_from_old_filename(filename)


def main():
    DATASETS_ROOT = "/Volumes/One_Touch/Airflow-rate-prediction/datasets" 
    OUTPUT_CSV_PATH = "airflow_ground_truth_gypsum_all.csv"

    DATASET_CONFIGS = {
        # New Gypsum
        "gypsum_0716": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07162025_noshutter", "structure_type": "new", "gt_file": "flow_rate.txt", "session": "gypsum_new"},
        "gypsum_0725": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07252025_noshutter", "structure_type": "new", "gt_file": "flow_rate.txt", "session": "gypsum_new"},
        "gypsum_0729": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07292025_noshutter", "structure_type": "new", "gt_file": "flow_rate.txt", "session": "gypsum_new"},
        # Old Gypsum
        "gypsum_0307": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_03072025", "structure_type": "old", "gt_file": "flow_rates.xlsx", "session": "gypsum_old"},
        # "gypsum_0903_10holes": {
        #     "material": "gypsum", 
        #     "dataset_subfolder": "Fluke_Gypsum_09032025_10holes_noshutter", 
        #     "structure_type": "multi_hole_csv", 
        #     "gt_file": "flow_rate.csv", 
        #     "session": "gypsum_10_hole"
        # },
        # # Brick Cladding (treating them all as one session for now, can be split later if needed)
        # "brick_cladding_0616": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0616_2025_noshutter", "structure_type": "new", "gt_file": "flowrate_data.txt", "session": "brick_cladding_all"},
        # "brick_cladding_0805": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0805_2025_noshutter", "structure_type": "new", "gt_file": "flowrate_data.txt", "session": "brick_cladding_all"},
        # "brick_cladding_0808": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0808_2025_noshutter", "structure_type": "new", "gt_file": "flowrate_data.txt", "session": "brick_cladding_all"},
    
        # # New HardyBoard
        #"hardyboard_0813": {"material": "hardyboard", "dataset_subfolder": "Fluke_HardyBoard_08132025_2holes_noshutter", "structure_type": "new", "gt_file": "flow_rate_data.txt", "session": "hardyboard_new"},
        # Old HardyBoard
        #"hardyboard_0313": {"material": "hardyboard", "dataset_subfolder": "Fluke_HardyBoard_03132025", "structure_type": "old", "gt_file": "flow_rate.txt", "session": "hardyboard_old"},

    }

    all_records = []
    print("Starting to process dataset folders...")

    for config_key, config in DATASET_CONFIGS.items():
        folder_path = os.path.join(DATASETS_ROOT, config['dataset_subfolder'])
        print(f"\nProcessing folder: {config['dataset_subfolder']}")

        if not os.path.isdir(folder_path):
            print(f"  - WARNING: Folder not found. Skipping."); continue
        
        # <<< CHANGE: ENSURE SESSION IS LOADED >>>
        material, structure_type, gt_file, session = config['material'], config['structure_type'], config['gt_file'], config['session']
        gt_path = os.path.join(folder_path, gt_file)
        is_two_holes = "2holes" in config['dataset_subfolder'].lower()

        if structure_type == 'multi_hole_csv':
            try:
                # Read the CSV, skip the "Table 1" row, and set the first column as the index
                gt_df = pd.read_csv(gt_path, skiprows=1, index_col=0)
                print(f"  - Successfully loaded multi-hole CSV ground truth from: {gt_file}")
            except Exception as e:
                print(f"  - ERROR: Could not read multi-hole CSV GT file '{gt_path}'. Error: {e}. Skipping."); continue

            processed_videos_count = 0
            video_files = [f for f in os.listdir(folder_path) if f.endswith('.mat') and not f.startswith('._')]
            for filename in video_files:
                video_id = filename.replace('.mat', '')
                pressure = parse_pressure_from_multi_hole_filename(filename)
                delta_T = parse_delta_T_from_multi_hole_filename(filename)
                if pressure is None: continue

                try:
                    rate_series = gt_df.loc[pressure]
                except KeyError:
                    print(f"  - WARNING: Pressure {pressure} from filename '{filename}' not found in GT file. Skipping.")
                    continue
                
                base_record = {'material': material, 'delta_T': delta_T, 'voltage': np.nan, 'pressure_Pa': pressure, 'session': session}
                
                # Iterate through holes 2 to 9 (ignoring slits 1 and 10)
                for hole_num in range(2, 10):
                    hole_id = str(hole_num)
                    col_name = f'Hole_{hole_id}'
                    if col_name in rate_series and pd.notna(rate_series[col_name]):
                        airflow_rate = rate_series[col_name]
                        all_records.append({
                            'video_id': video_id, 
                            'hole_id': hole_id, 
                            'airflow_rate': airflow_rate, 
                            **base_record
                        })
                processed_videos_count += 1
            print(f"  - Processed {processed_videos_count} videos from 'multi_hole_csv' structure, generating up to {processed_videos_count * 8} samples.")
            continue # Move to the next dataset in the config loop

        try:
            if gt_file.endswith('.xlsx'):
                gt_df = pd.read_excel(gt_path, header=None, skiprows=1, names=['V_str', 'Pa', 'rate_single'])
                gt_df['V'] = gt_df['V_str'].apply(lambda x: float(re.search(r'(\d+(\.\d+)?)', str(x)).group(1)))
            else: 
                if is_two_holes:
                    gt_df = pd.read_csv(gt_path, sep='\s+', header=None, skiprows=1, names=['V', 'Pa', 'rate_1', 'rate_2'])
                else:
                    gt_df = pd.read_csv(gt_path, sep='\s+', header=None, skiprows=1, names=['V', 'Pa', 'rate_single'])
            
            gt_df['V'] = gt_df['V'].astype(float)
            print(f"  - Successfully loaded ground truth from: {gt_file}")
        except Exception as e:
            print(f"  - ERROR: Could not read GT file '{gt_path}'. Error: {e}. Skipping."); continue

        if structure_type == 'old':
            processed_videos_count = 0
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path):
                    voltage = parse_voltage_from_old_foldername(subfolder_name)
                    if voltage is None: continue
                    
                    gt_row = gt_df[gt_df['V'] == voltage]
                    if gt_row.empty: continue
                    
                    video_files = [f for f in os.listdir(subfolder_path) if f.endswith('.mat') and not f.startswith('._')]
                    for filename in video_files:
                        all_records.append({
                            'video_id': filename.replace('.mat', ''), 'hole_id': '1', 'airflow_rate': gt_row.iloc[0]['rate_single'],
                            'material': material, 'delta_T': parse_delta_T_from_old_filename(filename), 
                            'voltage': voltage, 'pressure_Pa': gt_row.iloc[0]['Pa'],
                            'session': session # <<< CHANGE: ENSURE SESSION IS SAVED >>>
                        })
                        processed_videos_count += 1
            print(f"  - Processed {processed_videos_count} videos from 'old' structure.")

        else: # New structure
            processed_videos_count = 0
            video_files = [f for f in os.listdir(folder_path) if f.endswith('.mat') and not f.startswith('._')]
            for filename in video_files:
                video_id = filename.replace('.mat', '')
                voltage = parse_voltage_from_new_filename(filename)
                delta_T = parse_delta_T_from_new_filename(filename)
                if voltage is None: continue
                
                gt_row = gt_df[gt_df['V'] == voltage]
                if gt_row.empty: continue

                base_record = {'material': material, 'delta_T': delta_T, 'voltage': voltage, 'pressure_Pa': gt_row.iloc[0]['Pa'], 'session': session}
                
                if is_two_holes:
                    all_records.append({'video_id': video_id, 'hole_id': '1_centerhole', 'airflow_rate': gt_row.iloc[0]['rate_1'], **base_record})
                    all_records.append({'video_id': video_id, 'hole_id': '2_cornerhole', 'airflow_rate': gt_row.iloc[0]['rate_2'], **base_record})
                else:
                    all_records.append({'video_id': video_id, 'hole_id': '1', 'airflow_rate': gt_row.iloc[0]['rate_single'], **base_record})
                processed_videos_count += 1
            print(f"  - Processed {processed_videos_count} videos from 'new' structure.")

    if all_records:
        final_df = pd.DataFrame(all_records)
        # <<< CHANGE: SORT BY SESSION FOR CLARITY >>>
        final_df = final_df.sort_values(by=['material', 'session', 'voltage', 'delta_T']).reset_index(drop=True)
        final_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSuccessfully created combined ground truth file at: {OUTPUT_CSV_PATH}")
        print(f"Total records created: {len(final_df)}")
    else:
        print("\nNo records were created.")

if __name__ == "__main__":
    main()


# python scripts/create_ground_truth_labels.py