# src_cnn/data_utils.py
"""
Contains utility functions for parsing metadata and loading ground truth data.
"""
import pandas as pd
import re

def load_airflow_from_csv(filepath: str) -> dict:
    """
    Loads the ground truth airflow CSV and creates a mapping from
    FanVoltage (V) to the airflow_rate in L/min.
    """
    try:
        df = pd.read_csv(filepath)
        if 'V' not in df.columns or 'L/min' not in df.columns:
            raise ValueError("CSV file must contain 'V' and 'L/min' columns.")
        return pd.Series(df['L/min'].values, index=df['V']).to_dict()
    except FileNotFoundError:
        print(f"FATAL ERROR: Ground truth CSV file not found at '{filepath}'")
        raise
    except Exception as e:
        print(f"FATAL ERROR: Could not process ground truth CSV file. Error: {e}")
        raise

def parse_voltage_from_filename(filename: str) -> float:
    """
    Extracts the fan voltage (e.g., 1.4) from a filename 
    (e.g., 'T1.4V_...'). This is used to link a video to the ground truth.
    """
    match = re.search(r'T(\d+(\.\d+)?)V', filename)
    if match:
        return float(match.group(1))
    return None

def parse_delta_T(filename: str) -> float:
    """Extracts delta T from a filename (e.g., '..._22_30_8_.mat' -> 8.0)."""
    parts = filename.replace('.mat', '').split('_')
    try:
        # Assumes the format is ..._T_amb_T_surf_deltaT_
        if len(parts) > 3 and parts[-2].replace('.', '', 1).isdigit():
            return float(parts[-2])
    except (ValueError, IndexError):
        pass
    return None