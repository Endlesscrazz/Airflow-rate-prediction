# src_cnn_v3/logging_utils_v3.py
import os
import json
import datetime
from src_cnn_v3 import config_v3 as cfg

def get_summary_path():
    """Ensures the result directory exists and returns the log path."""
    os.makedirs(cfg.EXPERIMENT_RESULTS_DIR, exist_ok=True)
    return os.path.join(cfg.EXPERIMENT_RESULTS_DIR, "experiment_summary.txt")

def log_section(title, data_dict):
    """
    Appends a formatted section to the experiment_summary.txt file.
    """
    log_path = get_summary_path()
    
    with open(log_path, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"STEP: {title.upper()}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'-'*60}\n")
        
        for key, value in data_dict.items():
            if isinstance(value, (dict, list)):
                f.write(f"{key}:\n")
                f.write(json.dumps(value, indent=4))
                f.write("\n")
            else:
                f.write(f"{key:<30}: {value}\n")
        f.write("\n")

    print(f"  [Log] Experiment summary updated: {os.path.basename(log_path)}")