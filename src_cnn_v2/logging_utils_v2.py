# src_cnn_v2/logging_utils_v2.py
"""
Contains utility functions for logging experiment parameters and results.
"""
import os
import datetime
import json

def log_experiment_details(filepath, title, params_dict):
    """
    Appends a formatted block of parameters to a specified log file.

    Args:
        filepath (str): Path to the log file (e.g., 'experiment_summary.txt').
        title (str): The title for this block of parameters (e.g., "Data Creation Parameters").
        params_dict (dict): A dictionary of parameters to log.
    """
    # Use 'a' to append to the file if it exists, or create it if it doesn't.
    with open(filepath, 'a') as f:
        f.write(f"--- {title.upper()} ---\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for key, value in params_dict.items():
            # Nicely format lists or dictionaries
            if isinstance(value, (list, dict)):
                f.write(f"{key}:\n")
                # Use json.dumps for pretty printing nested structures
                f.write(json.dumps(value, indent=4))
                f.write("\n\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*50 + "\n\n")

    print(f"Successfully logged '{title}' to {os.path.basename(filepath)}")