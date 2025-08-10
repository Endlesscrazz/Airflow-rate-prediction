# src_feature_based/utils.py
"""
Contains common utility functions for the feature-based pipeline.
"""
import sys
import os

class Logger(object):
    """
    A simple logger class that redirects stdout and stderr to a log file
    while also printing to the console.
    """
    def __init__(self, filename="output.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
        except (IOError, ValueError):
            pass

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

def setup_logging(output_dir: str, script_name: str):
    """
    Sets up logging to redirect all print statements to a file.

    Args:
        output_dir (str): The main output directory for the current run.
        script_name (str): The name of the script being run (e.g., 'train_cv').
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, f"{script_name}_output.log")
    
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        from datetime import datetime
        f.write(f"--- LOG START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write("="*80 + "\n\n")

    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout 
    
    print(f"--- Logging configured. Output will be saved to: {log_file_path} ---")

def log_experiment_configs(selected_features):
    """
    Prints a formatted summary of the key configuration parameters for the run.
    """
    from src_feature_based import config as cfg
    
    print("\n" + "="*50)
    print("--- EXPERIMENT CONFIGURATION SUMMARY ---")
    print("="*50)
    
    print("\n[Data Parameters]")
    print(f"  - Focus Duration: {cfg.FOCUS_DURATION_SECONDS} seconds")
    print(f"  - True FPS: {cfg.TRUE_FPS}")
    
    print("\n[Modeling Parameters]")
    print(f"  - Target Scaling Enabled: {cfg.ENABLE_TARGET_SCALING}")
    print(f"  - Asymmetric Loss Enabled: {cfg.ENABLE_ASYMMETRIC_LOSS}")
    if cfg.ENABLE_ASYMMETRIC_LOSS:
        print(f"  - Asymmetric Loss Over-prediction Weight: {cfg.ASYMMETRIC_LOSS_OVER_WEIGHT}")
    print(f"  - CV Folds: {cfg.CV_FOLDS}")
    
    print("\n[Selected Features for Training]")
    for i, feature in enumerate(selected_features):
        print(f"  {i+1:02d}. {feature}")
        
    print("="*50 + "\n")