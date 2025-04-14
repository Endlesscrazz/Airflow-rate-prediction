# config.py
"""Configuration settings for the airflow prediction project."""

import os
import numpy as np # Needed for np.inf

# --- Paths ---
# Get the directory where THIS config file lives (e.g., /path/to/project/src)
_config_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the project root, e.g., /path/to/project)
_project_root = os.path.dirname(_config_dir)

# Define DATASET_FOLDER relative to the project root
DATASET_FOLDER = os.path.join(_project_root, "dataset_new")

# Define other paths relative to project root or config dir as appropriate
MODEL_SAVE_PATH = os.path.join(_project_root, "final_airflow_regressor.joblib")
PLOT_SAVE_PATH = os.path.join(_project_root, "actual_vs_predicted_regression.png")

# --- Other Settings ---
MAT_FRAMES_KEY = 'TempFrames'

# --- CNN Feature Extraction (No longer used by default) ---
CNN_INPUT_SIZE = (224, 224)

# --- Model Parameters ---
RANDOM_STATE = 42
# --- PCA DISABLED ---
PCA_N_COMPONENTS = None
# --- Default n_estimators for tree models ---
N_ESTIMATORS = 100

# --- Feature Engineering Parameters ---
BRIGHT_REGION_THRESHOLD = 0.8  # Factor of max intensity
VIDEO_FPS = 30                 # Assumed frames per second
GRADIENT_INTERVAL_SEC = 5      # Duration of each gradient window (seconds)
MAX_GRADIENT_INTERVALS = 3     # Calculate features for first N intervals (e.g., 3 -> 0-5s, 5-10s, 10-15s)


# --- Cross-Validation ---
CV_METHOD = 'LeaveOneOut' # Options: 'LeaveOneOut', 'KFold'
#K_FOLDS = 5 # Used only if CV_METHOD is 'KFold'

# --- Evaluation ---
BEST_MODEL_METRIC_SCORE = 'neg_mean_squared_error'
METRIC_COMPARISON_HIGHER_IS_BETTER = True
