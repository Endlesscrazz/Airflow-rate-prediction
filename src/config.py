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
PLOT_SAVE_PATH = os.path.join(_project_root, "output") # Optional path for plots

# --- Data Loading ---
MAT_FRAMES_KEY = 'TempFrames' # Key in the .mat file holding the frame data

# --- ADD PREPROCESSING CONFIGURATION ---
PREPROCESSING_APPLY_FILTER = False          # Set to True to apply median filter, False otherwise
PREPROCESSING_FILTER_KERNEL_SIZE = 3       # Kernel size for median filter (must be odd integer >= 3)
PREPROCESSING_APPLY_SCALING = False         # Set to True to apply min-max scaling, False otherwise
PREPROCESSING_SCALE_RANGE = (0, 1)         # Target range for min-max scaling, e.g., (0, 1) or (-1, 1)

# --- Model Parameters ---
RANDOM_STATE = 42
# --- PCA DISABLED ---
PCA_N_COMPONENTS = None

N_ESTIMATORS = 100                  # Used potentially by RandomForest, GradientBoosting

# --- Feature Engineering Parameters ---
BRIGHT_REGION_QUANTILE = 0.95  # Use 95th percentile as threshold (Can Adjust between ~0.85 to 0.99 if needed)
FRAME_INTERVAL_SIZE = 50       # Keep for gradients
MAX_FRAME_INTERVALS = 3         # Keep for gradients
CALCULATE_STD_FEATURES = True


# --- Interval settings can be REMOVED/COMMENTED OUT to process the whole video ---
# GRADIENT_INTERVAL_SEC = 5      # Duration of each gradient window (seconds)
# MAX_GRADIENT_INTERVALS = 3     # Calculate features for first N intervals (e.g., 3 -> 0-5s, 5-10s, 10-15s)
# FRAME_INTERVAL_SIZE = 50       # Number of frames per interval
# MAX_FRAME_INTERVALS = 3      # Max number of frame intervals to process

# --- Cross-Validation ---
CV_METHOD = 'LeaveOneOut' # Options: 'LeaveOneOut', 'KFold'
K_FOLDS = 5 # Used only if CV_METHOD is 'KFold' (Make sure to set CV_METHOD='KFold' to use this)

# --- Evaluation ---
BEST_MODEL_METRIC_SCORE = 'neg_mean_squared_error' # Metric used to select the best model during tuning/evaluation
METRIC_COMPARISON_HIGHER_IS_BETTER = True # Is a higher score better for BEST_MODEL_METRIC_SCORE? (True for R2, neg_mse; False for MSE, MAE, RMSE)

# --- Visualization Settings ---                        
RUN_VISUALIZATIONS = True                           
# Define visualization save directory relative to project root
VIS_SAVE_DIR = os.path.join(_project_root, "plots") 