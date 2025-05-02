# --- START OF FILE config.py ---

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

# Define output paths relative to project root
# --- Renamed for clarity ---
MODEL_SAVE_PATH = os.path.join(_project_root, "trained_models", " final_airflow_nn_regressor.joblib")
# --- Renamed for clarity ---
ACTUAL_VS_PREDICTED_PLOT_SAVE_PATH = os.path.join(_project_root, "output", "actual_vs_predicted_regression.png")
# --- Define specific directory for focus area images ---
FOCUS_AREA_VIS_SAVE_DIR = os.path.join(_project_root, "output", "focused_images")
# --- Define directory for loss curve plot ---
LOSS_CURVE_PLOT_SAVE_DIR = os.path.join(_project_root, "output") # Save loss curve in main output dir

# --- Data Loading ---
MAT_FRAMES_KEY = 'TempFrames' # Key in the .mat file holding the frame data

# --- Data Preprocessing (Applied before feature extraction if True) ---
# These flags would need to be checked and implemented in data_utils.load_raw_data or main.py
PREPROCESSING_APPLY_FILTER = False          # Set to True to apply median filter to frames
PREPROCESSING_FILTER_KERNEL_SIZE = 3       # Kernel size for median filter (must be odd integer >= 3)
PREPROCESSING_APPLY_SCALING = False         # Set to True to apply min-max scaling to frames (0-1 or -1 to 1)
PREPROCESSING_SCALE_RANGE = (0, 1)         # Target range for min-max scaling

# --- Model Training Parameters ---
RANDOM_STATE = 42
# --- PCA is disabled in the current pipeline ---
PCA_N_COMPONENTS = None
# N_ESTIMATORS = 100 # Default estimators for Tree models (Not used currently)

# --- Feature Engineering Parameters ---
GRADIENT_MAP_QUANTILE = 0.98  
BRIGHT_REGION_QUANTILE = 0.95  # Percentile for bright region threshold (e.g., 0.95 means top 5%)
FRAME_INTERVAL_SIZE = 50       # Number of frames per interval for grad/std features
MAX_FRAME_INTERVALS = 3        # Max number of intervals (e.g., 3 -> 0-50, 50-100, 100-150)
CALCULATE_STD_FEATURES = True  # Set to True to calculate std dev features, False for only grad/area

# --- Cross-Validation ---
CV_METHOD = 'LeaveOneOut' # Options: 'LeaveOneOut', 'KFold'
K_FOLDS = 5 # Used only if CV_METHOD is 'KFold'

# --- Evaluation ---
BEST_MODEL_METRIC_SCORE = 'neg_mean_squared_error' # Metric used to select the best model
METRIC_COMPARISON_HIGHER_IS_BETTER = True # Is a higher score better for BEST_MODEL_METRIC_SCORE?

# --- Visualization Settings ---
# --- Renamed for clarity ---
SAVE_FOCUS_AREA_VISUALIZATION = True # Set to True to save focus area images
NUM_SAMPLES_TO_VISUALIZE = 2       # How many samples to save images for
SAVE_LOSS_CURVE_PLOT = True        # Set to True to save the training loss curve plot
SAVE_ACTUAL_VS_PREDICTED_PLOT = True # Set to True to save the actual vs predicted plot

# --- END OF FILE config.py ---