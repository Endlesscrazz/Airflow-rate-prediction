# config.py
"""Configuration settings for the airflow prediction project."""

import os
import numpy as np

# --- Paths ---
_config_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_config_dir)
DATASET_FOLDER = os.path.join(_project_root, "dataset_cleaned") # Or your default dataset
MODEL_SAVE_PATH = os.path.join(_project_root, "trained_models", "final_airflow_nn_regressor.joblib") # Corrected typo
ACTUAL_VS_PREDICTED_PLOT_SAVE_PATH = os.path.join(_project_root, "output", "actual_vs_predicted_regression.png")
FOCUS_AREA_VIS_SAVE_DIR = os.path.join(_project_root, "output", "focused_images")
LOSS_CURVE_PLOT_SAVE_DIR = os.path.join(_project_root, "output")
OUTPUT_DIR = os.path.join(_project_root, "output")
FEATURES_SAVE_DIR = os.path.join(_project_root, "output/saved_features")


# --- Data Loading ---
MAT_FRAMES_KEY = 'TempFrames'

# --- Data Preprocessing (Applied before feature extraction if True) ---
PREPROCESSING_APPLY_FILTER = False
PREPROCESSING_FILTER_KERNEL_SIZE = 3
PREPROCESSING_APPLY_SCALING = False
PREPROCESSING_SCALE_RANGE = (0, 1)

# --- Model Training Parameters ---
RANDOM_STATE = 42
PCA_N_COMPONENTS = None # Assuming this means PCA is off by default for the main model
EVALUATE_ON_FULL_TRAINING_SET = True

# --- Feature Engineering Parameters ---
# ... (your existing feature engineering flags) ...
GRADIENT_MAP_QUANTILE = 0.98
BRIGHT_REGION_QUANTILE = 0.95
FRAME_INTERVAL_SIZE = 50
MAX_FRAME_INTERVALS = 3
CALCULATE_STD_FEATURES = True

# --- Cross-Validation ---
CV_METHOD = 'LeaveOneOut'   #KFold
K_FOLDS = 3

# --- Evaluation ---
BEST_MODEL_METRIC_SCORE = 'neg_mean_squared_error'
METRIC_COMPARISON_HIGHER_IS_BETTER = True

# --- Visualization Settings ---
SAVE_FOCUS_AREA_VISUALIZATION = True
NUM_SAMPLES_TO_VISUALIZE = 2
SAVE_LOSS_CURVE_PLOT = True
SAVE_ACTUAL_VS_PREDICTED_PLOT = True

# -----------------------------------------------------------------------------
# --- NEW: Hotspot Mask Generation Parameters (Defaults) ---
# -----------------------------------------------------------------------------
MASK_OUTPUT_DIR_DEFAULT_SUBFOLDER = "hotspot_masks/slope_p01_focus5s_q99_env1" # Subfolder within project root for masks

# Parameters for slope calculation
MASK_FPS = 5.0
MASK_FOCUS_DURATION_SEC = 5.0  # Duration (s) for slope analysis
MASK_SMOOTH_WINDOW = 1         # Temporal smoothing window for slope fit (1=None)
MASK_P_VALUE_THRESHOLD = 0.10  # P-value threshold for slope significance filter
MASK_ENVIR_PARA = 1            # Environment parameter (-1=Winter/cooling, 1=Summer/heating expected at leak)
MASK_AUGMENT_SLOPE = 1.0       # Exponent for augmenting filtered slope magnitude

# Parameters for frame pre-processing before slope calculation
MASK_NORMALIZE_TEMP_FRAMES = False # Normalize temperature per frame? (True/False)
MASK_FUSE_LEVEL = 0            # Spatial fuse level (0=None)

# Parameters for ROI
MASK_ROI_BORDER_PERCENT = 0.0  # Percent border (0.0 to <0.5) to exclude. 0.0 means no ROI by default.

# Parameters for extracting hotspot from activity map
MASK_ACTIVITY_QUANTILE = 0.99  # Quantile threshold for activity map (0-1)
MASK_MORPHOLOGY_OP = 'none'    # Morphological op: 'close', 'open_close', 'none'
MASK_APPLY_BLUR_TO_ACTIVITY_MAP = False # Apply Gaussian blur to activity map?
MASK_BLUR_KERNEL_SIZE = (3, 3) # Kernel size (H, W) for Gaussian blur if applied
# --- END OF NEW MASK GENERATION PARAMETERS ---
# -----------------------------------------------------------------------------