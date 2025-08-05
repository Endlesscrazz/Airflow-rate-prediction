# config.py
"""Configuration settings for the airflow prediction project."""

import os
import numpy as np

# --- PATH CONFIGURATION ---
# Check if a special environment variable is set (for CHPC).
# If not, fall back to the local path.
CHPC_SCRATCH_DIR = os.environ.get('CHPC_SCRATCH_DIR')

if CHPC_SCRATCH_DIR:
    # We are on the CHPC cluster
    print(">>> CHPC environment detected. Using scratch paths. <<<")
    # This is the parent directory of 'dataset_gypsum', 'dataset_brickcladding', etc.
    DATASET_PARENT_DIR = os.path.join(CHPC_SCRATCH_DIR, 'datasets')
    BASE_MASK_INPUT_DIR = os.path.join(CHPC_SCRATCH_DIR, 'output_SAM/datasets') # Assuming masks are also on scratch
else:
    # We are on a local machine
    # Define the relative path from the project root.
    # This assumes your local 'datasets' folder is at the project root.
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATASET_PARENT_DIR = os.path.join(PROJECT_ROOT, 'datasets')
    BASE_MASK_INPUT_DIR = os.path.join(PROJECT_ROOT, 'output_SAM/datasets')

# --- Project Root Path ---
_config_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_config_dir)

# --- Base Directories (Material-Agnostic) ---
BASE_OUTPUT_DIR = os.path.join(_project_root, "output")
BASE_TRAINED_MODELS_DIR = os.path.join(_project_root, "trained_models")

# --- Parent Directory for Datasets ---
DATASET_PARENT_DIR = os.path.join(_project_root, "datasets")

# --- Parent Directory for Precomputed Masks ---
BASE_MASK_INPUT_DIR = os.path.join(_project_root, "output_SAM/datasets")

# --- Data Loading ---
MAT_FRAMES_KEY = 'TempFrames'

## -- Mask Verfication
DATASET_FOLDER = os.path.join(_project_root,"datasets/dataset_gypsum")
# --- Model Training & Evaluation Parameters ---
RANDOM_STATE = 42
PCA_N_COMPONENTS = None
EVALUATE_ON_FULL_TRAINING_SET = True # This flag is used by temp_main_nested_cv.py to show train scores
SAVE_PERMUTATION_IMPORTANCE_PLOT = True # If you want to run perm importance on final dev model

# --- Cross-Validation Method for Final Tuning and Learning Curve ---
# (Nested CV itself will define its own outer/inner folds)
CV_METHOD_FOR_FINAL_TUNING = 'KFold' # e.g., KFold for the GridSearchCV on full dev set
K_FOLDS_FOR_FINAL_TUNING = 5         # e.g., 5 folds

# --- Nested CV Parameters ---
NESTED_CV_N_OUTER_FOLDS = 5
NESTED_CV_N_REPEATS = 3 # Only if using RepeatedKFold, not used with simple KFold for outer
NESTED_CV_N_INNER_FOLDS = 3

# --- Learning Curve Parameters ---
LEARNING_CURVE_TRAIN_SIZES = np.linspace(0.2, 1.0, 7).tolist() # Percentages of training set
LEARNING_CURVE_CV_FOLDS = 5 # Folds for CV within learning_curve function

# --- Visualization Settings ---
SAVE_ACTUAL_VS_PREDICTED_PLOT = True
SAVE_LOSS_CURVE_PLOT = True

# --- PREPROCESSING FLAGS FOR MASK GENERATION (used by hotspot_mask_generation.py) ---
MASK_SPATIAL_BLUR_KSIZE = 3       # Kernel size for initial spatial blur (e.g., 3 for 3x3). Set to 0 or 1 to disable.
MASK_SUBTRACT_PIXEL_MEAN_FOR_SLOPE = True # Subtract per-pixel mean from focus window before slope calc?

# --- Default Parameters for Hotspot Mask Generation ---
MASK_FPS = 5.0
MASK_FOCUS_DURATION_SEC = 5.0
MASK_SMOOTH_WINDOW = 1
MASK_P_VALUE_THRESHOLD = 0.10
MASK_ENVIR_PARA = 1
MASK_AUGMENT_SLOPE = 1.0
MASK_NORMALIZE_TEMP_FRAMES = False
MASK_FUSE_LEVEL = 0
MASK_ROI_BORDER_PERCENT = 0.10
MASK_ACTIVITY_QUANTILE = 0.99
MASK_MORPHOLOGY_OP = 'open_close'
MASK_APPLY_BLUR_TO_ACTIVITY_MAP = False
MASK_BLUR_KERNEL_SIZE = (3, 3)

# --- Feature Engineering Parameters ---
FIXED_AREA_THRESHOLD = 0.5
# Transformation flags can be defined here or directly in main/temp_main
LOG_TRANSFORM_DELTA_T = True
LOG_TRANSFORM_AREA = True
NORMALIZE_AVG_RATE_INITIAL = True
# NORMALIZE_AVG_MAG_INITIAL = True # Add if used

