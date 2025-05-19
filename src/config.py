# config.py
"""Configuration settings for the airflow prediction project."""

import os
import numpy as np

# --- Paths ---
_config_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_config_dir)
DATASET_FOLDER = os.path.join(_project_root, "dataset_cleaned")

# --- Central Output and Model Directories ---
OUTPUT_DIR = os.path.join(_project_root, "output")
TRAINED_MODELS_DIR = os.path.join(_project_root, "trained_models")
BEST_MODELS_DIR = os.path.join(TRAINED_MODELS_DIR, "best_models") 
FEATURES_SAVE_DIR = os.path.join(OUTPUT_DIR, "saved_features")

# Path templates for saving models
# For individual tuned models (e.g., trained_models/SVR_tuned_model.joblib)
INDIVIDUAL_MODEL_SAVE_PATH_TEMPLATE = os.path.join(TRAINED_MODELS_DIR, "{model_name}_tuned_model.joblib")
# For the single overall best model (e.g., trained_models/best_models/BEST_SVR_final_model.joblib)
OVERALL_BEST_MODEL_SAVE_PATH_TEMPLATE = os.path.join(BEST_MODELS_DIR, "BEST_{model_name}_final_model.joblib")


# --- Data Loading ---
MAT_FRAMES_KEY = 'TempFrames'

# --- Data Preprocessing ---
PREPROCESSING_APPLY_FILTER = False
PREPROCESSING_FILTER_KERNEL_SIZE = 3
PREPROCESSING_APPLY_SCALING = False
PREPROCESSING_SCALE_RANGE = (0, 1)

# --- Model Training Parameters ---
RANDOM_STATE = 42
PCA_N_COMPONENTS = None
EVALUATE_ON_FULL_TRAINING_SET = True
SAVE_PERMUTATION_IMPORTANCE_PLOT = True 

# --- Feature Engineering Parameters ---
# THRESHOLD_ABS_CHANGE_FOR_AREA = 0.5 # Example

# --- Cross-Validation ---
CV_METHOD = 'LeaveOneOut'   # Options: 'LeaveOneOut', 'KFold'
K_FOLDS = 3                 # Used if CV_METHOD is 'KFold'

# --- Evaluation ---
BEST_MODEL_METRIC_SCORE = 'neg_mean_squared_error' # Metric for GridSearchCV
# METRIC_COMPARISON_HIGHER_IS_BETTER = True # Not directly used by GridSearchCV's default behavior

# --- Visualization Settings ---
SAVE_ACTUAL_VS_PREDICTED_PLOT = True
SAVE_LOSS_CURVE_PLOT = True

# --- Hotspot Mask Generation Parameters (Defaults) ---
MASK_DIR = os.path.join(_project_root, "hotspot_masks", "slope_p01_focus5s_q99_env1_roi0") 
MASK_OUTPUT_DIR_DEFAULT_SUBFOLDER = "hotspot_masks_generated" 

MASK_FPS = 5.0
MASK_FOCUS_DURATION_SEC = 5.0
MASK_SMOOTH_WINDOW = 1
MASK_P_VALUE_THRESHOLD = 0.10
MASK_ENVIR_PARA = 1
MASK_AUGMENT_SLOPE = 1.0
MASK_NORMALIZE_TEMP_FRAMES = False
MASK_FUSE_LEVEL = 0
MASK_ROI_BORDER_PERCENT = 0.0
MASK_ACTIVITY_QUANTILE = 0.99
MASK_MORPHOLOGY_OP = 'none'
MASK_APPLY_BLUR_TO_ACTIVITY_MAP = False
MASK_BLUR_KERNEL_SIZE = (3, 3)
# --- END OF MASK GENERATION PARAMETERS ---