# config.py
"""Configuration settings for the airflow prediction project."""

import os

# --- Paths ---
# Assumes 'dataset_new' is in the same directory as the scripts
DATASET_FOLDER = os.path.join(os.getcwd(), "dataset_new")
MODEL_SAVE_PATH = "final_airflow_predictor_pipeline.joblib"
PLOT_SAVE_PATH = "confusion_matrix_cnn.png"

MAT_FRAMES_KEY = 'TempFrames'

# --- CNN Feature Extraction ---
# Input size expected by the pre-trained model (e.g., MobileNetV2 default)
CNN_INPUT_SIZE = (224, 224)

# --- Model Parameters ---
RANDOM_STATE = 42  # For reproducibility
PCA_N_COMPONENTS = None #0.95 # Retain 95% variance, set to None to disable PCA
N_ESTIMATORS = 100 # For RandomForest and GradientBoosting

# --- Cross-Validation ---
CV_METHOD = 'LeaveOneOut' # Options: 'LeaveOneOut', 'StratifiedKFold'
#K_FOLDS = 5 # Used only if CV_METHOD is 'StratifiedKFold'

# --- Evaluation ---
# Metric to choose the best model for final training and plotting
BEST_MODEL_METRIC = 'f1_weighted' # Options: 'accuracy', 'f1_weighted'