# src_cnn/config.py
"""
Central configuration file for the CNN-based deep learning pipeline.
"""
import os
import torch

# --- Core Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

# --- Path Configuration ---
# This structure assumes the script is run from the project's root directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Path to the parent directory containing raw .mat video files
RAW_DATASET_PARENT_DIR = '/Volumes/One_Touch/Airflow-rate-prediction/datasets'

# FOR CHPC NODE
#RAW_DATASET_PARENT_DIR = os.path.join(PROJECT_ROOT, "datasets")
# Path to the parent directory containing corresponding .npy mask files
RAW_MASK_PARENT_DIR = os.path.join(PROJECT_ROOT, "output_SAM/datasets")
# Path to the top-level directory where processed CNN datasets will be saved.
PROCESSED_DATASET_DIR = os.path.join(PROJECT_ROOT, "CNN_dataset")
GROUND_TRUTH_CSV_PATH = os.path.join(PROJECT_ROOT, "airflow_ground_truth_combined.csv")

# --- Dataset Creation Parameters ---
# These parameters control how the video data is processed into sequences.
ROI_PADDING_PERCENT = 0.20
TRUE_FPS = 5
FOCUS_DURATION_SECONDS =  10 #10
NUM_FRAMES_PER_SAMPLE = 25
IMAGE_TARGET_SIZE = (128, 128)

# --- NEW: Experiment Control Flags ---
# Test 1: Per-fold scaling for tabular features
ENABLE_PER_FOLD_SCALING = True  
SCALER_KIND = "robust"          
SAVE_SCALERS = True             


IMAGE_CHANNELS_BY_TYPE = {
    "thermal": 1,
    "thermal_masked": 2,
    "flow": 2,
    "hybrid": 3
}

# --- NEW: Normalization Constants by Channel Count ---
# Defines mean/std for different input types
NORM_CONSTANTS = {
    1: {"mean": [0.5], "std": [0.5]},
    2: {"mean": [0.5, 0.5], "std": [0.5, 0.5]}, # For Thermal + Mask
    # You can add specific means/stds for flow later if calculated
    # 2: {"mean": [0.0, 0.0], "std": [1.0, 1.0]}, # For Flow
    3: {"mean": [0.5, 0.0, 0.0], "std": [0.5, 1.0, 1.0]} # For Hybrid
}

DATASET_CONFIGS = {
    "gypsum_0716": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07162025_noshutter"},
    "gypsum_0725": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07252025_noshutter"},
    "gypsum_0729": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07292025_noshutter"},
    # "brick_cladding_0616": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0616_2025_noshutter"},
    # "brick_cladding_0805": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0805_2025_noshutter"},
    # "brick_cladding_0808": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0808_2025_noshutter"},
    "hardyboard_0813": {"material": "hardyboard", "dataset_subfolder": "Fluke_HardyBoard_08132025_2holes_noshutter"}
}

# --- Feature Engineering Lists & Flags ---
CONTEXT_FEATURES = [
    'delta_T_log',
    'material_hardyboard',
    'material_gypsum'
]
DYNAMIC_FEATURES = [
    'hotspot_area_log',
    'hotspot_avg_temp_change_rate_initial_norm',
    'overall_std_deltaT',
    'temp_max_overall_initial',
    'temp_std_avg_initial',
    #'temperature_kurtosis',
]
HANDCRAFTED_FEATURES_TO_EXTRACT = [
    'hotspot_area',
    'hotspot_avg_temp_change_rate_initial',
    'overall_std_deltaT',
    'temp_max_overall_initial',
    'temp_std_avg_initial',
    #'temperature_kurtosis',
]
# Flags for feature pre-processing in create_dataset.py
LOG_TRANSFORM_AREA = True
NORMALIZE_AVG_RATE_INITIAL = True

# --- Training Hyperparameters ---
BATCH_SIZE = 8
NUM_EPOCHS_CV = 150
NUM_EPOCHS_FINAL = 150
CV_FOLDS = 5

# --- Optuna-Tuned Hyperparameters for the Best Model ---
OPTUNA_PARAMS = {
    'lr': 0.0004021134836365209,
    'weight_decay': 0.00011195074140390493,
    'dropout_rate': 0.4313499563157644,
    'lstm_hidden_size': 256,
    'lstm_layers': 2,
    'optimizer': 'AdamW'
}
