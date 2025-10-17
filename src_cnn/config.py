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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATASET_PARENT_DIR = '/Volumes/One_Touch/Airflow-rate-prediction/datasets'
RAW_MASK_PARENT_DIR = os.path.join(PROJECT_ROOT, "output_SAM/datasets")

# --- UNIFIED OUTPUT DIRECTORY FOR THE EXPERIMENT ---
# Manually change this to group related experiments (e.g., for a specific material).
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "CNN_dataset/gypusm_8_hole_dataset")

# --- PATHS DERIVED FROM OUTPUT_DIR ---
# The master features file will live at the top level of the experiment directory.
MASTER_FEATURES_PATH = os.path.join(OUTPUT_DIR, "master_features.csv")

# Ground truth path remains independent.
GROUND_TRUTH_CSV_PATH = os.path.join(PROJECT_ROOT, "airflow_ground_truth_gypsum_8_hole.csv")

# --- Dataset Creation Parameters ---
# These parameters control how the video data is processed into sequences.
ROI_PADDING_PERCENT = 0.20
TRUE_FPS = 5
FOCUS_DURATION_SECONDS =  15 #10
NUM_FRAMES_PER_SAMPLE = 25
IMAGE_TARGET_SIZE = (128, 128)

# Flags to normalize features
LOG_TRANSFORM_AREA = True
NORMALIZE_AVG_RATE_INITIAL = True
NORMALIZE_CUMULATIVE_FEATURES = True

# --- NEW: Experiment Control Flags ---
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

ALL_FEATURES_TO_CALCULATE = [
    'hotspot_area', 'hotspot_avg_temp_change_rate_initial', 'hotspot_avg_temp_change_magnitude_initial',
    'peak_pixel_temp_change_rate_initial', 'peak_pixel_temp_change_magnitude_initial', 'temp_mean_avg_initial',
    'temp_std_avg_initial', 'temp_min_overall_initial', 'temp_max_overall_initial',
    'stabilized_mean_deltaT', 'overall_mean_deltaT', 'max_abs_mean_deltaT',
    'stabilized_std_deltaT', 'overall_std_deltaT', 
    # 'mean_area_significant_change','stabilized_area_significant_change', 'max_area_significant_change',
    'hotspot_solidity', 'time_to_peak_mean_temp',
    'temperature_skewness', 'temperature_kurtosis', 
    # 'rate_of_std_change_initial', 'peak_to_average_ratio',
    # 'radial_profile_0', 'radial_profile_1', 'radial_profile_2', 'radial_profile_3', 'radial_profile_4',
    'bbox_area', 'bbox_aspect_ratio','perimeter',
    'circularity','mean_gradient_at_edge',
    # 'temp_p25','temp_p75','temp_iqr'
    # Feature discussed in meeting
    'cumulative_raw_delta_sum','cumulative_abs_delta_sum',
    # Own idea feature
    'auc_mean_temp_delta',
    'mean_pixel_volatility',
]

DATASET_CONFIGS = {
    # "gypsum_0716": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07162025_noshutter"},
    # "gypsum_0725": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07252025_noshutter"},
    # "gypsum_0729": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07292025_noshutter"},

    # "gypsum_0307": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_03072025"},  # OLD
    "gypsum_0903": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_09032025_10holes_noshutter"},
    # "brick_cladding_0616": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0616_2025_noshutter"},
    # "brick_cladding_0805": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0805_2025_noshutter"},
    # "brick_cladding_0808": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0808_2025_noshutter"},
    # "hardyboard_0813": {"material": "hardyboard", "dataset_subfolder": "Fluke_HardyBoard_08132025_2holes_noshutter"},
    # "hardyboard_0313": {"material": "hardyboard", "dataset_subfolder": "Fluke_HardyBoard_03132025"}, # OLD
}

# --- Feature Engineering Lists & Flags ---
CONTEXT_FEATURES = [
    'delta_T_log',
]
# All possible material categories used for one-hot encoding
ALL_POSSIBLE_MATERIALS = [
    'gypsum',
    # 'brick_cladding',
    # 'hardyboard',
]

DYNAMIC_FEATURES = [
    'hotspot_avg_temp_change_rate_initial_norm',
    'cumulative_raw_delta_sum_norm',
    'cumulative_abs_delta_sum_norm',
    'auc_mean_temp_delta_norm',
    'hotspot_avg_temp_change_magnitude_initial',
    'peak_pixel_temp_change_magnitude_initial',
    'temp_mean_avg_initial',
    'temp_std_avg_initial',
    'temp_max_overall_initial',
    'stabilized_mean_deltaT',
    'overall_mean_deltaT',
    'max_abs_mean_deltaT',
    'stabilized_std_deltaT',
    'overall_std_deltaT',
    'mean_gradient_at_edge',
]


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
