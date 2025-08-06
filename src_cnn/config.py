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
# e.g., '.../Airflow-rate-prediction/datasets_raw/dataset_gypsum'
RAW_DATASET_PARENT_DIR = os.path.join(PROJECT_ROOT, "datasets")
# Path to the parent directory containing corresponding .npy mask files
RAW_MASK_PARENT_DIR = os.path.join(PROJECT_ROOT, "output_SAM/datasets")
# Path to the top-level directory where processed CNN datasets will be saved.
PROCESSED_DATASET_DIR = os.path.join(PROJECT_ROOT, "CNN_dataset")

# --- Dataset Creation Parameters ---
# These parameters control how the video data is processed into sequences.
ROI_PADDING_PERCENT = 0.20
TRUE_FPS = 5
FOCUS_DURATION_SECONDS = 5 #10
NUM_FRAMES_PER_SAMPLE = 25
IMAGE_TARGET_SIZE = (128, 128)

# Configuration for which raw datasets to process.
# The keys are identifiers, 'dataset_subfolder' is the folder name in RAW_DATASET_PARENT_DIR,
# and 'mask_subfolder' is the folder name in RAW_MASK_PARENT_DIR.
DATASET_CONFIGS = {
    "gypsum_single_hole": {"material": "gypsum", "dataset_subfolder": "dataset_gypsum", "mask_subfolder": "dataset_gypsum"},
    "gypsum_single_hole2": {"material": "gypsum", "dataset_subfolder": "dataset_gypsum2", "mask_subfolder": "dataset_gypsum2"},
    "brick_cladding_single_hole": {"material": "brick_cladding", "dataset_subfolder": "dataset_brickcladding", "mask_subfolder": "dataset_brickcladding"},
    "brick_cladding_two_holes": {"material": "brick_cladding", "dataset_subfolder": "dataset_two_holes_brickcladding", "mask_subfolder": "dataset_two_holes_brickcladding"}
}

# --- Feature Engineering Lists & Flags ---
CONTEXT_FEATURES = [
    'delta_T_log',
    'material_brick_cladding',
    'material_gypsum'
]
DYNAMIC_FEATURES = [
    'hotspot_area_log',
    'hotspot_avg_temp_change_rate_initial_norm',
    'overall_std_deltaT',
    'temp_max_overall_initial',
    'temp_std_avg_initial'
]
HANDCRAFTED_FEATURES_TO_EXTRACT = [
    'hotspot_area',
    'hotspot_avg_temp_change_rate_initial',
    'overall_std_deltaT',
    'temp_max_overall_initial',
    'temp_std_avg_initial',
]
# Flags for feature pre-processing in create_dataset.py
LOG_TRANSFORM_AREA = True
NORMALIZE_AVG_RATE_INITIAL = True

# --- Training Hyperparameters ---
BATCH_SIZE = 8
NUM_EPOCHS_CV = 150
NUM_EPOCHS_FINAL = 150

# --- Optuna-Tuned Hyperparameters for the Best Model ---
OPTUNA_PARAMS = {
    'lr': 0.000242,
    'weight_decay': 0.000289,
    'dropout_rate': 0.309,
    'lstm_hidden_size': 512,
    'lstm_layers': 3,
    'optimizer': 'AdamW'
}