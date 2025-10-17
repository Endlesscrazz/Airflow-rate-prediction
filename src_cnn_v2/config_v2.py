# src_cnn_v2/config_v2.py
"""
Central configuration file for the V2 (bottom-up, cropped) deep learning pipeline.
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

# --- V2 UNIFIED OUTPUT DIRECTORY ---
# This is the parent folder for all outputs related to this new approach.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "CNN_dataset/gypusm_all_dataset_v2")

# --- PATHS DERIVED FROM OUTPUT_DIR ---
# Master features file is still needed to get metadata like delta_T and labels.
MASTER_METADATA_PATH = os.path.join(OUTPUT_DIR, "master_metadata_v2.csv")

# Ground truth path remains independent, used to create the master features file.
GROUND_TRUTH_CSV_PATH = os.path.join(PROJECT_ROOT, "airflow_ground_truth_gypsum_all.csv")

# --- V2 Dataset Creation Parameters ---
TRUE_FPS = 5
FOCUS_DURATION_SECONDS = 15
NUM_FRAMES_PER_SAMPLE = 25

# --- NEW: Cropping and Augmentation Config ---
V2_DATASET_PARAMS = {
    "CROP_SIZE": 32,  # Pixel dimension of the square crop (e.g., 32x32)
    "NUM_AUGMENTATIONS": 10,  # Number of noisy copies to create PER training sample
    "NOISE_LEVEL": 0.05,  # Standard deviation of Gaussian noise to add
    "OUTPUT_SUBDIR": "dataset_1ch_cropped_augmented"
}

# --- V2 Model Input Config ---
# For this approach, we only use delta_T as a non-image feature.
CONTEXT_FEATURES = [
    'delta_T', # We can log-transform this inside the training script if needed
]
ALL_POSSIBLE_MATERIALS = [
    'gypsum',
]

# --- V2 Training Hyperparameters ---
BATCH_SIZE = 16 # Can likely be increased due to smaller data size
NUM_EPOCHS_CV = 100
CV_FOLDS = 5
ENABLE_PER_FOLD_SCALING = True
SCALER_KIND = "robust"
SAVE_SCALERS = True

# Normalization constants for the single-channel thermal crops
NORM_CONSTANTS = {
    1: {"mean": [0.5], "std": [0.5]},
}

# Simplified initial hyperparameters. These should be re-tuned later.
INITIAL_PARAMS = {
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'dropout_rate': 0.4,
    'lstm_hidden_size': 128,
    'lstm_layers': 2,
    'optimizer': 'AdamW'
}

# --- Shared Config with Original Pipeline ---
# This dictionary is needed by the data creation script to find the raw files.
DATASET_CONFIGS = {
    "gypsum_0716": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07162025_noshutter"},
    "gypsum_0725": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07252025_noshutter"},
    "gypsum_0729": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07292025_noshutter"},

    "gypsum_0307": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_03072025"},  # OLD
    #"gypsum_0903": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_09032025_10holes_noshutter"},
    # "brick_cladding_0616": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0616_2025_noshutter"},
    # "brick_cladding_0805": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0805_2025_noshutter"},
    # "brick_cladding_0808": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0808_2025_noshutter"},
    # "hardyboard_0813": {"material": "hardyboard", "dataset_subfolder": "Fluke_HardyBoard_08132025_2holes_noshutter"},
    # "hardyboard_0313": {"material": "hardyboard", "dataset_subfolder": "Fluke_HardyBoard_03132025"}, # OLD
}