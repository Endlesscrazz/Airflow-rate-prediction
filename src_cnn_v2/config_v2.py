# src_cnn_v2/config_v2.py
"""
Central configuration file for the V2 (bottom-up, cropped) deep learning pipeline.
Organized for clean experiment management with versioning.
"""
import os
import torch

# --- Core Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

# --- 1. SET THE EXPERIMENT NAME ---
EXPERIMENT_NAME = "hardyboard_all_dataset_v2"

# --- 2. SET THE EXPERIMENT VERSION ---
# Change this for each new run to create a unique results folder.
EXPERIMENT_VERSION = "iter-13-cs15-hp-tuned-overpred-1.7-rs42"

# --- 3. SET DATA CREATION PARAMETERS ---
# These parameters define the dataset that will be generated.
# Changing these will result in a new data folder being created.
CROP_SIZE = 15
NUM_AUGMENTATIONS = 100 
TRUE_FPS = 5
FOCUS_DURATION_SECONDS = 30 
NUM_FRAMES_PER_SAMPLE = 150


# --- Path Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "CNN_dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Output_CNN-LSTM")

# === CHPC Paths ===
RAW_DATASET_PARENT_DIR = '/scratch/general/vast/u1527145/datasets'
# This path should point to the output of find_leaking_holes.py
RAW_MASK_PARENT_DIR = '/scratch/general/vast/u1527145/Airflow-rate-prediction/Output_SAM/datasets'


# --- DERIVED PATHS (Automatically constructed) ---
OUTPUT_DIR = os.path.join(DATA_DIR, EXPERIMENT_NAME)
EXPERIMENT_RESULTS_DIR = os.path.join(RESULTS_DIR, EXPERIMENT_NAME, EXPERIMENT_VERSION)
MASTER_METADATA_PATH = os.path.join(OUTPUT_DIR, "master_metadata_v2.csv")
GROUND_TRUTH_CSV_PATH = os.path.join(PROJECT_ROOT, "airflow_ground_truth_hardyboard_all.csv")

# --- Versioned paths for SPLIT files ---
TRAIN_SPLIT_PATH = os.path.join(OUTPUT_DIR, f"train_split_seed{RANDOM_STATE}.csv")
VAL_SPLIT_PATH = os.path.join(OUTPUT_DIR, f"val_split_seed{RANDOM_STATE}.csv")
TEST_SPLIT_PATH = os.path.join(OUTPUT_DIR, f"test_split_seed{RANDOM_STATE}.csv")

# --- Versioned paths for FINAL metadata files ---
# Includes NUM_FRAMES_PER_SAMPLE (nf) for unique folder names per experiment
DATASET_SUBDIR = f"dataset_cs{CROP_SIZE}_nf{NUM_FRAMES_PER_SAMPLE}_aug{NUM_AUGMENTATIONS}"
DATASET_DIR = os.path.join(OUTPUT_DIR, DATASET_SUBDIR)
TRAIN_METADATA_PATH = os.path.join(DATASET_DIR, f"train_metadata_seed{RANDOM_STATE}.csv")
VAL_METADATA_PATH = os.path.join(DATASET_DIR, f"val_metadata_seed{RANDOM_STATE}.csv")
TEST_METADATA_PATH = os.path.join(DATASET_DIR, f"test_metadata_seed{RANDOM_STATE}.csv")


# --- V2 Dataset Creation Parameters ---
V2_DATASET_PARAMS = {
    "OUTPUT_SUBDIR": DATASET_SUBDIR,
    "CROP_SIZE": CROP_SIZE,
    "NUM_AUGMENTATIONS": NUM_AUGMENTATIONS,
    "ENABLE_GEOMETRIC_AUGMENTATION": False, # Can be enabled/disabled for experiments
    "AUGMENTATION_PARAMS": {
        "NOISE_LEVEL": 0.05,
        "ROTATION_DEGREES": 10,
        "TRANSLATION_FRAC": 0.1,
    }
}

# --- V2 Training Hyperparameters ---
BATCH_SIZE = 16 # Lowered for 150-frame sequences
NUM_EPOCHS = 100
ENABLE_PER_FOLD_SCALING = True
SCALER_KIND = "robust"
SAVE_SCALERS = True

MAX_FLOW_RATES = {
    "gypsum_all_dataset_v2": 1.3,
    "gypsum_10_hole_dataset_v2": 4.6,
    "brickcladding_all_dataset_v2": 1.5686,
    "hardyboard_all_dataset_v2": 1.58645,
}
MAX_FLOW_RATE = MAX_FLOW_RATES.get(EXPERIMENT_NAME, 5.0)

# Normalization constants for the single-channel thermal crops
# NOTE: If using instance normalization in the data prep, these should be (0, 1)
NORM_CONSTANTS = {
    1: {"mean": [0.0], "std": [1.0]}, # Standard normalization for [0,1] data to [-1,1]
}

# Tuned hyperparameters from Optuna search
# GYPSUM ALL
# INITIAL_PARAMS = {
#     'lr': 9.6681665526918e-05,
#     'weight_decay': 1.0347688244846932e-06,
#     'dropout_rate': 0.2601194262689661,
#     'lstm_hidden_size': 64,
#     'lstm_layers': 3,
#     'optimizer': 'Adam',
# }
# BRICKCLADDING ALL
INITIAL_PARAMS = {  #hyperparam tuned
    "lr": 0.00010386262010394423,
    "weight_decay": 3.0503844758115838e-05,
    "dropout_rate": 0.297074888861393,
    "lstm_hidden_size": 256,
    "lstm_layers": 2,
    "optimizer": "AdamW"
}

# HARDYBOARD
# INITIAL_PARAMS = { #maksysm hyperparams
#     'lr': 0.001,
#     'weight_decay': 0.0001,
#     'dropout_rate': 0.3,
#     'lstm_hidden_size': 256, 
#     'lstm_layers': 3,
#     'optimizer': 'AdamW'
# }
INITIAL_PARAMS = {  # tuned hyperparams
    'lr': 9.365848987187114e-05,
    'weight_decay': 2.6103804428607636e-05,
    'dropout_rate': 0.11326216866883788,
    'lstm_hidden_size': 128,
    'lstm_layers': 1,
    'optimizer': 'AdamW',
}


# This dictionary is needed by the data creation script to find the raw files.
DATASET_CONFIGS = {
    # "gypsum_0716": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07162025_noshutter"},
    # "gypsum_0725": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07252025_noshutter"},
    # "gypsum_0729": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_07292025_noshutter"},
    # "gypsum_0307": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_03072025"},  # OLD
   
    "gypsum_0903": {"material": "gypsum", "dataset_subfolder": "Fluke_Gypsum_09032025_10holes_noshutter_Sameem"},
   
    # "brick_cladding_0616": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0616_2025_noshutter"},
    # "brick_cladding_0805": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0805_2025_noshutter"},
    # "brick_cladding_0808": {"material": "brick_cladding", "dataset_subfolder": "Fluke_BrickCladding_2holes_0808_2025_noshutter"},
    # "hardyboard_0813": {"material": "hardyboard", "dataset_subfolder": "Fluke_HardyBoard_08132025_2holes_noshutter"},
    # "hardyboard_0313": {"material": "hardyboard", "dataset_subfolder": "Fluke_HardyBoard_03132025"}, # OLD
}