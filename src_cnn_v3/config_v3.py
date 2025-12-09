# src_cnn_v3/config_v3.py
import os
import torch

# --- Core Settings ---
if torch.cuda.is_available():
    DEVICE = "cuda"       # For CHPC / Colab
elif torch.backends.mps.is_available():
    DEVICE = "mps"        # For Mac (Metal Performance Shaders)
else:
    DEVICE = "cpu"        # Fallback
    
RANDOM_STATE = 42

# --- Experiment Settings ---
EXPERIMENT_NAME = "gypsum-10-hole_v3_hybrid"
EXPERIMENT_VERSION = "v3_iter2_32x32_features_ON"

# --- V3 Specifics ---
RESIZE_DIM = (32, 32)
USE_HANDCRAFTED_FEATURES = True
SAM_CHECKPOINT_PATH = os.path.join('SAM', 'sam_checkpoints', 'sam_vit_b_01ec64.pth')
SAM_MODEL_TYPE = "vit_b"

# --- Hyperparameters (Centralized Control) ---
INITIAL_PARAMS = {
    'lr': 1e-3,              # Start with higher LR if using Scheduler
    'weight_decay': 1e-4,    # L2 Regularization
    'dropout_rate': 0.3,     # Dropout probability
    'lstm_hidden_size': 256, # Increased capacity (was 128)
    'lstm_layers': 3,        # Deeper temporal logic
    'optimizer': 'AdamW'
}

# --- Target Normalization ---
# We normalize airflow targets to [0, 1] during training
MAX_FLOW_RATES = {
    "gypsum-10-hole_v3_hybrid": 24.2645,
    # Can Add others as needed
}
# Default to 10.0 if not found, preventing division by zero errors
MAX_FLOW_RATE = MAX_FLOW_RATES.get(EXPERIMENT_NAME, 25.0)

# --- Preprocessing Flags (Control Flow) ---
V3_PREPROCESS_PARAMS = {
    "ENABLE_TEMPORAL_NORM": True,  # (Frame / Mean) - Removes ambient fluctuation. Baked in during creation.
    "ENABLE_SPATIAL_BLUR": True,   # (3x3 Box Blur) - Reduces sensor noise. Baked in during creation.
    "ENABLE_INSTANCE_NORM": True,  # (x / max) - Contrast invariance. Applied On-the-Fly in DataLoader.
    "BLUR_KERNEL_SIZE": (3, 3)
}

# --- Data Generation Parameters ---
FOCUS_DURATION_SECONDS = 30
TRUE_FPS = 5
NUM_FRAMES_PER_SAMPLE = 150
NUM_AUGMENTATIONS = 20

# --- Path Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "CNN_dataset_v3")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Output_CNN-LSTM_V3")
RAW_DATASET_PARENT_DIR = '/scratch/general/vast/u1527145/datasets'
INTERMEDIATE_DATA_DIR = os.path.join(PROJECT_ROOT, "Output_SAM_V3", "datasets")

OUTPUT_DIR = os.path.join(DATA_DIR, EXPERIMENT_NAME)
EXPERIMENT_RESULTS_DIR = os.path.join(RESULTS_DIR, EXPERIMENT_NAME, EXPERIMENT_VERSION)
GROUND_TRUTH_CSV_PATH = os.path.join(PROJECT_ROOT, "airflow_ground_truth_gypsum_10holes.csv")
MASTER_METADATA_PATH = os.path.join(OUTPUT_DIR, "master_metadata_v3.csv")

# --- SPLIT PATHS ---
TRAIN_SPLIT_PATH = os.path.join(OUTPUT_DIR, f"train_split_seed{RANDOM_STATE}.csv")
VAL_SPLIT_PATH = os.path.join(OUTPUT_DIR, f"val_split_seed{RANDOM_STATE}.csv")
TEST_SPLIT_PATH = os.path.join(OUTPUT_DIR, f"test_split_seed{RANDOM_STATE}.csv")

# --- DATASET DIR ---
DATASET_SUBDIR = f"dataset_32x32_aug{NUM_AUGMENTATIONS}"
DATASET_DIR = os.path.join(OUTPUT_DIR, DATASET_SUBDIR)

# --- TEMPLATES ---
GYPSUM_10HOLE_TEMPLATE = {
    1:  (145, 228), 2:  (204, 70), 3:  (367, 46), 4:  (526, 74),
    5:  (488, 204), 6:  (357, 198), 7:  (401, 319), 8:  (258, 315),
    9:  (553, 322), 10: (353, 414)
}

# --- DATASET CONFIGS ---
DATASET_CONFIGS = {
    "gypsum_10holes_0903": {
        "material": "gypsum", 
        "dataset_subfolder": "Fluke_Gypsum_09032025_10holes_noshutter_Sameem",
        "num_leaks": 10,  
        "template": GYPSUM_10HOLE_TEMPLATE 
    },
    # Example for other datasets:
    # "hardyboard_0813": {
    #     "material": "hardyboard", 
    #     "dataset_subfolder": "Fluke_HardyBoard_08132025_2holes_noshutter",
    #     "num_leaks": 2,
    #     "template": None
    # }
}