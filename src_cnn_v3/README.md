# CNN-LSTM Airflow Rate Prediction (V3 Hybrid Pipeline)

This project predicts quantitative airflow leakage rates (L/min) from thermal infrared video sequences. This document describes the V3 "Shape-Aware" Workflow.

Unlike V2 (which used fixed square crops), V3 uses a Hybrid Architecture:

- **Shape-Aware Vision:** Uses Segment Anything (SAM) to find the precise leak shape, then calculates an Oriented Bounding Box (OBB) to perform a Rotated Crop & Resize. This standardizes inputs (32x32) regardless of whether the leak is a circle, a vertical slit, or a horizontal slit.
- **Hybrid Modeling:** Fuses the visual data (CNN-LSTM) with handcrafted geometric features (Area, Aspect Ratio, Extent) processed via an MLP.

---

## Project Structure

Airflow-rate-prediction/
├── CNN_dataset_v3/
│   └── <experiment_name>/
│       ├── master_metadata_v3.csv       # Merges Ground Truth + SAM Features
│       ├── train_split_seed42.csv
│       ├── val_split_seed42.csv
│       ├── test_split_seed42.csv
│       └── dataset_32x32_aug20/         # Standardized Tensors
│           ├── train_metadata.csv
│           ├── ... (all .npy files: Time x 32 x 32)
├── Output_CNN-LSTM_V3/
│   └── <experiment_name>/
│       └── <experiment_version>/
│           ├── best_model.pth           # Saved PyTorch Model
│           ├── feature_scaler.pkl       # Scikit-Learn Scaler for MLP features
│           ├── training_log.csv
│           └── test_report.xlsx
├── Output_SAM_V3/                       # Intermediate Steps
│   └── datasets/
│       └── <dataset_subfolder>/
│           └── <video_id>/
│               ├── <video_id>_features.json  # OBB & Geometry Data
│               ├── <video_id>_mask_X.npy     # Binary Masks
│               └── <video_id>_verification.png
├── scripts/
│   └── create_ground_truth_labels.py    # Universal GT Generator
├── scripts_V3/
│   └── generate_masks_v3.py             # The Detection & Segmentation Engine
├── src_cnn_v3/
│   ├── config_v3.py                     # Central Configuration & Templates
│   ├── core/
│   │   ├── leak_detection.py            # Fused Signal Logic
│   │   ├── image_processing.py          # Crop/Rotate/Resize Logic
│   │   └── preprocessing.py             # Normalization Logic
│   ├── create_metadata_v3.py            # Merges GT CSV with JSON Features
│   ├── split_data_v3.py                 # Stratified Group K-Fold
│   ├── create_cnn_dataset_v3.py         # Generates .npy tensors
│   ├── train_v3.py                      # Hybrid Training Loop
│   ├── predict_v3.py                    # Inference & Metrics
│   ├── models_v3.py                     # HybridCropRegressor Architecture
│   └── dataset_utils_v3.py              # HybridDataset (Video + Features)
End-to-End Workflow

### Step 1: Configuration & Templates
The V3 pipeline relies on Spatial Templates to identify leaks in multi-hole setups.

Open src_cnn_v3/config_v3.py.
Define your templates (coordinates of leaks).
Update DATASET_CONFIGS to include all datasets you want to train on.

# config_v3.py example
GYPSUM_10HOLE_TEMPLATE = { 1: (145, 228), ... 10: (353, 414) }
HARDYBOARD_2HOLE_TEMPLATE = { 1: (322, 328), 2: (130, 499) }

DATASET_CONFIGS = {
    "gypsum_10_hole": {
        "dataset_subfolder": "Fluke_Gypsum_...",
        "template": GYPSUM_10HOLE_TEMPLATE,
        "num_leaks": 10
    },
    "hardyboard_2_hole": {
        "dataset_subfolder": "Fluke_HardyBoard_...",
        "template": HARDYBOARD_2HOLE_TEMPLATE,
        "num_leaks": 2
    }
}

### Step 2: Ground Truth Generation
Generates a master CSV containing flow rates for all configured datasets. The script now supports both old (Voltage-based) and new (Pressure-folder based) formats.

```
python scripts/create_ground_truth_labels.py
```

Output: airflow_ground_truth_gypsum_10holes.csv (or combined name).

### Step 3: Mask & Feature Generation (The V3 Engine)
This is the most computationally intensive preprocessing step. It runs Fused Signal Detection -> SAM Segmentation -> OBB Calculation.

```
python scripts_V3/generate_masks_v3.py
```

Input: Raw .mat videos.
Logic: Detects peaks, matches them to the Template using Hungarian Algorithm, segments them using SAM, calculates Geometry (Area, Aspect Ratio, Extent).
Output: JSON files in Output_SAM_V3/.

### Step 4: Dataset Compilation

#### 4.1 Merge Metadata
Merges the Ground Truth CSV (Step 2) with the Geometric Features JSONs (Step 3).

```
python -m src_cnn_v3.create_metadata_v3
```
Output: CNN_dataset_v3/<exp_name>/master_metadata_v3.csv.

#### 4.2 Split Data
Performs Stratified Group K-Fold. Ensures all frames from one video stay in the same set, while balancing high/low flow rates across splits.

```
python -m src_cnn_v3.split_data_v3
```

#### 4.3 Generate Tensors (Crop-Rotate-Resize)
Reads the OBB info, loads the video, and performs the Rotated Crop. It resizes everything to 32x32.

```
python -m src_cnn_v3.create_cnn_dataset_v3
```
Output: Thousands of .npy files containing standardized 32x32 video stacks (e.g., 150 frames deep).

### Step 5: Hybrid Training
Trains the HybridCropRegressor.

Visual Stream: CNN-LSTM processes the 32x32 video tensor.
Feature Stream: MLP processes [Delta_T, Area, Aspect, Extent].
Fusion: Concatenates embeddings and predicts flow rate.

```
python -m src_cnn_v3.train_v3
```

Artifacts: Saves best_model.pth and feature_scaler.pkl to Output_CNN-LSTM_V3/.

### Step 6: Evaluation
Runs the model on the held-out Test Set and generates detailed reports (Scatter plots, Residuals, Tolerance Curves).

```
python -m src_cnn_v3.predict_v3
```

## Strategy: Combining Datasets (Multi-Material Training)
To improve model robustness, we can combine data from different materials (e.g., Gypsum and Hardyboard) into a single "Universal" or "Pre-trained" model.

1. Unified Ground Truth
Ensure scripts/create_ground_truth_labels.py iterates over both dataset configs.

code
Python
DATASET_CONFIGS = {
    "gypsum": { ... },
    "hardyboard": { ... }
}
This produces a single CSV with a material column.

2. Unified Config
In src_cnn_v3/config_v3.py, ensure DATASET_CONFIGS includes both entries with their respective Templates.

3. Normalization Strategy
Different materials have different thermal properties.

Input Normalization: Instance Normalization (enabled in preprocessing.py) handles the contrast differences between materials automatically.
Target Normalization:
Option A (Separate Heads): Not currently implemented.
Option B (Global Max): Set MAX_FLOW_RATE in config to the highest flow rate observed across ALL datasets (e.g., 25.0 for Gypsum). The model will learn that Hardyboard simply has lower values within that range.
4. Training
Run the standard training script. The StratifiedGroupKFold will automatically ensure that both Gypsum and Hardyboard videos are distributed across Train/Val/Test sets, allowing the model to learn generalized physics features (like "Slit vs Circle") that apply to both materials.

## Dependencies
Core: numpy, pandas, scipy, matplotlib
Deep Learning: torch, torchvision
Computer Vision: opencv-python-headless, scikit-image
Segmentation: segment-anything (Meta), ultralytics (FastSAM - optional for inference)
Utils: tqdm, joblib
