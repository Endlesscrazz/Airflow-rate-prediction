# V3 Pipeline Implementation Tasks

## Phase 1: Robust Mask & Feature Generation
- [ ] **1.1 Config Setup:** Create `src_cnn_v3/config_v3.py` with standard resize dims (32x32) and feature toggles.
- [ ] **1.2 Core Detection:** Create `src_cnn_v3/core/leak_detection.py`. Implement `FusedSignalDetector` (Theil-Sen + Z-score).
- [ ] **1.3 Execution Script:** Create `scripts/generate_masks_v3.py`.
    - [ ] Load videos.
    - [ ] Detect leak epicenters.
    - [ ] Run SAM to get binary masks.
    - [ ] Calculate Oriented Bounding Boxes (OBBs).
    - [ ] Save `_features.json` and verification plots.
- [ ] **1.4 Verification:** Inspect `_verification.png` files for 5-10 videos. Ensure OBBs match leak shapes (slits vs circles).

## Phase 2: Shape-Aware Dataset Creation
- [ ] **2.1 Core Image Proc:** Create `src_cnn_v3/core/image_processing.py`. Implement `crop_rotate_resize_v3()`.
- [ ] **2.2 Dataset Script:** Create `src_cnn_v3/create_cnn_dataset_v3.py`.
    - [ ] Read `_features.json`.
    - [ ] Apply transformations.
    - [ ] Save standardized 32x32 `.npy` files.
- [ ] **2.3 Metadata:** Create `src_cnn_v3/create_metadata_v3.py` to merge ground truth with new shape features.
- [ ] **2.4 Verification:** Inspect standardized `.npy` files (save as PNGs temporarily). Ensure slits look like horizontal/vertical bars.

## Phase 3: Hybrid Model & Training
- [ ] **3.1 Dataset Class:** Create `src_cnn_v3/dataset_utils_v3.py` (return img + features + target).
- [ ] **3.2 Model:** Create `src_cnn_v3/models_v3.py`. Implement `HybridCropRegressor` with `use_handcrafted_features` flag.
- [ ] **3.3 Training:** Create `src_cnn_v3/train_v3.py`.
- [ ] **3.4 Evaluation:** Create `src_cnn_v3/predict_v3.py`.
- [ ] **3.5 Verification:** Run ablation study (Features ON vs OFF).