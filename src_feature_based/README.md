# Handcrafted Feature-Based Airflow Prediction Pipeline

This directory contains a complete, end-to-end pipeline for predicting airflow rates from thermal videos using a handcrafted feature-based approach with traditional machine learning models.

It is broken down into sequential scripts that handle data processing, feature generation, data splitting, model training/tuning, and final evaluation.

## Project Structure

-   **`config.py`**: The central configuration file. **This is the main file to edit** to change paths, feature sets, and model parameters.
-   **`data_utils.py`**: Low-level helper functions for parsing filenames and loading ground truth data.
-   **`feature_engineering.py`**: The core logic for calculating all 30+ numerical features from video data.
-   **`modeling.py`**: Defines the machine learning models (XGBoost, MLP, etc.) and the scikit-learn pipelines.
-   **`plotting.py`**: Contains functions for generating diagnostic plots like learning curves.
-   **`utils.py`**: Contains utility functions, including the logger.

The main workflow is executed using scripts from the `scripts` directory in the parent folder.

## Reproducibility Workflow: A Step-by-Step Guide

To reproduce the results from scratch, follow these steps in order. All commands should be run from the main project root directory (`Airflow-rate-prediction/`).

### Step 0: Initial Setup

1.  **Place Raw Data:**
    *   Place the raw thermal video directories (e.g., `Fluke_Gypsum_07162025_noshutter`) inside a top-level folder named `datasets_raw`.
    *   Place the corresponding mask directories inside a top-level folder named `masks_raw`.
    *   Place the `airflow_ground_truth.csv` file in the project's root directory.

2.  **Configure the Pipeline:**
    *   Open `src_feature_based/config.py`.
    *   Verify that `RAW_DATA_ROOT` and `RAW_MASK_PARENT_DIR` point to the correct parent folders (`datasets_raw` and `masks_raw`).
    *   Adjust any other parameters as needed (e.g., `FOCUS_DURATION_SECONDS`, the list of `SELECTED_FEATURES` to use for the final model).

### Step 1: Generate Master Feature Set

This script scans all raw data, calculates every possible handcrafted feature for each sample, and saves the result into a single `master_features.csv` file. This is the most time-consuming step.

**Command:**
```bash
python -m scripts.generate_features
```
**Output:** This will create the file `output_feature_based/master_features.csv`.

### Step 2: Create Train and Hold-Out Sets

This script takes the master feature file and performs a stratified, group-aware split to create a development set (for training and tuning) and a final hold-out set (for unbiased evaluation).

**Command:**
```bash
python -m scripts.split_data
```
**Output:** This will create `train_features.csv` and `holdout_features.csv` inside the `output_feature_based/` directory.

### Step 3: Train, Tune, and Save the Best Model

This is the main training script. It loads the `train_features.csv`, performs a full hyperparameter search using GridSearch with Grouped Cross-Validation for multiple model types (MLP, XGBoost, etc.), identifies the best overall model, and trains a final version of it on the entire development set.

**Command:**
```bash
python -m scripts.train_feature_models_cv
```
**Output:**
-   Console output showing the best CV R² score and parameters for each model type.
-   The final trained model is saved to `output_feature_based/trained_cv_model/final_model_[BestModelName].joblib`.
-   The target scaler (if used) is saved to `output_feature_based/trained_cv_model/final_target_scaler_[BestModelName].joblib`.
-   Diagnostic plots (Learning Curve, Feature Importance) are saved to `output_feature_based/plots/`.

### Step 4: Evaluate on the Hold-Out Set

This is the final step to get the unbiased performance of the best model. It loads the saved model from Step 3 and evaluates it on the `holdout_features.csv`.

**Command:**
You must provide the name of the best model found in the previous step as a command-line argument. For example, if the previous step reported `MLPRegressor` was the best:
```bash
python -m scripts.evaluate_feature_model MLPRegressor
```
**Output:**
-   The final Hold-Out R² and RMSE scores printed to the console.
-   A scatter plot `holdout_performance_plot.png` saved in the `output_feature_based/plots/` directory, visualizing the model's performance.

By following these four steps, the entire analysis can be reproduced from the raw data to the final performance metrics and plots.