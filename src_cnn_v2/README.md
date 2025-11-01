# CNN-LSTM Airflow Rate Prediction (V2 Pipeline)

This project predicts quantitative airflow leakage rates from thermal infrared video sequences using a deep learning pipeline. This document describes the V2 workflow, a "bottom-up" methodology using a CNN-LSTM model trained on small, precisely located image crops of thermal hotspots.

## Project Structure

Airflow-rate-prediction/
├── CNN_dataset/
│ └── <experiment_name>/
│ ├── master_metadata_v2.csv
│ ├── train_split_seed42.csv
│ ├── val_split_seed42.csv
│ └── test_split_seed42.csv
│ └── dataset_cs15_nf150_aug99/
│ ├── train_metadata_seed42.csv
│ ├── ... (all .npy image crop files)
├── Output_CNN-LSTM/
│ └── <experiment_name>/
│ └── <experiment_version>/
│ ├── best_model_v2.pth
│ ├── scaler_v2.pkl
│ ├── training_log.csv
│ ├── test_set_report.xlsx
│ └── experiment_summary.txt
├── Output_SAM/
│ └── datasets/
│ └── <dataset_subfolder>/
│ └── <video_id>/
│ └── <video_id>_coordinates.json
├── scripts/
│ └── create_ground_truth_labels.py
└── src_cnn_v2/
├── config_v2.py
├── find_leaking_holes.py
├── create_metadata_v2.py
├── split_data_v2.py
├── create_cnn_dataset_v2.py
├── train_v2.py
├── predict_v2.py
└── ... (model and utility files)

## End-to-End Workflow

Follow these steps to configure and run a complete experiment.

### Step 1: Initial Configuration

1. **Prepare Raw Data**: Ensure your raw `.mat` video files and their corresponding ground truth flow rate files (e.g., `flowrate_data.txt`) are located in your raw dataset directory.

2. **Configure the Experiment**: Open `src_cnn_v2/config_v2.py` and set the following critical variables:

    - `EXPERIMENT_NAME`: A unique name for your dataset (e.g., `"hardyboard_all_dataset_v2"`). This defines the top-level folder for all processed data and results.
    - `EXPERIMENT_VERSION`: A unique name for this specific run (e.g., `"run-1-tuned-hyperparams"`). This creates a unique subfolder for the final results.
    - `GROUND_TRUTH_CSV_PATH`: The full path to the master ground truth CSV file you will generate.
    - `RAW_DATASET_PARENT_DIR`: The absolute path to the parent directory containing your raw `.mat` video datasets.
    - `RAW_MASK_PARENT_DIR`: The absolute path to the parent directory where the output of the leak finding script will be saved (e.g., `.../Output_SAM/datasets`).
    - `DATASET_CONFIGS`: Update this dictionary to map your experiment's raw data subfolders.

### Step 2: Data Preparation Pipeline

These scripts process the raw data and prepare it for the model. They are typically run once per experiment. It is highly recommended to run these on a CHPC node.

1. **Generate Master Ground Truth CSV**:
   - **Action**: Configure and run the `scripts/create_ground_truth_labels.py` script.
   - **Purpose**: This script reads your raw `flowrate_data.txt` files and creates a single, unified `airflow_ground_truth_*.csv` file. This only needs to be done once per material.

2. **Find Leak Coordinates**:
   - **Action**: Run the `find_leaking_holes.py` script (preferably via its SLURM script on a CHPC node).
   - **Command**:
     ```bash
     sbatch run_coord_gen.slurm
     ```
   - **Purpose**: This is the core analysis step. It processes every `.mat` video, performs advanced thermal analysis, and saves the precise (y, x) coordinates of each leak into a `_coordinates.json` file.

3. **Create and Split Master Metadata**:
   - **Action**: Run the `create_metadata_v2.py` and `split_data_v2.py` scripts in order.
   - **Commands**:
     ```bash
     python -m src_cnn_v2.create_metadata_v2
     python -m src_cnn_v2.split_data_v2
     ```
   - **Purpose**: 
     - `create_metadata` verifies that all required files exist.
     - `split_data` then creates the final, leak-proof `train_split`, `val_split`, and `test_split` files.

4. **Generate Final .npy Dataset**:
   - **Action**: Run the `create_cnn_dataset_v2.py` script.
   - **Command**:
     ```bash
     python -m src_cnn_v2.create_cnn_dataset_v2
     ```
   - **Purpose**: This script reads the split files and the coordinate files, extracts the fixed-size image crops from the raw videos, performs data augmentation, and saves the final `.npy` files and `metadata.csv` files for the model.

### Step 3: Model Training and Evaluation

1. **Hyperparameter Search (Optional but Recommended)**:
   - **Action**: Run the `hyperparam_search_v2.py` script to find the optimal model parameters for your dataset.
   - **Purpose**: This script uses Optuna to automatically test many different hyperparameter combinations and find the best set, which you can then copy into `config_v2.py`.

2. **Final Model Training**:
   - **Action**: Run the main training script. This will train the model, use the validation set for early stopping, and save the best performing model.
   - **Command**:
     ```bash
     python -m src_cnn_v2.train_v2
     ```

### Step 4: Final Evaluation and Analysis

1. **Generate Test Set Predictions**:
   - **Action**: Run the `predict_v2.py` script. This loads your best trained model and evaluates its performance on the held-back test set.
   - **Command**:
     ```bash
     python -m src_cnn_v2.predict_v2
     ```
   - **Output**: This generates a `test_set_report.xlsx` with detailed per-sample predictions and a summary of final performance metrics (R², MAE, RMSE, etc.).

2. **Create Visualizations**:
   - **Action**: Run the `visualizations_v2.py` script.
   - **Command**:
     ```bash
     python -m src_cnn_v2.visualizations_v2
     ```
   - **Output**: This generates plots such as the model's learning curves and a scatter plot of predicted vs. true values for the test set, saving them to your experiment's results folder.

## Dependencies

To run the pipeline, the following dependencies are required:

- Python 3.7+
- TensorFlow 2.x
- PyTorch 1.7+
- NumPy
- Optuna
- Matplotlib
- scikit-learn

Install dependencies using `pip` or a virtual environment.

```bash
pip install -r requirements.txt