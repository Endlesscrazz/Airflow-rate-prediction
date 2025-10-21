# CNN-LSTM Airflow Rate Prediction (V2 Pipeline)

This project aims to predict quantitative airflow leakage rates from thermal infrared video sequences using a deep learning approach.

This document describes the **V2 Pipeline**, a "bottom-up" methodology designed to be robust for small and varied datasets. It uses a simplified CNN-LSTM model trained on small, augmented image crops of thermal hotspots.

## Project Structure

The project is organized into a data pipeline (`CNN_dataset`), a results repository (`Output_CNN-LSTM`), and several source code directories. The primary focus for this workflow is the `src_cnn_v2` directory.
Airflow-rate-prediction/
│
├── CNN_dataset/
│ └── <experiment_name>/ # Processed data for a specific experiment
│ ├── master_metadata_v2.csv # Lightweight master list of all valid samples
│ ├── train_split.csv # List of samples for the training set
│ ├── val_split.csv # List of samples for the validation set
│ ├── test_split.csv # List of samples for the test set
│ └── dataset_1ch_cropped_augmented/ # Folder with all .npy image crops
│ ├── train_metadata_v2.csv # Final metadata for the DataLoader
│ ├── val_metadata_v2.csv
│ └── test_metadata_v2.csv
│
├── Output_CNN-LSTM/
│ └── <experiment_name>/ # All final results for the experiment
│ ├── best_model_v2.pth # The trained model weights
│ ├── scaler_v2.pkl # The fitted feature scaler
│ ├── training_log.csv # Epoch-by-epoch training history
│ ├── test_set_predictions.csv # Detailed per-sample results on the test set
│ ├── experiment_summary.txt # A complete log of all experimental parameters
│ └── ... (plots and visualizations)
│
├── data/ # (Optional) Location for raw ground truth CSVs
│
├── output_SAM/ # Location of generated hotspot masks
│
├── src_cnn_v2/ # <== CORE SOURCE CODE FOR THIS PIPELINE
│ ├── config_v2.py # Central configuration file for the V2 pipeline
│ ├── create_metadata_v2.py # Step 1: Validates raw data and creates master list
│ ├── split_data_v2.py # Step 2: Creates train/val/test splits
│ ├── create_dataset_v2.py # Step 3: Generates cropped & augmented .npy files
│ ├── train_v2.py # Step 4: Trains the model and evaluates on the test set
│ ├── hyperparam_search_v2.py # (Optional) Finds optimal hyperparameters
│ ├── predict_v2.py # (Optional) Runs standalone inference on the test set
│ ├── visualizations_v2.py # (Optional) Generates plots from training results
│ ├── models_v2.py # Defines the CNN-LSTM model architecture
│ ├── dataset_utils_v2.py # Defines the custom PyTorch Dataset
│ └── logging_utils_v2.py # Helper for logging experiment parameters
│
└── ... (other project files and folders)


## End-to-End Workflow

Follow these steps to run a complete experiment on a new dataset.

### Step 0: Initial Setup

1.  **Prepare Raw Data:** Ensure your raw `.mat` video files are in a known location and you have the corresponding ground truth `airflow_ground_truth_*.csv` file.
2.  **Generate Masks:** Run the `run_SAM.py` script to generate the `.npy` hotspot masks for all videos in your dataset.
3.  **Configure the Experiment:** Open `src_cnn_v2/config_v2.py` and modify the following key variables:
    *   `EXPERIMENT_NAME`: Set a unique name for this experiment (e.g., `"hardyboard_all_dataset_v2"`). This will be the name of the data and results folders.
    *   `GROUND_TRUTH_CSV_PATH`: Point this to the correct ground truth file for your dataset.
    *   `DATASET_CONFIGS`: Update this dictionary to include the subfolder names of your raw dataset.
    *   `RAW_DATASET_PARENT_DIR` and `RAW_MASK_PARENT_DIR`: Verify these top-level paths are correct.

### Step 1: Data Preparation

These scripts prepare the data on disk. They are typically run once per experiment. It is highly recommended to run these on a CHPC node using the provided SLURM script.

**To run on CHPC:**
```bash
sbatch run_dataprep_v2.slurm
This will execute the following three Python scripts in order:

create_metadata_v2.py: Creates the master list of all valid samples.
split_data_v2.py: Creates the stratified, grouped train/validation/test splits.
create_cnn_dataset_v2.py: Generates the final cropped and augmented .npy image files.
Step 2: Hyperparameter Search (Optional but Recommended)
This step uses Optuna to find the best set of hyperparameters for your model and dataset.

To run on CHPC:

code
Bash
sbatch run_tuning_v2.slurm
This executes hyperparam_search_v2.py. After the job completes, check the output log. It will print an INITIAL_PARAMS dictionary with the optimal values found. Copy this dictionary and paste it into src_cnn_v2/config_v2.py, replacing the old one.

Step 3: Final Model Training
This step trains the model using the prepared data and the (optionally tuned) hyperparameters.

To run on CHPC:

code
Bash
sbatch run_training_v2.slurm
This executes train_v2.py. The script will:

Train the model on the training set.
Use the validation set for early stopping and to save the best model.
Automatically run a final evaluation on the held-back test set.
Save all results (model, logs, predictions) to the Output_CNN-LSTM/<experiment_name> directory.
Step 4: Analysis and Visualization
After training is complete, download the entire Output_CNN-LSTM/<experiment_name> directory to your local machine. You can then run the analysis scripts locally.

Generate Plots:
code
Bash
python src_cnn_v2/visualizations_v2.py
This will generate learning_curves.png, test_set_predictions_vs_true.png, and other plots inside your results folder.
Get a Detailed Report:
code
Bash
python src_cnn_v2/predict_v2.py
