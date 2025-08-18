#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=airflow-optuna
#SBATCH --output=logs/optuna_search_%j.out
#SBATCH --time=12:00:00  # Request significant time for many trials
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

#==============================================================================
# Environment Setup
#==============================================================================
PROJECT_NAME="Airflow-rate-prediction"
CONDA_ENV_NAME="airflow_cnn_env"
PROJECT_DIR="/scratch/general/vast/u1527145/${PROJECT_NAME}"

cd "$PROJECT_DIR" || exit 1

module purge
module load miniconda3/23.11.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

#==============================================================================
# --- MODIFIED: Experiment Configuration ---
#==============================================================================
# Define which dataset you want to run the hyperparameter search on.
# This makes it easy to switch between experiments (e.g., 1ch vs 2ch).

# --- Best performing dataset based on your previous work ---
DATASET_NAME="dataset_2ch_thermal_masked_f10s"
INPUT_CHANNELS=2

# You could easily switch to another dataset by changing the two lines above, e.g.:
# DATASET_NAME="dataset_1ch_thermal"
# INPUT_CHANNELS=1

# Construct the full paths based on the configuration
DATASET_DIR_PATH="${PROJECT_DIR}/CNN_dataset/${DATASET_NAME}"
METADATA_FILE_PATH="${DATASET_DIR_PATH}/metadata.csv"

# Update the job name and output log to be more descriptive
#SBATCH --job-name=optuna-${DATASET_NAME}
#SBATCH --output=logs/optuna_${DATASET_NAME}_%j.out

#==============================================================================
# Run the Hyperparameter Search
#==============================================================================
echo "--- Starting Optuna Hyperparameter Search ---"
echo "Target Dataset: ${DATASET_NAME}"
echo "Input Channels: ${INPUT_CHANNELS}"
echo "Dataset Path: ${DATASET_DIR_PATH}"
echo "Metadata Path: ${METADATA_FILE_PATH}"

# --- MODIFIED: The python command now includes the new required arguments ---
# Note the change from src_cnn.hyperparam_search to scripts.hyperparam_search
python -m scripts.hyperparam_search \
    --n_trials 100 \
    --num_epochs 75 \
    --fold 0 \
    --in_channels ${INPUT_CHANNELS} \
    --dataset_dir "${DATASET_DIR_PATH}" \
    --metadata_path "${METADATA_FILE_PATH}"

echo "--- Search Complete ---"