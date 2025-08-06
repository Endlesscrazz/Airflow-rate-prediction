#!/bin/bash

#==============================================================================
# SBATCH Directives
#==============================================================================
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=airflow-cnn-cv
#SBATCH --array=0-4
#SBATCH --output=logs/cnn_lstm_cv_%A_%a.out
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1527145@utah.edu

#==============================================================================
# Environment Setup
#==============================================================================
PROJECT_NAME="Airflow-rate-prediction"
CONDA_ENV_NAME="airflow_cnn_env"
USER_SCRATCH_DIR="/scratch/general/vast/u1527145"
PROJECT_DIR="${USER_SCRATCH_DIR}/${PROJECT_NAME}"

echo "========================================================"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job starting on $(date)"
echo "Running on host: $(hostname)"
echo "Project Directory: $PROJECT_DIR"
echo "========================================================"

cd "$PROJECT_DIR" || { echo "Error: Could not change directory to $PROJECT_DIR"; exit 1; }

module purge
module load miniconda3/23.11.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

echo "--- Verifying Environment ---"
echo "Conda Env: $CONDA_DEFAULT_ENV"
python -c 'import torch; print(f"Torch version: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")'
echo "---------------------------"

export TORCH_HOME="${PROJECT_DIR}/cache"
mkdir -p logs cache

echo "--- GPU Info ---"
nvidia-smi
echo "----------------"

export CHPC_SCRATCH_DIR="/scratch/general/vast/u1527145"

# --- Experiment 1: Simplified 'avg' model on 2-channel flow data ---
# This is the crucial diagnostic test.
# MODEL_TYPE="avg"
# DATASET_DIR="cnn_dataset/dataset_cnn_lstm_flow"
# IN_CHANNELS=2


# --- Experiment 2: Full 'lstm' model on 2-channel flow data ---
# This is your best-performing model so far, for comparison.
# To run this, comment out the block above and uncomment this one.
# MODEL_TYPE="lstm"
# DATASET_DIR="CNN_dataset/dataset_2ch_flow"
# IN_CHANNELS=2

# --- Experiment 2)a: Full 'lstm' model on 1-channel thermal data ---
# This is your best-performing model so far, for comparison.
# To run this, comment out the block above and uncomment this one.
MODEL_TYPE="lstm"
DATASET_DIR="CNN_dataset/dataset_1ch_thermal"
IN_CHANNELS=1


# --- Experiment 3: Full 'lstm' model on 3-channel hybrid data ---
# To run this, comment out the other blocks and uncomment this one.
# MODEL_TYPE="lstm"
# DATASET_DIR="cnn_dataset/dataset_cnn_lstm_3-channel"
# IN_CHANNELS=3

#==============================================================================
# Run the Training Script
#==============================================================================
echo "--- Starting Training for Fold ${SLURM_ARRAY_TASK_ID} ---"

# Calculate the total number of folds from the SLURM array variable
TOTAL_FOLDS=$((${SLURM_ARRAY_TASK_MAX} + 1))

echo "Running with Fold: ${SLURM_ARRAY_TASK_ID}, Total Folds: ${TOTAL_FOLDS}"

# Run the python script, passing both required arguments.
# Running as a module with `python -m` is good practice.
python -m src_cnn.train_cnn \
    --fold ${SLURM_ARRAY_TASK_ID} \
    --total_folds ${TOTAL_FOLDS} \
    --model_type "${MODEL_TYPE}" \
    --dataset_dir "${PROJECT_DIR}/${DATASET_DIR}" \
    --in_channels ${IN_CHANNELS}


echo "========================================================"
echo "Job Task ${SLURM_ARRAY_TASK_ID} finished on $(date)"
echo "========================================================"