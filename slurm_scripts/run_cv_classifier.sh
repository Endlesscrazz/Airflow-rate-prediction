#!/bin/bash

#==============================================================================
# SBATCH Directives
#==============================================================================
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=airflow-cls-cv      # New Job Name: "cls" for classifier
#SBATCH --array=0-4
#SBATCH --output=logs/classifier_cv_%A_%a.out # New log file names
#SBATCH --time=03:00:00                # Keeping the longer time just in case
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
# These variables can remain the same
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

# This setup is specific to your CHPC and is preserved
module purge
module load miniconda3/23.11.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

echo "--- Verifying Environment ---"
echo "Conda Env: $CONDA_DEFAULT_ENV"
python -c 'import torch; print(f"Torch version: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")'
echo "---------------------------"

# Create directories if they don't exist
export TORCH_HOME="${PROJECT_DIR}/cache"
mkdir -p logs cache

echo "--- GPU Info ---"
nvidia-smi
echo "----------------"

# This export might be redundant if you're already in the dir, but it's harmless
export CHPC_SCRATCH_DIR="/scratch/general/vast/u1527145"

#==============================================================================
# Run the Training Script
#==============================================================================
echo "--- Starting CLASSIFIER Training for Fold ${SLURM_ARRAY_TASK_ID} ---"

# Calculate the total number of folds from the SLURM array variable
TOTAL_FOLDS=$((${SLURM_ARRAY_TASK_MAX} + 1))

echo "Running with Fold: ${SLURM_ARRAY_TASK_ID}, Total Folds: ${TOTAL_FOLDS}"

# --- CRITICAL CHANGE ---
# Call the new train_classifier.py script
python -m src_cnn.train_classifier --fold ${SLURM_ARRAY_TASK_ID} --total_folds ${TOTAL_FOLDS}
# -----------------------


echo "========================================================"
echo "Job Task ${SLURM_ARRAY_TASK_ID} finished on $(date)"
echo "========================================================"