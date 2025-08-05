#!/bin/bash

#==============================================================================
# SBATCH Directives for FINAL MODEL training
#==============================================================================
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=airflow-cnn-final  # New job name
#SBATCH --output=logs/cnn_final_%j.out # Log for a single job
#SBATCH --time=04:00:00               # Request enough time for 150 epochs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                  # Only need 1 GPU for this
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1527145@utah.edu

#==============================================================================
# Environment Setup (Same as before)
#==============================================================================
PROJECT_NAME="Airflow-rate-prediction"
CONDA_ENV_NAME="airflow_cnn_env"
PROJECT_DIR="/scratch/general/vast/u1527145/${PROJECT_NAME}"

echo "========================================================"
echo "Job starting on $(date)"
echo "Project Directory: $PROJECT_DIR"
echo "========================================================"

cd "$PROJECT_DIR" || exit 1

module purge
module load miniconda3/23.11.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

export TORCH_HOME="${PROJECT_DIR}/cache"
mkdir -p logs cache

nvidia-smi

#==============================================================================
# Run the Training Script
#==============================================================================
echo "--- Starting FINAL Model Training on ALL Development Data ---"

python -m src_cnn.train_final_model

echo "========================================================"
echo "Job finished on $(date)"
echo "========================================================"