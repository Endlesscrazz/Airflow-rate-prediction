#!/bin/bash

#==============================================================================
# SBATCH Directives for V2 Model Training
#==============================================================================
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=airflow-v2-train
#SBATCH --output=logs/v2_training_%j.out
#SBATCH --time=04:00:00  # Adjust as needed
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
CONDA_ENV_NAME="airflow_cnn_env" # Or your new env name
USER_SCRATCH_DIR="/scratch/general/vast/u1527145"
PROJECT_DIR="${USER_SCRATCH_DIR}/${PROJECT_NAME}"

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
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

#==============================================================================
# Run the V2 Training Script
#==============================================================================
echo "--- Starting V2 Model Training ---"

python src_cnn_v2/train_v2.py

echo "========================================================"
echo "Training job finished on $(date)"
echo "========================================================"