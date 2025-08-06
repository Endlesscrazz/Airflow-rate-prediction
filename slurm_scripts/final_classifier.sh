#!/bin/bash

#==============================================================================
# SBATCH Directives
#==============================================================================
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=airflow-cls-final   # Job Name: "cls-final" for final classifier
#SBATCH --output=logs/final_classifier_train_%j.out # Log file name with job ID
#SBATCH --time=03:00:00                # Time requested
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

export CHPC_SCRATCH_DIR="/scratch/general/vast/u1527145"

#==============================================================================
# Run the Training Script
#==============================================================================
echo "--- Starting FINAL Classifier Training ---"

# Call the new train_final_classifier.py script
python -m src_cnn.train_final_classifier

echo "========================================================"
echo "Job finished on $(date)"
echo "========================================================"