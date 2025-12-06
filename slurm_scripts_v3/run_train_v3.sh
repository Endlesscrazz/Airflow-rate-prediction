#!/bin/bash

#==============================================================================
# SBATCH Directives for V3 Hybrid Model Training
#==============================================================================
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --nodelist=notch448           # Owner node
#SBATCH --job-name=v3_train_hybrid
#SBATCH --output=logs_v3/training-%j.out
#SBATCH --error=logs_v3/training-%j.err
#SBATCH --time=24:00:00               # 24 hours (plenty for 100 epochs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # 8 CPU cores for DataLoader
#SBATCH --gres=gpu:1                  # 1 GPU
#SBATCH --mem=32G                     # 32GB RAM
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1527145@utah.edu

#==============================================================================
# Environment Setup
#==============================================================================
PROJECT_DIR="/scratch/general/vast/u1527145/Airflow-rate-prediction"

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job starting on $(date)"
echo "Running on host: $(hostname)"
echo "Project Directory: $PROJECT_DIR"
echo "========================================================"

cd "$PROJECT_DIR" || { echo "Error: Could not change directory to $PROJECT_DIR"; exit 1; }

# 1. Prepare Directories
mkdir -p logs_v3
mkdir -p cache

# 2. Activate V3 Clean Environment
module purge
module load python/3.11
source airflow_v3/bin/activate


# 4. Verify GPU Access
echo "--- Environment & GPU Check ---"
which python
python -c 'import torch; print(f"Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")'
nvidia-smi
echo "-------------------------------"

# Prevent Torch from downloading datasets to Home dir
export TORCH_HOME="${PROJECT_DIR}/cache"

#==============================================================================
# Run the V3 Training
#==============================================================================
echo "--- Starting V3 Hybrid Model Training ---"

# The script reads all parameters from src_cnn_v3/config_v3.py
python src_cnn_v3/train_v3.py

echo "========================================================"
echo "Training job finished on $(date)"
echo "========================================================"