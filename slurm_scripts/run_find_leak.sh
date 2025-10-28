#!/bin/bash

#==============================================================================
# SBATCH Directives for Leak Coordinate Generation
#==============================================================================
#SBATCH --job-name=leak_coord_gen       # A descriptive name for your job
#SBATCH --account=yqu-gpu-np            # Use the same account as your training job
#SBATCH --partition=yqu-gpu-np          # Use the same partition
#SBATCH --output=leak_logs/coord_gen-%j.out # Path to save the standard output
#SBATCH --error=leak_logs/coord_gen-%j.err  # Path to save the error output
#SBATCH --time=02:30:00                   # Time limit (HH:MM:SS)
#SBATCH --nodes=1                         # We need only one machine
#SBATCH --ntasks-per-node=1               # This script itself is a single task
#SBATCH --cpus-per-task=8                 # Request 8 CPU cores for joblib to use
#SBATCH --mem=32G                         # Request 16GB of memory

#==============================================================================
# Environment Setup (COPIED FROM YOUR WORKING run_train_v2.sh)
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

# Change to the project directory first
cd "$PROJECT_DIR" || { echo "Error: Could not change directory to $PROJECT_DIR"; exit 1; }

# Create the logs directory if it doesn't exist
mkdir -p leak_logs

# Correct environment activation sequence for your CHPC
module purge
module load miniconda3/23.11.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

# Verification step to ensure the correct environment is active
echo "--- Verifying Environment ---"
echo "Conda Env: $CONDA_DEFAULT_ENV"
which python
python -c 'import numpy; import scipy; print(f"NumPy version: {numpy.__version__}"); print(f"SciPy version: {scipy.__version__}")'
echo "---------------------------"

#==============================================================================
# Run the Leak Finding Script
#==============================================================================
echo "--- Starting Leak Coordinate Generation ---"
echo "--Fluke_HardyBoard_08132025_2holes_noshutter--"

# Run the script using the robust 'python -m' method.
# The command arguments are taken directly from your request.
python -m src_cnn_v2.find_leaking_holes \
    --dataset_dir /scratch/general/vast/u1527145/datasets/Fluke_HardyBoard_08132025_2holes_noshutter \
    --base_output_dir /scratch/general/vast/u1527145/Airflow-rate-prediction/Output_SAM/datasets/Fluke_HardyBoard_08132025_2holes_noshutter \
    --num_leaks 2 \
    --debug

echo "========================================================"
echo "Leak coordinate generation finished on $(date)"
echo "========================================================"