#!/bin/bash

#==============================================================================
# SBATCH Directives for V3 Mask Generation
#==============================================================================
#SBATCH --job-name=v3_mask_gen          # Name of the job
#SBATCH --account=yqu-gpu-np            # Account
#SBATCH --partition=yqu-gpu-np          # Partition
#SBATCH --output=logs_v3/mask_gen-%j.out     # Standard output log
#SBATCH --error=logs_v3/mask_gen-%j.err      # Error log
#SBATCH --time=08:00:00                 # Time limit (4 hours)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16              # Request 16 Cores for fast Theil-Sen calculation
#SBATCH --mem=32G                       # Request 32GB RAM for Video + SAM
#SBATCH --gres=gpu:1                    # Request 1 GPU for SAM (Critical for speed)

#==============================================================================
# Environment Setup
#==============================================================================
USER_SCRATCH_DIR="/scratch/general/vast/u1527145"
PROJECT_DIR="${USER_SCRATCH_DIR}/Airflow-rate-prediction"

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job starting on $(date)"
echo "Running on host: $(hostname)"
echo "Project Directory: $PROJECT_DIR"
echo "========================================================"

# 1. Go to project directory
cd "$PROJECT_DIR" || { echo "Error: Could not change directory"; exit 1; }

# 2. Create logs directory
mkdir -p logs_v3

# 3. Load System Python
module purge
module load python/3.11

# 4. Activate the CLEAN V3 Environment (Not Conda)
# This prevents the 'libffi' and 'requests' errors we saw earlier
source airflow_v3/bin/activate


# Verification
echo "--- Environment Verification ---"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo "--------------------------------"

#==============================================================================
# Run the V3 Generator
#==============================================================================
echo "--- Starting V3 Mask & Feature Generation ---"

# We don't need arguments anymore; the script reads everything from config_v3.py
python scripts_v3/generate_masks_v3.py

echo "========================================================"
echo "Finished on $(date)"
echo "========================================================"