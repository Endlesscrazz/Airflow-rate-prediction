#!/bin/bash

#==============================================================================
# SBATCH Directives for V3 Data Preparation Pipeline
#==============================================================================
#SBATCH --job-name=v3_dataprep
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --nodelist=notch448           # Using your preferred node
#SBATCH --output=logs_v3/dataprep-%j.out
#SBATCH --error=logs_v3/dataprep-%j.err
#SBATCH --time=04:00:00               # 4 hours should be plenty for 10-hole dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # High CPU for parallel tensor creation
#SBATCH --mem=32G                     # RAM for loading videos
#SBATCH --gres=gpu:1                  # Keep GPU to match env, though mostly CPU work

#==============================================================================
# Environment Setup (V3 Specific)
#==============================================================================
USER_SCRATCH_DIR="/scratch/general/vast/u1527145"
PROJECT_DIR="${USER_SCRATCH_DIR}/Airflow-rate-prediction"

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job starting on $(date)"
echo "Project Directory: $PROJECT_DIR"
echo "========================================================"

cd "$PROJECT_DIR" || { echo "FATAL: Could not cd to project dir"; exit 1; }

# 1. Clean Logs
mkdir -p logs_v3

# 2. Activate V3 Clean Environment
module purge
module load python/3.11
source airflow_v3/bin/activate


echo "Python: $(which python)"
echo "Environment Active."

#==============================================================================
# PIPELINE EXECUTION
#==============================================================================

# --- STEP 1: Metadata Generation ---
echo -e "\n\n>>> STEP 1: Creating Master Metadata..."
python src_cnn_v3/create_metadata_v3.py
if [ $? -ne 0 ]; then echo "Step 1 Failed"; exit 1; fi

# --- STEP 2: Data Splitting ---
echo -e "\n\n>>> STEP 2: Splitting Data (Train/Val/Test)..."
python src_cnn_v3/split_data_v3.py
if [ $? -ne 0 ]; then echo "Step 2 Failed"; exit 1; fi

# --- STEP 3: Tensor Creation ---
echo -e "\n\n>>> STEP 3: Generating Dataset Tensors (Cropping & Augmentation)..."
python src_cnn_v3/create_cnn_dataset_v3.py
if [ $? -ne 0 ]; then echo "Step 3 Failed"; exit 1; fi

echo "========================================================"
echo "V3 Data Pipeline Finished Successfully on $(date)"
echo "Summary written to: Output_CNN-LSTM_V3/.../experiment_summary.txt"
echo "========================================================"