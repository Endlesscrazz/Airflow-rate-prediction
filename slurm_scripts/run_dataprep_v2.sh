#!/bin/bash

#==============================================================================
# SBATCH Directives for V2 Data Preparation (Universal)
#==============================================================================
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --nodelist=notch448           # dedicated owner GPU node
#SBATCH --job-name=airflow-v2-tune
#SBATCH --output=logs/v2_dataprep-%j.out
#SBATCH --time=2-00:00:00             # 2 days (max wall time)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1527145@utah.edu

#==============================================================================
# Environment Setup
#==============================================================================
PROJECT_DIR="/scratch/general/vast/u1527145/Airflow-rate-prediction"
CONDA_ENV_NAME="airflow_cnn_env"

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job starting on $(date)"
echo "Running on host: $(hostname)"
echo "========================================================"

cd "$PROJECT_DIR" || exit 1
module purge; module load miniconda3/23.11.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"
mkdir -p logs

#==============================================================================
# Run the V2 Data Preparation Scripts in Order
#==============================================================================
echo "--- STEP 1: Creating Lightweight Master Metadata ---"
python -m src_cnn_v2.create_metadata_v2

echo "--- STEP 2: Splitting Data (using seed from config) ---"
python -m src_cnn_v2.split_data_v2

echo "--- STEP 3: Creating/Updating Cropped & Augmented Dataset ---"
echo "--- (This will skip creating .npy files that already exist) ---"
python -m src_cnn_v2.create_cnn_dataset_v2

echo "========================================================"
echo "Data preparation job finished on $(date)"
echo "========================================================"