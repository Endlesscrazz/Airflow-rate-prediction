#!/bin/bash

#==============================================================================
# SBATCH Directives for FINAL MODEL training
#==============================================================================
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=airflow-cnn-final
#SBATCH --output=logs/cnn_final_%j.out      # Use %j for unique job ID
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1527145@utah.edu

#==============================================================================
# --- Final Model: LSTM on 1-ch Thermal (10s focus, Optuna-tuned) ---
# MODEL_TYPE="lstm"
# DATASET_DIR="CNN_dataset/dataset_1ch_thermal_hard_crop"
# IN_CHANNELS=1

# Example: Final Model for 2-ch Thermal+Mask ---
MODEL_TYPE="lstm"
DATASET_DIR="CNN_dataset/dataset_2ch_thermal_masked_f10s"
IN_CHANNELS=2

# Example: Final Model for 2-ch Thermal+Mask (one faling video removed from dataset)
MODEL_TYPE="lstm"
DATASET_DIR="CNN_dataset/dataset_2ch_thermal_masked"
IN_CHANNELS=2


#==============================================================================

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
echo "---"
echo "FINAL MODEL CONFIG:"
echo "  - Model Type: ${MODEL_TYPE}"
echo "  - Dataset: ${DATASET_DIR}"
echo "  - Input Channels: ${IN_CHANNELS}"
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
# Run the Training Script
#==============================================================================
echo "--- Starting FINAL Model Training on ALL Development Data ---"

python -m src_cnn.train_final_model \
    --model_type "${MODEL_TYPE}" \
    --dataset_dir "${PROJECT_DIR}/${DATASET_DIR}" \
    --in_channels ${IN_CHANNELS}


echo "========================================================"
echo "Job finished on $(date)"
echo "========================================================"