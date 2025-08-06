#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=airflow-optuna
#SBATCH --output=logs/optuna_search_%j.out
#SBATCH --time=12:00:00  # Request significant time for many trials
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

#==============================================================================
# Environment Setup
#==============================================================================
PROJECT_NAME="Airflow-rate-prediction"
CONDA_ENV_NAME="airflow_cnn_env"
PROJECT_DIR="/scratch/general/vast/u1527145/${PROJECT_NAME}"

cd "$PROJECT_DIR" || exit 1

module purge
module load miniconda3/23.11.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

#==============================================================================
# Run the Hyperparameter Search
#==============================================================================
echo "--- Starting Optuna Hyperparameter Search ---"

# We will run 100 trials, using Fold 0 as our validation set.
# Each trial will run for a max of 75 epochs.
python -m src_cnn.hyperparam_search \
    --n_trials 100 \
    --num_epochs 75 \
    --fold 0

echo "--- Search Complete ---"