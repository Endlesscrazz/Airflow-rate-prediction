#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=airflow-v2-tune
#SBATCH --output=logs/v2_tuning_%j.out
#SBATCH --time=12:00:00  # Request a good amount of time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# --- Environment Setup (same as your train script) ---
PROJECT_DIR="/scratch/general/vast/u1527145/Airflow-rate-prediction"
CONDA_ENV_NAME="airflow_cnn_env"

cd "$PROJECT_DIR" || exit 1
module purge
module load miniconda3/23.11.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"
mkdir -p logs

echo "--- Starting Hyperparameter Search ---"

# Run the search with 100 trials
python src_cnn_v2/hyperparam_search_v2.py --n_trials 100 --study_name "gypsum-v2-tuning"

echo "--- Search Complete ---"