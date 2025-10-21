#!/bin/bash
#==============================================================================
# SLURM Directives – YQU GPU node on Notchpeak
#==============================================================================

#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --nodelist=notch448           # dedicated owner GPU node
#SBATCH --job-name=airflow-v2-tune
#SBATCH --output=logs/v2_tuning-%j.out
#SBATCH --time=2-00:00:00             # 2 days (max wall time)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

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
python src_cnn_v2/hyperparam_search_v2.py \
  --n_trials 200 \
  --study_name "gypsum-10-hole-dataset-dataset-cs16-v2-tuning" \
  --storage "sqlite:///optuna_gypsum_v2.db"

echo "--- Search Complete ---"