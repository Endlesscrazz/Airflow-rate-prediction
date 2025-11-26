#!/bin/bash
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --job-name=hp-brick
#SBATCH --output=hyperparam-logs/brick_tuning-%j.out
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# --- Environment Setup ---
PROJECT_DIR="/scratch/general/vast/u1527145/Airflow-rate-prediction"
CONDA_ENV_NAME="airflow_cnn_env"

cd "$PROJECT_DIR" || exit 1
module purge
module load miniconda3/23.11.0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"
mkdir -p hyperparam-logs

# --- 1. SET EXPERIMENT TARGET ---
export Target_Experiment="brickcladding_all_dataset_v2"

echo "=================================================="
echo "STARTING JOB: Brick Cladding Hyperparam Search"
echo "Experiment Name: $Target_Experiment"
echo "=================================================="

# --- 2. VERIFY PATHS (Logging) ---
echo "Checking Configuration Paths..."
python -c "import src_cnn_v2.config_v2 as c; print(f'-> READING DATA FROM: {c.DATASET_DIR}'); print(f'-> SAVING DB TO:      {c.EXPERIMENT_RESULTS_DIR}')"

# --- 3. RUN OPTUNA ---
echo "--- Starting Optuna Optimization ---"
python src_cnn_v2/hyperparam_search_v2.py

echo "--- Job Complete ---"