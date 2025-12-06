#!/bin/bash
#SBATCH --job-name=v3_tuning
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --nodelist=notch448
#SBATCH --output=hyperparm-logs_v3/tune-%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Setup
mkdir -p logs_v3
module purge
module load python/3.11
source airflow_v3/bin/activate


echo "Starting Optuna Search..."

python src_cnn_v3/hyperparam_search_v3.py