# Predicting Airflow Rate from Thermal Videos using Deep Learning

This project tackles the challenging task of predicting airflow leakage rates by analyzing thermal infrared (IR) videos. The primary goal is to develop a model that can learn the relationship between the spatio-temporal patterns of a thermal plume and the quantitative airflow rate.

The project began with a traditional machine learning approach using handcrafted features, which ultimately proved insufficient due to the high noise and low diversity of the dataset. The strategy was successfully pivoted to a Deep Learning approach, culminating in a hybrid **CNN-LSTM model** that can generalize to unseen data.

## 📈 Key Findings & Best Results

After extensive experimentation, the optimal approach was identified and validated:

*   **Best Model:** A hybrid **CNN-LSTM with Attention** (`UltimateHybridRegressor`).
*   **Best Data Representation:** **1-Channel Raw Thermal Video Sequences**. This surprisingly outperformed more complex representations like optical flow, indicating that the raw temperature evolution contains the most valuable signal.
*   **Cross-Validation Performance:** The model achieved a stable average **R² of 0.4382 ± 0.1192** across a 5-fold group cross-validation.
*   **Hold-Out Set Performance:** The final model successfully generalized to a completely unseen hold-out set, achieving a **final R² of 0.5254**. This positive result on unseen data is the key success of this project.


*Figure 1: The final model's predictions on the unseen hold-out set, demonstrating a clear positive correlation and successful generalization.*

## 📂 Project Structure

The repository is organized into the two main phases of the project. The successful deep learning pipeline is contained within `src_cnn`.

```
.
├── CNN_dataset/                # Processed datasets for the CNN models reside here (not checked into git)
├── src_cnn/
│   ├── cnn_models.py           # PyTorch model architectures (CNN-LSTM, CNN-Average)
│   ├── cnn_utils.py            # Custom PyTorch Dataset class
│   ├── create_dataset.py       # Unified script to generate thermal, flow, or hybrid datasets
│   ├── train_cnn.py            # Main script for running cross-validation
│   ├── train_final_model.py    # Script to train the final model on all dev data
│   ├── evaluate_holdout.py     # Script to evaluate the final model on the hold-out set
│   ├── hyperparam_search.py    # Optuna script for hyperparameter optimization
│   └── agg_results.py          # Utility to aggregate CV results
│
├── src_feature_based/          # Code from the initial handcrafted feature engineering phase
├── visualizations/             # Output directory for GIF visualizations
├── trained_models_final/       # Output directory for final trained model weights
├── results_..._CV/             # Output directories for cross-validation results
└── ...
```

## 🔬 Methodology & Workflow

The project followed a systematic, data-driven workflow to overcome initial challenges.

### Phase 1: Handcrafted Feature Engineering

The project began by extracting over 20 summary-statistic features (e.g., hotspot area, temperature change rate) from the videos. While rigorously evaluated with Nested Cross-Validation, this approach consistently suffered from extreme overfitting (Training R² ≈ 0.9 vs. Validation R² ≈ 0.3), demonstrating that these features were not robust enough to capture the complex underlying physics.

### Phase 2: Pivot to Deep Learning

Based on the limitations of Phase 1, the strategy was pivoted to a deep learning approach to allow the model to learn features directly from pixel data.

1.  **Data Representation Experiments:** A unified pipeline (`create_dataset.py`) was built to generate datasets of different representations. Comparative experiments proved that **1-Channel Thermal video** was a superior input to 2-Channel Optical Flow, which had previously failed to generalize (Hold-Out R² of -0.05).

2.  **Model Architecture Validation:** A powerful CNN-LSTM model with Attention was implemented. A diagnostic test comparing it to a simpler "CNN-Average" model (which produced a CV R² of -0.82) conclusively proved that **modeling the temporal sequence with an LSTM is critical for this task.**

3.  **Hyperparameter Optimization with Optuna:** An automated hyperparameter search (`hyperparam_search.py`) was conducted using Optuna to find the optimal model architecture and training parameters. The search revealed that a larger model with stronger regularization (`weight_decay`) was optimal, improving the potential R² on a single validation fold to **0.626**.

4.  **Final Validation:** The best hyperparameters from the Optuna search were validated with a full 5-fold cross-validation, confirming their robustness and leading to the final successful model.

## 🚀 Reproducibility Guide

Follow these steps to set up the environment, process data, and reproduce the results.

### 1. Setup Environment

First, clone the repository and set up the Python environment.

```bash
# Clone the repository
git clone https://github.com/Endlesscrazz/Airflow-rate-prediction.git
cd Airflow-rate-prediction

# Create and activate a conda environment
conda create -n airflow_env python=3.9
conda activate airflow_env

# Install required packages
pip install -r requirements.txt
```

### 2. Data Preparation

The raw `.mat` video files and `.npy` mask files are not included in this repository. You must place them in a directory structure as expected by the `src_feature_based/config.py` file.

Once the raw data is in place, generate the 1-channel thermal dataset, which was found to be the best performer.

```bash
# This will create the dataset in CNN_dataset/dataset_1ch_thermal/
python -m src_cnn.create_dataset --type thermal
```

### 3. Create Train / Hold-Out Split

Before training, you need to split the full metadata file into a training/development set and a final hold-out set.

```bash
# This script is located in the src_feature_based directory
# It will create train_metadata.csv and holdout_metadata.csv inside the dataset folder
python -m src_feature_based.split_data --metadata_path CNN_dataset/dataset_1ch_thermal/metadata.csv
```

### 4. Run Cross-Validation (with Optuna-Tuned Hyperparameters)

To verify the model's performance, run the 5-fold cross-validation. The `train_cnn.py` script has been pre-configured with the best hyperparameters found by the Optuna search.

*Note: This is best run on a machine with a GPU. The following command is an example for running a single fold locally. For a full 5-fold run, use the `run_chpc_job.sh` script on an HPC cluster or loop through folds 0-4.*

```bash
# Run a single fold (e.g., fold 0) of the cross-validation
python -m src_cnn.train_cnn \
    --fold 0 \
    --total_folds 5 \
    --model_type "lstm" \
    --dataset_dir "CNN_dataset/dataset_1ch_thermal" \
    --in_channels 1
```

After running all 5 folds, aggregate the results:
```bash
python -m src_cnn.agg_results --model_type lstm --in_channels 1
```

### 5. Train Final Model

After cross-validation, train the final model on the entire development set (`train_metadata.csv`).

```bash
# This will save the final model to trained_models_final/final_model_lstm_1ch_optuna.pth
python -m src_cnn.train_final_model \
    --model_type "lstm" \
    --dataset_dir "CNN_dataset/dataset_1ch_thermal" \
    --in_channels 1
```

### 6. Evaluate on Hold-Out Set

Finally, evaluate the performance of the trained model on the unseen hold-out set to get the final reported R² score.

```bash
# This command loads the optuna-tuned model and evaluates it
python -m src_cnn.evaluate_holdout \
    --model_type "lstm" \
    --dataset_dir "CNN_dataset/dataset_1ch_thermal" \
    --in_channels 1 \
    --optuna_tuned
```

## 🔮 Future Work

While the current model is successful, several avenues for further improvement exist:
*   **Input Sequence Analysis:** Experiment with shorter, more information-dense video clips (e.g., the first 5 seconds) to see if it provides a cleaner signal.
*   **Advanced Architectures:** Explore 3D-CNNs (e.g., R3D_18), which are designed to process spatio-temporal data jointly and may capture dynamics more effectively.
*   **Data Acquisition:** The most significant improvements will likely come from acquiring more diverse training data across a wider range of materials, `delta_T` conditions, and airflow rates.