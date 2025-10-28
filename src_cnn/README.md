# Airflow Rate Prediction from Thermal Videos

## 1. Project Overview

This project aims to predict quantitative airflow leakage rates by analyzing thermal infrared (IR) video sequences. The primary challenge lies in building a model that can generalize across different building materials (e.g., gypsum, hardyboard), which exhibit distinct thermal properties.

We employ a multi-modal deep learning approach that combines a **CNN-LSTM architecture** for automatic feature extraction from video data with a **Multi-Layer Perceptron (MLP)** that processes handcrafted physical and statistical features. This hybrid model is designed to learn both the spatial patterns ("what a leak looks like") and the temporal dynamics ("how a leak evolves over time") while being guided by essential contextual information like material type and temperature difference.

The entire pipeline is designed to be robust, reproducible, and data-driven, featuring automated feature selection, leakage-free cross-validation, and strong regularization techniques to combat overfitting on our small and diverse dataset.

## 2. CNN-LSTM Model Architecture

The core of our deep learning approach is the `UltimateHybridRegressor`, a multi-modal neural network with three parallel input branches that converge into a final prediction head.

### Branch 1: Spatio-Temporal Visual Branch (CNN-LSTM)
This branch learns features directly from the video pixels.
-   **Time-Distributed CNN (ResNet-18):** A pre-trained ResNet-18 acts as a powerful feature extractor. It processes each frame of the 2-channel (Thermal + Mask) video sequence independently, converting each frame into a high-level feature vector that describes the spatial characteristics of the hotspot.
-   **LSTM with Attention:** The sequence of feature vectors from the CNN is fed into a Long Short-Term Memory (LSTM) network. The LSTM learns the temporal patterns, such as the rate of cooling and changes in hotspot shape. An **Attention mechanism** is applied to the LSTM's output, allowing the model to dynamically weigh the importance of each frame in the sequence when making a prediction.

### Branch 2: Contextual Features Branch (MLP)
This branch provides the model with essential, high-level context.
-   **Input:** Low-dimensional, critical features like `delta_T_log` and one-hot encoded `material_type`.
-   **Network:** A small MLP processes these features to create a contextual embedding.

### Branch 3: Dynamic Features Branch (MLP)
This branch leverages our domain expertise by incorporating a curated set of handcrafted features.
-   **Input:** A set of automatically selected, statistically relevant features (e.g., `hotspot_area_log`, `temperature_kurtosis`, `rate_of_std_change_initial`) that describe the hotspot's geometry, texture, and temporal evolution.
-   **Network:** A deeper MLP processes these features to learn complex interactions between them.

### Final Prediction Head
The outputs from all three branches are concatenated and passed through a final MLP "head" with aggressive dropout for regularization. This head learns to combine the visual, contextual, and statistical information to produce the final single regression output for the airflow rate.

## 3. Project Workflow & How to Run

This project follows a structured, multi-step workflow. Execute the scripts from the project's root directory.

### Phase 1: Data Preparation & Feature Selection

**Step 1: Generate Master Feature Bank**
This script calculates all possible handcrafted features for every sample in your ground truth CSV and saves them to a master file.

```bash
python -m scripts.generate_master_features
Input: Raw videos/masks and the GROUND_TRUTH_CSV_PATH defined in src_cnn/config.py.
Output: A master_features.csv file at the MASTER_FEATURES_PATH location in config.py.
Step 2: Run Automated Feature Selection
This script analyzes the master feature file to find the most predictive features for the model.

code
Bash
python -m scripts.select_features --k 15
Input: master_features.csv.
Output: A Python list of the top k feature names printed to the console.
Step 3: Update Configuration
Manually copy the list of feature names from the previous step and paste it into the DYNAMIC_FEATURES list in src_cnn/config.py. This tells the rest of the pipeline which features to use.

Phase 2: CNN Dataset Creation
Step 4: Create the CNN Dataset
This script processes the video sequences and merges them with the selected features to create the final dataset for the deep learning model.

code
Bash
python -m scripts.create_dataset_CNN --type thermal_masked
Input: master_features.csv, raw videos/masks.
Output: A CNN_dataset/.../ folder containing .npy video files and a metadata.csv.
Phase 3: Model Training & Evaluation
Step 5: Split the Data
Create the training and hold-out sets using a stratified group split to prevent data leakage.

code
Bash
python -m scripts.split_data --input_csv path/to/your/metadata.csv
Input: The metadata.csv generated in Step 4.
Output: train_metadata.csv and holdout_metadata.csv in the same directory.
Step 6: Train Cross-Validation Models
Run the 5-fold cross-validation training. This can be done locally or using the provided SLURM script on an HPC cluster.

Local Example (running fold 0):

code
Bash
python -m scripts.train --fold 0 --total_folds 5 --model_type lstm --dataset_dir path/to/your/CNN_dataset --in_channels 2
Output: 5 trained models (.pth) and 5 scalers (.pkl) in a trained_models..._CV directory.
Step 7: Train Final Model (Optional)
Train a single model on the entire training set for final deployment.

code
Bash
python -m scripts.train_final_model --model_type lstm --dataset_dir path/to/your/CNN_dataset --in_channels 2
Output: A single final model and scaler in the trained_models_final directory.
Step 8: Evaluate on Hold-Out Set
Evaluate the performance of your trained models on the unseen hold-out data.

To evaluate the more robust Ensemble Model (recommended):

code
Bash
python -m scripts.evaluate_holdout --model_type lstm --dataset_dir path/to/your/CNN_dataset --in_channels 2 --optuna_tuned --ensemble
To evaluate the Single Final Model:

code
Bash
python -m scripts.evaluate_holdout --model_type lstm --dataset_dir path/to/your/CNN_dataset --in_channels 2 --optuna_tuned
Output: Performance metrics (R², RMSE, MAE) printed to the console and a True vs. Predicted plot saved in the holdout_results directory.