# main.py
"""Main script to run the airflow prediction experiment."""

import os
import pandas as pd
import joblib
import numpy as np
import time

# Import project modules
import config
import data_utils
import feature_engineering
import modeling
import plotting

# Import sklearn components
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, make_scorer)
from sklearn.exceptions import FitFailedWarning
import warnings

# Filter warnings related to fitting failures in CV (can happen with tiny folds)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not figure out the number of classes*")

def run_experiment():
    """Loads data, extracts CNN features, trains models via CV, evaluates, and saves the best model."""

    print("--- Starting Airflow Prediction Experiment (CNN Features) ---")

    # Check TensorFlow availability early
    if not feature_engineering.TF_AVAILABLE:
        print("Error: TensorFlow is required for CNN feature extraction but not found/imported. Exiting.")
        return

    # 1. Load Raw Data
    all_raw_data = data_utils.load_raw_data(config.DATASET_FOLDER)
    if not all_raw_data:
        print("Error: No data loaded. Exiting.")
        return

    # 2. Load CNN Base Model (Load once)
    print("\n--- Loading CNN Base Model ---")
    # Define expected input shape for the CNN base
    cnn_input_shape = config.CNN_INPUT_SIZE + (3,) # Add 3 channels
    cnn_base_model = feature_engineering.load_cnn_base(input_shape=cnn_input_shape)
    if cnn_base_model is None:
        print("Error: Failed to load CNN base model. Exiting.")
        return

    # 3. Extract CNN Features
    feature_list = []
    print("\n--- Extracting CNN Features ---")
    start_time = time.time()
    for i, sample in enumerate(all_raw_data):
        print(f"Extracting features for sample {i+1}/{len(all_raw_data)}: {os.path.basename(sample['filepath'])}")
        cnn_features_vector = feature_engineering.extract_cnn_features(
            sample["frames"],
            cnn_base_model,
            target_size=config.CNN_INPUT_SIZE
        )

        if cnn_features_vector is not None:
            # Combine with delta_T and label
            combined_features = {
                # Create feature names dynamically or use a prefix
                **{f"cnn_{j}": val for j, val in enumerate(cnn_features_vector)},
                "delta_T": sample["delta_T"],
                "airflow_rate": sample["airflow_rate"]
            }
            feature_list.append(combined_features)
        else:
            print(f"  Skipping sample {i+1} due to feature extraction error.")

    end_time = time.time()
    print(f"CNN Feature extraction took: {end_time - start_time:.2f} seconds.")

    if not feature_list:
        print("Error: No features could be extracted. Exiting.")
        return

    # 4. Create DataFrame and Prepare X, y
    df = pd.DataFrame(feature_list)
    print("\n--- Feature DataFrame Head (CNN Features) ---")
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")

    cols_all_nan = df.columns[df.isnull().all()]
    if not cols_all_nan.empty:
        print(f"\nWarning: Columns with all NaN values found and will be dropped: {list(cols_all_nan)}")
        df = df.drop(columns=cols_all_nan)

    if "airflow_rate" not in df.columns:
         print("Error: 'airflow_rate' column not found in DataFrame.")
         return

    # Check for NaNs that might cause issues later (SimpleImputer should handle them)
    if df.isnull().any().any():
        print("\nWarning: NaN values detected in DataFrame before imputation.")
        # print(df.isnull().sum()) # Uncomment for detailed counts

    le = LabelEncoder()
    df["airflow_rate_encoded"] = le.fit_transform(df["airflow_rate"].astype(str))
    X = df.drop(columns=["airflow_rate"])
    y = df["airflow_rate_encoded"]

    if X.empty or X.shape[1] == 0:
        print("Error: Feature matrix X is empty after processing.")
        return

    print(f"\nFeature Matrix shape (X): {X.shape}")
    print(f"Target Vector shape (y): {y.shape}")
    unique_labels = sorted(y.unique())
    print(f"Unique target labels: {unique_labels}")
    if len(unique_labels) <= 1: print("Error: Only one class found. Cannot perform classification."); return

    # 5. Define Models and Cross-Validation Strategy (Same as before)
    classifiers = modeling.get_classifiers()
    results = {}

    if config.CV_METHOD == 'LeaveOneOut': cv_strategy = LeaveOneOut(); n_splits = len(X)
    elif config.CV_METHOD == 'StratifiedKFold':
        min_class_count = y.value_counts().min(); n_splits = min(config.K_FOLDS, min_class_count)
        if n_splits < 2: print("Warning: Smallest class count < 2. Falling back to LeaveOneOut."); cv_strategy = LeaveOneOut(); n_splits = len(X)
        else:
             if n_splits != config.K_FOLDS: print(f"Warning: Reducing K from {config.K_FOLDS} to {n_splits} due to small class size.")
             cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    else: print("Error: Invalid CV_METHOD. Using LeaveOneOut."); cv_strategy = LeaveOneOut(); n_splits = len(X)
    print(f"\n--- Using {config.CV_METHOD} Cross-Validation ({n_splits} splits) ---")


    # 6. Run Cross-Validation and Evaluate (Same loop structure as before)
    print("\n--- Running Cross-Validation ---")
    for name, model in classifiers.items():
        print(f"Evaluating model: {name}...")
        pipeline = modeling.build_pipeline(model) # Build pipeline with Imputer, Scaler, optional PCA
        all_preds = np.array([-1.0] * len(y)); all_actuals = np.array(y.copy())
        fold_count = 0
        try:
            for train_idx, test_idx in cv_strategy.split(X, y):
                fold_count += 1; X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]; y_train = y.iloc[train_idx]
                if len(test_idx) == 0: continue
                # Handle PCA components check if PCA is enabled in config
                if 'pca' in pipeline.named_steps and isinstance(pipeline.named_steps['pca'].n_components, int):
                     max_components = min(X_train.shape); current_pca_n = pipeline.named_steps['pca'].n_components
                     if current_pca_n > max_components: pipeline.named_steps['pca'].n_components = max_components
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test); all_preds[test_idx] = y_pred
            # Evaluate predictions
            valid_indices = np.where(all_preds != -1.0)[0]
            if len(valid_indices) == 0: print(f"Error: No valid predictions for model {name}."); continue
            eval_preds, eval_actuals = all_preds[valid_indices], all_actuals[valid_indices]
            accuracy = accuracy_score(eval_actuals, eval_preds)
            f1_w = f1_score(eval_actuals, eval_preds, average='weighted', zero_division=0)
            report = classification_report(eval_actuals, eval_preds, labels=unique_labels, target_names=[str(l) for l in unique_labels], zero_division=0)
            cm = confusion_matrix(eval_actuals, eval_preds, labels=unique_labels)
            results[name] = {"accuracy": accuracy, "f1_weighted": f1_w, "report": report, "cm": cm}
            print(f"\n--- Results for {name} ---"); print(f"Accuracy: {accuracy:.4f}"); print(f"F1 Score (Weighted): {f1_w:.4f}"); print("Classification Report:\n", report); print("Confusion Matrix:\n", cm)
        except Exception as e: print(f"Error during CV for {name}: {type(e).__name__} - {e}"); results[name] = None

    # 7. Determine Best Model and Plot CM (Same as before)
    valid_results = {name: res for name, res in results.items() if res is not None}
    if not valid_results: print("\nError: No models completed evaluation."); return
    metric = config.BEST_MODEL_METRIC
    best_model_name = max(valid_results, key=lambda name: valid_results[name][metric])
    print(f"\n--- Best Model based on {metric}: {best_model_name} ---")
    plotting.plot_confusion_matrix(cm=valid_results[best_model_name]['cm'], labels=[str(l) for l in unique_labels], title=f'Confusion Matrix - {best_model_name} (CV)')

    # 8. Train Final Model on All Data (Same as before)
    print(f"\n--- Training final {best_model_name} model on all data ---")
    final_model_instance = modeling.get_classifiers()[best_model_name]; final_pipeline = modeling.build_pipeline(final_model_instance)
    try:
        # Final check for PCA components relative to full dataset if PCA enabled
        if 'pca' in final_pipeline.named_steps and isinstance(final_pipeline.named_steps['pca'].n_components, int):
            max_components = min(X.shape)
            if final_pipeline.named_steps['pca'].n_components > max_components: final_pipeline.named_steps['pca'].n_components = max_components
        final_pipeline.fit(X, y)
        print("Final model training complete.")
        joblib.dump(final_pipeline, config.MODEL_SAVE_PATH) # Save using updated path
        print(f"Final pipeline saved to: {config.MODEL_SAVE_PATH}")
    except Exception as e: print(f"Error during final model training/saving: {type(e).__name__} - {e}")
    print("\n--- Experiment Finished ---")

if __name__ == "__main__":
    run_experiment()
