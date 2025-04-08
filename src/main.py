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
import tuning # Tuning disabled for initial simplified run

from modeling import get_classifiers #, build_ensemble_pipeline # Ensemble disabled for initial run
# Import sklearn components
from sklearn.model_selection import LeaveOneOut, StratifiedKFold # cross_val_score # Not needed for individual evaluation loop style
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

# Filter warnings related to fitting failures in CV (can happen with tiny folds)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not figure out the number of classes*")


def run_experiment():
    """Loads data, extracts features, evaluates individual models,
    and identifies the best performing one based on CV.""" # Simplified goal

    print("--- Starting Airflow Prediction Experiment (with CV) ---")
    print("--- NOTE: CNN Features, PCA, RFECV, and Ensemble are DISABLED for this run ---")

    # Check TensorFlow availability early (though not used in this config)
    # Optional: Keep check for informational purposes if user might re-enable later
    if not feature_engineering.TF_AVAILABLE:
        print("Info: TensorFlow not found/imported. CNN feature extraction would fail if enabled.")
        # return # Don't exit if only doing handcrafted

    # 1. Load Raw Data
    all_raw_data = data_utils.load_raw_data(config.DATASET_FOLDER)
    if not all_raw_data:
        print("Error: No data loaded. Exiting.")
        return

    # 2. Load CNN Base Model (Optional - not used in feature extraction below)
    # print("\n--- Loading CNN Base Model (INFO ONLY - Not used for features in this run) ---")
    # cnn_input_shape = config.CNN_INPUT_SIZE + (3,)  # Add 3 channels
    # cnn_base_model = feature_engineering.load_cnn_base(input_shape=cnn_input_shape)
    # if cnn_base_model is None and feature_engineering.TF_AVAILABLE: # Check only if TF should be working
    #     print("Error: Failed to load CNN base model even though TF seems available.")
        # return # Don't exit if only doing handcrafted
    cnn_base_model = None # Ensure it's None if not loaded/used

    # 3. Extract Features (Handcrafted Only for this run)
    feature_list = []
    print("\n--- Extracting Handcrafted Features Only ---")
    start_time = time.time()
    for i, sample in enumerate(all_raw_data):
        print(f"Extracting features for sample {i+1}/{len(all_raw_data)}: {os.path.basename(sample['filepath'])}")

        # --- CNN Feature Extraction DISABLED ---
        # cnn_features_vector = feature_engineering.extract_cnn_features(
        #     sample["frames"],
        #     cnn_base_model,
        #     target_size=config.CNN_INPUT_SIZE
        # )
        cnn_features_vector = None # Explicitly None

        handcrafted_feats = feature_engineering.extract_handcrafted_features(sample["frames"])

        # Combine handcrafted features with delta_T and label.
        # if cnn_features_vector is not None and handcrafted_feats is not None: # Original condition
        if handcrafted_feats is not None: # Use only handcrafted check
            combined_features = {
                # **{f"cnn_{j}": val for j, val in enumerate(cnn_features_vector)}, # CNN Features DISABLED
                **handcrafted_feats,
                "delta_T": sample["delta_T"],
                "airflow_rate": sample["airflow_rate"]
            }
            feature_list.append(combined_features)
        else:
            print(f"  Skipping sample {i+1} due to handcrafted feature extraction error.") # Adjusted message

    end_time = time.time()
    print(f"Feature extraction took: {end_time - start_time:.2f} seconds.")

    if not feature_list:
        print("Error: No features could be extracted. Exiting.")
        return

    # 4. Create DataFrame and Prepare X, y
    df = pd.DataFrame(feature_list)
    print("\n--- Feature DataFrame Head (Handcrafted + DeltaT) ---")
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")

    # Drop columns with all NaN values
    cols_all_nan = df.columns[df.isnull().all()]
    if not cols_all_nan.empty:
        print(f"\nWarning: Dropping columns with all NaN values: {list(cols_all_nan)}")
        df = df.drop(columns=cols_all_nan)

    if "airflow_rate" not in df.columns:
         print("Error: 'airflow_rate' column not found in DataFrame.")
         return

    if df.isnull().any().any():
        print("\nWarning: NaN values detected in DataFrame before imputation (will be handled by pipeline).")

    le = LabelEncoder()
    df["airflow_rate_encoded"] = le.fit_transform(df["airflow_rate"].astype(str))
    # Save label mapping for later use if needed (e.g., prediction interpretation)
    label_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    print("\nLabel Encoding Mapping:", label_mapping)

    # Remove label columns from features to avoid leakage
    X = df.drop(columns=["airflow_rate", "airflow_rate_encoded"])
    y = df["airflow_rate_encoded"]

    #5. Hyperparameter Tuning (Run for all models and store results)
    print("\n--- Running Hyperparameter Tuning for All Models ---")
    all_classifiers_for_tuning = get_classifiers()
    best_params_all_models = {}
    best_scores_all_models = {}

    for name in all_classifiers_for_tuning.keys():
        print(f"\n--- Running Grid Search for {name} ---")
        # Pass the specific classifier name to the tuning function
        best_params, best_score = tuning.run_grid_search(X, y, classifier_name=name)
        # Store results, removing the 'model__' prefix from keys for direct use
        best_params_cleaned = {k.replace('model__', ''): v for k, v in best_params.items()}
        best_params_all_models[name] = best_params_cleaned
        best_scores_all_models[name] = best_score

    print("\n--- Tuning Results Summary ---")
    for name in best_params_all_models:
         print(f"{name}: Best Score = {best_scores_all_models[name]:.4f}, Best Params = {best_params_all_models[name]}")

    # --- CV Strategy Setup (Same as before) ---
    if config.CV_METHOD == 'LeaveOneOut':
        cv_strategy = LeaveOneOut()
        n_splits = len(X) # Should be 22
    elif config.CV_METHOD == 'StratifiedKFold':
        min_class_count = y.value_counts().min()
        n_splits = max(2, min(config.K_FOLDS, min_class_count))
        if n_splits < 2:
             print(f"Error: Cannot perform StratifiedKFold with smallest class count ({min_class_count}) < 2. Exiting.")
             return
        else:
             if n_splits != config.K_FOLDS:
                 print(f"Warning: Reducing K from {config.K_FOLDS} to {n_splits} due to small class size ({min_class_count}).")
             cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    else:
        print(f"Warning: Invalid CV_METHOD '{config.CV_METHOD}'. Defaulting to LeaveOneOut.")
        cv_strategy = LeaveOneOut()
        n_splits = len(X)
    # --- End CV Setup ---

    # 6. Evaluate individual models USING TUNED parameters
    print(f"\n--- Evaluating Models using Best Parameters found during Tuning ---")
    print(f"--- Using {config.CV_METHOD} Cross-Validation ({n_splits} splits) ---")

    results = {}
    all_classifiers_to_eval = get_classifiers() # Get fresh instances

    for name, model_instance in all_classifiers_to_eval.items():
        print(f"\nEvaluating model: {name} with tuned parameters...")

        # Get the best parameters found for this model
        tuned_params = best_params_all_models.get(name, {}) # Get params or empty dict if none found

        if tuned_params:
             print(f"  Applying Tuned Parameters: {tuned_params}")
             try:
                 # Apply the parameters to the model instance
                 model_instance.set_params(**tuned_params)
             except ValueError as e:
                  print(f"  Warning: Could not set parameters {tuned_params} for {name}. Using defaults. Error: {e}")
        else:
             print("  No tuned parameters found or applied. Using default parameters.")


        # Build pipeline WITHOUT PCA, using the (potentially updated) model_instance
        pipeline_model = modeling.build_pipeline(model_instance, pca_components=None)

        
        # if name == "SVC":
        #     from sklearn.feature_selection import SelectKBest, f_classif
        #     print("  Adding SelectKBest(k=12) for SVC pipeline") # Experiment with k
        #     pipeline_model = Pipeline([
        #         ('imputer', SimpleImputer(strategy='mean')),
        #         ('selector', SelectKBest(score_func=f_classif, k=12)), # Adjust k
        #         ('scaler', StandardScaler()),
        #         ('model', model_instance) # model_instance already has tuned params
        #     ])
        # else:
        #     # Build standard pipeline for other models
        #     pipeline_model = modeling.build_pipeline(model_instance, pca_components=None)

        # --- CV Evaluation Loop (Same as before) ---
        all_preds = np.array([-1] * len(y)) # Initialize with placeholder
        all_actuals = np.array(y.copy())
        fold_count = 0
        try:
            start_cv_time = time.time()
            for train_idx, test_idx in cv_strategy.split(X, y):
                fold_count += 1
                if len(test_idx) == 0: continue
                if len(train_idx) == 0: continue

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train = y.iloc[train_idx] # y_test not needed here, use all_actuals

                if len(np.unique(y_train)) < len(label_mapping):
                     # Suppress this warning for LOO as it's expected
                     if not isinstance(cv_strategy, LeaveOneOut):
                          print(f"  Warning: Fold {fold_count} - Training data missing classes.")

                pipeline_model.fit(X_train, y_train)
                y_pred = pipeline_model.predict(X_test)
                all_preds[test_idx] = y_pred

            end_cv_time = time.time()
            print(f"  CV for {name} took {end_cv_time - start_cv_time:.2f} seconds.")

            valid_indices = np.where(all_preds != -1)[0]
            if len(valid_indices) < len(y):
                 print(f"  Warning: Only {len(valid_indices)}/{len(y)} samples predicted.")
            if len(valid_indices) == 0:
                print(f"  Error: No valid predictions collected for model {name}.")
                results[name] = None
                continue

            eval_preds = all_preds[valid_indices]
            eval_actuals = all_actuals[valid_indices]

            accuracy = accuracy_score(eval_actuals, eval_preds)
            f1_w = f1_score(eval_actuals, eval_preds, average='weighted', zero_division=0)
            report = classification_report(eval_actuals, eval_preds, zero_division=0,
                                            labels=np.arange(len(label_mapping)),
                                            target_names=[str(label_mapping[i]) for i in np.arange(len(label_mapping))])
            cm = confusion_matrix(eval_actuals, eval_preds, labels=np.arange(len(label_mapping)))

            results[name] = {"accuracy": accuracy, "f1_weighted": f1_w, "report": report, "cm": cm}

            print(f"--- Results for {name} (Aggregated over CV folds with Tuned Params) ---")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score (Weighted): {f1_w:.4f}")
            print("Classification Report:\n", report)
            print("Confusion Matrix:\n", cm)

        except Exception as e:
            print(f"Error during CV evaluation for {name}: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            results[name] = None
        # --- End CV Evaluation Loop ---


    # 7. Analyze Results and Plot Best Individual Model (Now using tuned results)
    valid_results = {name: res for name, res in results.items() if res is not None}
    if not valid_results:
        print("\nError: No models completed evaluation successfully.")
        return

    metric = config.BEST_MODEL_METRIC
    try:
        # Find best model based on the evaluation using TUNED parameters
        best_model_name = max(valid_results, key=lambda name: valid_results[name][metric])
        print(f"\n--- Best Individual Model based on CV {metric} (using tuned parameters): {best_model_name} ---")
        print(f"  {metric}: {valid_results[best_model_name][metric]:.4f}")

        best_cm = valid_results[best_model_name]['cm']
        plot_labels = [str(label_mapping[i]) for i in np.arange(len(label_mapping))]
        # Update plot save path to reflect tuning
        plot_save_path = config.PLOT_SAVE_PATH.replace(".png", "_tuned.png")
        plotting.plot_confusion_matrix(cm=best_cm,
                                       labels=plot_labels,
                                       title=f'Confusion Matrix - {best_model_name} (Tuned, CV Aggregated)',
                                       save_path=plot_save_path) # Pass save path if function supports it

    except Exception as e:
        print(f"Error determining or plotting best model: {e}")

    # --- Final Model Training and Saving (Optional - consider saving the best tuned model) ---
    print(f"\n--- Skipping final model training and saving for now ---")
    # If you want to save the best *tuned* model:
    # print(f"\n--- Training final {best_model_name} model (tuned) on all data ---")
    # try:
    #     final_model_instance = get_classifiers()[best_model_name]
    #     tuned_params = best_params_all_models.get(best_model_name, {})
    #     if tuned_params:
    #         print(f"  Applying Tuned Parameters: {tuned_params}")
    #         final_model_instance.set_params(**tuned_params)
    #
    #     final_pipeline = modeling.build_pipeline(final_model_instance, pca_components=None)
    #     final_pipeline.fit(X, y)
    #     print("Final model training complete.")
    #     save_path = f"final_{best_model_name}_tuned_pipeline.joblib"
    #     joblib.dump(final_pipeline, save_path)
    #     print(f"Final {best_model_name} (tuned) pipeline saved to: {save_path}")
    # except Exception as e:
    #     print(f"Error during final model training/saving: {type(e).__name__} - {e}")


    print("\n--- Experiment Finished ---")

if __name__ == "__main__":
    run_experiment()