# main.py
"""Main script to run the airflow prediction regression experiment."""

import os
import pandas as pd
import joblib
import numpy as np
import time
import warnings

# Import project modules
import config
import data_utils
import feature_engineering # Updated feature engineering
import modeling          # Updated modeling (regressors)
import plotting          # Updated plotting
import tuning            # Updated tuning

from modeling import get_regressors # Get regressors now
# Import sklearn components
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict # Use KFold, add cross_val_predict
# Import regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import FitFailedWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*SettingWithCopyWarning*")
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore potential future warnings from libraries


def run_experiment():
    """Loads data, extracts temporal features, tunes hyperparameters for regressors,
       evaluates models using regression metrics, and saves the best regressor."""

    print("--- Starting Airflow Prediction Experiment (Regression) ---")
    print("--- Features: delta_T, mean_abs_temp_grad ---")
    print("--- NOTE: PCA, CNN, Ensembles are DISABLED ---")


    # 1. Load Raw Data (remains the same)
    all_raw_data = data_utils.load_raw_data(config.DATASET_FOLDER)
    # ... (error check) ...

     # 2. Extract Specific Features (Multi-Interval Bright Region Gradient)
    feature_list = []
    # Dynamically determine expected feature names based on config
    expected_grad_features = [
        f"bright_region_grad_{int(i*config.GRADIENT_INTERVAL_SEC)}_{int((i+1)*config.GRADIENT_INTERVAL_SEC)}"
        for i in range(config.MAX_GRADIENT_INTERVALS)
    ]
    print(f"Expecting gradient features: {expected_grad_features}")

    print("\n--- Extracting Multi-Interval Bright Region Temporal Features ---")
    start_time = time.time()
    processed_samples = 0
    skipped_samples = 0

    for i, sample in enumerate(all_raw_data):
        print(f"Processing sample {i+1}/{len(all_raw_data)}: {os.path.basename(sample['filepath'])}")
        # Call the feature extraction function
        interval_features = feature_engineering.extract_bright_region_features(sample["frames"])

        # Store features for this sample, initializing expected ones to NaN
        combined_features = {name: np.nan for name in expected_grad_features}
        combined_features["delta_T"] = sample["delta_T"]
        combined_features["airflow_rate"] = sample["airflow_rate"] # Target

        # Update with actually calculated values (if any)
        if interval_features: # Check if the dict is not empty
             combined_features.update(interval_features)
             feature_list.append(combined_features)
             processed_samples += 1
        else:
             print(f"  Skipping sample {i+1} due to major feature extraction error (empty dict returned).")
             skipped_samples += 1
             # Optionally append NaNs anyway if you want to impute later,
             # but skipping is safer if the cause is unknown.
             # feature_list.append(combined_features)


    end_time = time.time()
    print(f"\nFeature extraction finished. Processed: {processed_samples}, Skipped: {skipped_samples}.")
    print(f"Feature extraction took: {end_time - start_time:.2f} seconds.")

    if not feature_list:
        print("Error: No features could be extracted. Exiting.")
        return

    # 3. Create DataFrame and Prepare X, y for Regression
    df = pd.DataFrame(feature_list)
    print("\n--- Feature DataFrame Head (Multi-Interval Grads + DeltaT) ---")
    print(df.head())
    print(f"\nDataFrame shape before NaN drop: {df.shape}")

    # Define feature columns (delta_T + expected gradient features)
    feature_columns = ["delta_T"] + expected_grad_features

    # Check if all expected columns are present (might not be if extraction failed early)
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Columns missing after feature extraction: {missing_cols}. They will be treated as NaN.")
        # Add missing columns filled with NaN before imputation/dropping
        for col in missing_cols:
             df[col] = np.nan


    # Target variable y
    y = df["airflow_rate"].astype(float)
    # Features X (select defined feature columns)
    X = df[feature_columns]

    # Report NaNs *before* imputation/dropping (imputer will handle them)
    if X.isnull().any().any():
        print("\nWarning: NaN values detected in feature matrix X before imputation:")
        print(X.isnull().sum())
    else:
        print("\nNo NaNs detected in feature matrix X before imputation.")


    print(f"\nFeature matrix X shape: {X.shape}")
    print(f"Target vector y shape: {y.shape}")
    print("X Head:\n", X.head())
    print("\ny Description:\n", y.describe())

    # Check if X or y became empty after potential previous drops (unlikely now)
    if X.empty or y.empty:
         print("Error: Feature matrix X or target vector y is empty. Exiting.")
         return


    # 4. Hyperparameter Tuning (for Regressors)
    print("\n--- Running Hyperparameter Tuning for All Regressors ---")
    # This function now uses regression grids and scoring
    all_best_params, all_best_scores = tuning.run_grid_search_all_models(X, y)

    print("\n--- Tuning Results Summary (Score: neg_mean_squared_error) ---")
    for name in all_best_params:
         score = all_best_scores[name]
         mse = -score if score != -np.inf else np.inf
         rmse = np.sqrt(mse) if mse != np.inf else np.inf
         print(f"{name}: Best Score={score:.4f} (MSE={mse:.4f}, RMSE={rmse:.4f}), Best Params={all_best_params[name]}")


    # 5. Setup Cross-Validation Strategy for Evaluation
    if config.CV_METHOD == 'LeaveOneOut':
        cv_strategy = LeaveOneOut()
        n_splits = len(X)
        cv_name = "LeaveOneOut"
    elif config.CV_METHOD == 'KFold':
        n_splits = min(config.K_FOLDS, len(X)) # Ensure K is not larger than N
        if n_splits < 2:
             print(f"Warning: K_FOLDS ({config.K_FOLDS}) too large or dataset too small ({len(X)}). Falling back to LeaveOneOut.")
             cv_strategy = LeaveOneOut()
             n_splits = len(X)
             cv_name = "LeaveOneOut"
        else:
             cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
             cv_name = f"{n_splits}-Fold KFold"
    else:
        print(f"Warning: Invalid CV_METHOD '{config.CV_METHOD}'. Defaulting to LeaveOneOut.")
        cv_strategy = LeaveOneOut()
        n_splits = len(X)
        cv_name = "LeaveOneOut"


    # 6. Evaluate individual regressors USING TUNED parameters with Cross-Validation
    print(f"\n--- Evaluating Regressors using Best Parameters found during Tuning ---")
    print(f"--- Using {cv_name} Cross-Validation ({n_splits} splits) ---")

    evaluation_results = {}
    all_regressors_to_eval = get_regressors() # Get fresh instances

    for name, model_instance in all_regressors_to_eval.items():
        print(f"\nEvaluating model: {name} with tuned parameters...")

        # Get the best parameters found for this model (cleaning 'model__' prefix)
        tuned_params_raw = all_best_params.get(name, {})
        tuned_params = {k.replace('model__', ''): v for k,v in tuned_params_raw.items()}

        if tuned_params:
             print(f"  Applying Tuned Parameters: {tuned_params}")
             try:
                 model_instance.set_params(**tuned_params)
             except ValueError as e:
                  print(f"  Warning: Could not set parameters {tuned_params} for {name}. Using defaults. Error: {e}")
        else:
             print("  No tuned parameters found or applied. Using default parameters.")

        # Build pipeline WITHOUT PCA (config.PCA_N_COMPONENTS is None)
        pipeline_model = modeling.build_pipeline(model_instance, pca_components=None)

        # Get cross-validated predictions
        try:
            start_cv_time = time.time()
            # Use cross_val_predict to get predictions for each sample when it was in the test set
            y_pred_cv = cross_val_predict(pipeline_model, X, y, cv=cv_strategy, n_jobs=-1)
            end_cv_time = time.time()
            print(f"  CV predictions for {name} took {end_cv_time - start_cv_time:.2f} seconds.")

            # Calculate regression metrics using the CV predictions
            mse = mean_squared_error(y, y_pred_cv)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred_cv)
            r2 = r2_score(y, y_pred_cv)
            # Calculate neg_mse for comparison with tuning score
            neg_mse_eval = -mse

            # Store results
            results_dict = {
                'neg_mean_squared_error': neg_mse_eval, # Store the metric used for comparison
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_pred_cv': y_pred_cv # Store predictions for plotting the best model
            }
            evaluation_results[name] = results_dict

            print(f"--- Results for {name} (Aggregated over {cv_name} folds with Tuned Params) ---")
            print(f"MSE:  {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}")
            print(f"R²:   {r2:.4f}")

        except Exception as e:
            print(f"Error during CV evaluation for {name}: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            evaluation_results[name] = None # Mark as failed

    # 7. Analyze Results and Plot Best Individual Regressor
    valid_results = {name: res for name, res in evaluation_results.items() if res is not None}
    if not valid_results:
        print("\nError: No models completed evaluation successfully.")
        return

    # Find best model based on the score used in tuning/config
    metric_to_optimize = config.BEST_MODEL_METRIC_SCORE # e.g., 'neg_mean_squared_error' or 'r2'
    higher_is_better = config.METRIC_COMPARISON_HIGHER_IS_BETTER

    try:
        if higher_is_better:
            best_model_name = max(valid_results, key=lambda name: valid_results[name][metric_to_optimize])
        else:
            # If using a metric where lower is better (e.g., 'mse' directly)
            best_model_name = min(valid_results, key=lambda name: valid_results[name][metric_to_optimize])

        print(f"\n--- Best Individual Regressor based on CV {metric_to_optimize}: {best_model_name} ---")
        best_result_metrics = valid_results[best_model_name]
        print(f"  Optimization Score ({metric_to_optimize}): {best_result_metrics[metric_to_optimize]:.4f}")
        print(f"  MSE:  {best_result_metrics['mse']:.4f}")
        print(f"  RMSE: {best_result_metrics['rmse']:.4f}")
        print(f"  MAE:  {best_result_metrics['mae']:.4f}")
        print(f"  R²:   {best_result_metrics['r2']:.4f}")


        # Plot actual vs predicted for the best model using CV predictions
        plotting.plot_actual_vs_predicted(y_true=y, # Original y
                                         y_pred=best_result_metrics['y_pred_cv'], # CV predictions
                                         title=f'Actual vs. Predicted - {best_model_name} (Tuned, CV)',
                                         save_path=config.PLOT_SAVE_PATH)

    except Exception as e:
        print(f"Error determining or plotting best model: {type(e).__name__} - {e}")


    # 8. Final Model Training and Saving (Optional)
    print(f"\n--- Training final {best_model_name} regressor (tuned) on all data ---")
    try:
        # Get a fresh instance and apply best params
        final_model_instance = get_regressors()[best_model_name]
        best_params_raw = all_best_params.get(best_model_name, {})
        best_params_final = {k.replace('model__', ''): v for k,v in best_params_raw.items()}

        if best_params_final:
            print(f"  Applying Tuned Parameters: {best_params_final}")
            final_model_instance.set_params(**best_params_final)

        # Build final pipeline
        final_pipeline = modeling.build_pipeline(final_model_instance, pca_components=None)
        final_pipeline.fit(X, y)
        print("Final model training complete.")

        # Save the final pipeline
        joblib.dump(final_pipeline, config.MODEL_SAVE_PATH)
        print(f"Final {best_model_name} (tuned) pipeline saved to: {config.MODEL_SAVE_PATH}")

    except Exception as e:
        print(f"Error during final model training/saving: {type(e).__name__} - {e}")


    print("\n--- Experiment Finished ---")

if __name__ == "__main__":
    run_experiment()
