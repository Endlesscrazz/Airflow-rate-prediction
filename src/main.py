# --- START OF FILE main.py ---

# main.py
"""Main script to run the airflow prediction regression experiment."""

import os
import pandas as pd
import joblib
import numpy as np
import time
import warnings

# Import project modules
import config           # Uses updated paths and flags
import data_utils
import feature_engineering # Updated with visualization config checks
import modeling          # Updated for NN only
import plotting          # Regression plots
import tuning            # Updated for NN only

from modeling import get_regressors # Will now return only NN
# Import sklearn components
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor # Import to check instance type later
# Import regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*SettingWithCopyWarning*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# warnings.filterwarnings("ignore", category=UserWarning, message=".*Skipping features.*")


def run_experiment():
    """Loads data, extracts frame-based temporal features, visualizes focus area,
       tunes NN hyperparameters, evaluates NN model, analyzes features (limited for NN),
       and saves the best NN regressor."""

    print("--- Starting Airflow Prediction Experiment (Regression) ---")
    print("--- Model: Neural Network (MLPRegressor) ---")
    print("--- Features: delta_T, Bright Region Area, Frame-Based Gradients & Std Devs ---")
    print(f"--- Visualization Saving Enabled: {getattr(config, 'SAVE_FOCUS_AREA_VISUALIZATION', False)} ---")

    # 1. Load Raw Data
    # Consider adding preprocessing calls here if PREPROCESSING flags are True
    # all_raw_data = data_utils.load_raw_data(config.DATASET_FOLDER)
    # if config.PREPROCESSING_APPLY_FILTER:
    #     all_raw_data = data_utils.apply_median_filter(all_raw_data, config.PREPROCESSING_FILTER_KERNEL_SIZE)
    # if config.PREPROCESSING_APPLY_SCALING:
    #     all_raw_data = data_utils.apply_scaling(all_raw_data, config.PREPROCESSING_SCALE_RANGE)
    # (Requires implementing these functions in data_utils.py)
    all_raw_data = data_utils.load_raw_data(config.DATASET_FOLDER)
    if not all_raw_data: print("Error: No data loaded. Exiting."); return

    # 2. Extract Specific Features & Visualize
    feature_list = []

    # --- Define expected FRAME-BASED feature names ---
    interval_frames = getattr(config, 'FRAME_INTERVAL_SIZE', 50)
    max_intervals = getattr(config, 'MAX_FRAME_INTERVALS', 3)
    calculate_std = getattr(config, 'CALCULATE_STD_FEATURES', False)

    expected_grad_features = []
    expected_std_features = []
    for i in range(max_intervals):
        start = i * interval_frames; end = (i + 1) * interval_frames
        expected_grad_features.append(f"hotspot_grad_{start}_{end}")
        if calculate_std:
            expected_std_features.append(f"hotspot_grad_{start}_{end}")

    # --- Use the CORRECT Area Feature Name ---
    expected_area_feature = ["hotspot_area"] 

    # Combine feature names
    expected_feature_names = expected_area_feature + expected_grad_features + expected_std_features
    print(f"\nExpecting features: delta_T, {expected_feature_names}")

    print("\n--- Extracting Features & Potentially Saving Visualizations ---")
    start_time = time.time()
    processed_samples = 0
    skipped_samples = 0

    for i, sample in enumerate(all_raw_data):
        # Pass sample index 'i' for visualization control
        # print(f"Processing sample {i+1}/{len(all_raw_data)}: {os.path.basename(sample.get('filepath', 'Unknown'))}") # Reduced verbosity
        try:
            full_filepath = sample.get('filepath', 'Unknown')
            if full_filepath == 'Unknown': raise ValueError("Filepath missing.")
            mat_filename = os.path.basename(full_filepath)
            mat_filename_no_ext, _ = os.path.splitext(mat_filename)
            source_folder_name = os.path.basename(os.path.dirname(full_filepath))
            if not source_folder_name: source_folder_name = "_root_"
        except Exception as path_e:
            print(f"  Skipping sample {i+1} due to error processing filepath: {path_e}")
            skipped_samples += 1
            continue

        if "frames" not in sample or not isinstance(sample["frames"], np.ndarray) or sample["frames"].ndim != 3:
             print(f"  Skipping sample {i+1} due to missing or invalid 'frames' data.")
             combined_features = {name: np.nan for name in expected_feature_names}
             combined_features["delta_T"] = sample.get("delta_T", np.nan)
             combined_features["airflow_rate"] = sample.get("airflow_rate", np.nan)
             feature_list.append(combined_features)
             skipped_samples += 1
             continue

        # Pass the sample index 'i' to the feature extraction function
        extracted_features_dict = feature_engineering.extract_bright_region_features(
            frames=sample["frames"],
            source_folder_name=source_folder_name,
            mat_filename_no_ext=mat_filename_no_ext,
            visualize_index=i
        )

        # Initialize combined_features for this sample using ALL expected names
        combined_features = {name: np.nan for name in expected_feature_names}
        combined_features["delta_T"] = sample.get("delta_T", np.nan) # Use .get for safety
        combined_features["airflow_rate"] = sample.get("airflow_rate", np.nan) # Use .get for safety

        if extracted_features_dict and isinstance(extracted_features_dict, dict):
            # Populate combined_features with values from extracted_features_dict
            # This correctly handles cases where std features might not be present
            for key, value in extracted_features_dict.items():
                 if key in combined_features: # Check if the extracted key is expected
                     combined_features[key] = value

            feature_list.append(combined_features)
            processed_samples += 1
        else:
            print(f"  Skipping sample {i+1} (feature extraction failed or returned invalid result: {type(extracted_features_dict)}). Appending NaNs.")
            # Append row with NaNs for all expected features
            feature_list.append(combined_features)
            skipped_samples += 1

    end_time = time.time()
    print(f"\nFeature extraction finished. Processed samples (attempted): {processed_samples}, Skipped (invalid input): {skipped_samples}.")
    print(f"Feature extraction took: {end_time - start_time:.2f} seconds.")

    if not feature_list: print("Error: No feature rows generated."); return

    # 3. Create DataFrame and Prepare X, y
    df = pd.DataFrame(feature_list)
    # print("\n--- Feature DataFrame Head (Includes Area) ---")
    # with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
    #     print(df.head())
    # print(f"\nDataFrame shape: {df.shape}")
    # print("\nDataFrame Info:"); df.info()
    print("\nDataFrame Null Counts (Before Log Transform / NaN Drop):"); print(df.isnull().sum()) # Check NaNs BEFORE transform

    # Define feature columns using the CORRECT expected names list
    feature_columns = ["delta_T"] + expected_feature_names

    # --- Check for missing columns (robustness) ---
    cols_actually_in_df = df.columns.tolist()
    missing_cols = [col for col in feature_columns if col not in cols_actually_in_df]
    if missing_cols:
        print(f"Warning: Expected columns missing from DataFrame: {missing_cols}. They will not be included in X.")
        # Update feature_columns to only include columns that actually exist
        feature_columns = [col for col in feature_columns if col in cols_actually_in_df]
        if not feature_columns:
             print("Error: No expected feature columns found in DataFrame. Exiting.")
             return

    # --- Prepare y (handle potential NaNs in target) ---
    if df["airflow_rate"].isnull().any():
         print(f"\nWarning: Dropping {df['airflow_rate'].isnull().sum()} rows with NaN target.")
         df = df.dropna(subset=["airflow_rate"])
         if df.empty: print("Error: All rows dropped due to NaN target."); return
         print(f"  DataFrame shape after dropping NaN target rows: {df.shape}")

    y = df["airflow_rate"].astype(float)
    # Select feature columns into X *from the potentially NaN-dropped df*
    X = df[feature_columns].copy()

    # --- Apply Log Transform ---
    print("\nApplying Log Transform (log1p) to delta_T and hotspot_area...")
    if 'delta_T' in X.columns:
        if (X['delta_T'] <= 0).any(): print("Warning: Non-positive delta_T.")
        X['delta_T'] = np.log1p(X['delta_T'].astype(float))
    # --- Use the new area feature name ---
    if 'hotspot_area' in X.columns:
        if (X['hotspot_area'] < 0).any(): print("Warning: Negative hotspot_area.")
        # Apply log transform if area > 0, otherwise maybe keep 0 or NaN? log1p(0) is 0.
        X['hotspot_area'] = np.log1p(X['hotspot_area'].astype(float))
    print("X head after log transform:")
    print(X.head())
    # Check for NaNs/Infs introduced by log transform
    if X.isnull().any().any() or np.isinf(X.values).any():
         print("\nWarning: NaN or Inf values detected in X AFTER log transform. Check input data or transform logic.")
         print(X.isnull().sum())
         print(f"Inf count: {np.isinf(X.values).sum()}")

    # --- Check and Handle All-NaN Columns in X (AFTER log transform) ---
    cols_before_drop = X.shape[1]
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        print(f"\nWarning: Dropping all-NaN feature columns: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
        print(f"  X shape changed from {len(y)}x{cols_before_drop} to {X.shape}")
    else:
        print("\nNo all-NaN feature columns detected in X.")

    # Check for remaining NaNs/Infs that need imputation (should be handled by imputer)
    if X.isnull().any().any():
        print("\nWarning: NaN values detected in feature matrix X before imputation:")
        print(X.isnull().sum()[X.isnull().sum() > 0])
    if np.isinf(X.values).any():
         print("\nWarning: Inf values detected in feature matrix X before imputation.")
         # Imputer might fail on Inf, consider replacing them:
         # X = X.replace([np.inf, -np.inf], np.nan) # Replace Inf with NaN for imputer
         # Or replace with a large number if appropriate

    # Final checks before proceeding
    if X.empty or y.empty: print("Error: X or y is empty after processing."); return
    if X.shape[1] == 0: print("Error: X has no columns left after processing."); return
    if X.shape[0] != len(y): print(f"Error: Mismatch X rows ({X.shape[0]}) vs y length ({len(y)})."); return

    print(f"\nFinal Feature matrix X shape: {X.shape}")
    print(f"Final Target vector y shape: {y.shape}")
    print("\nFinal X Head (Ready for Pipeline):"); print(X.head())
    print("\ny Description (Final):"); print(y.describe())

    # 4. Hyperparameter Tuning
    print("\n--- Running Hyperparameter Tuning for MLPRegressor ---")
    all_best_params, all_best_scores = tuning.run_grid_search_all_models(X, y)

    print("\n--- Tuning Results Summary (Score: neg_mean_squared_error) ---")
    for name in all_best_params:
         score = all_best_scores[name]; mse = -score if score != -np.inf else np.inf
         rmse = np.sqrt(mse) if mse != np.inf else np.inf
         print(f"{name}: Best Score={score:.4f} (MSE={mse:.4f}, RMSE={rmse:.4f}), Best Params={all_best_params[name]}")

    # 5. Setup Cross-Validation Strategy
    num_samples_final = len(y)
    if config.CV_METHOD == 'LeaveOneOut':
        cv_strategy = LeaveOneOut(); n_splits = num_samples_final; cv_name = "LeaveOneOut"
    elif config.CV_METHOD == 'KFold':
        k_folds_config = getattr(config, 'K_FOLDS', 5)
        n_splits = min(k_folds_config, num_samples_final)
        if n_splits < 2: cv_strategy = LeaveOneOut(); n_splits = num_samples_final; cv_name = "LeaveOneOut (Fallback)"
        else: cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE); cv_name = f"{n_splits}-Fold KFold"
    else: cv_strategy = LeaveOneOut(); n_splits = num_samples_final; cv_name = "LeaveOneOut (Default)"
    print(f"\nUsing {cv_name} Cross-Validation with {n_splits} splits for evaluation.")

    # 6. Evaluate MLPRegressor using TUNED parameters
    print(f"\n--- Evaluating MLPRegressor using Best Parameters found during Tuning ---")
    evaluation_results = {}
    all_regressors_to_eval = get_regressors() # Only MLP

    for name, model_instance in all_regressors_to_eval.items():
        print(f"\nEvaluating model: {name} with tuned parameters...")
        tuned_params_raw = all_best_params.get(name, {})
        tuned_params = {k.replace('model__', ''): v for k, v in tuned_params_raw.items()}

        if tuned_params:
             print(f"  Applying Tuned Parameters: {tuned_params}")
             try: model_instance.set_params(**tuned_params)
             except ValueError as e: print(f"  Warning: Using defaults. Error setting params: {e}"); model_instance = get_regressors()[name]
        else: print("  No tuned parameters found. Using default parameters.")

        pipeline_model = modeling.build_pipeline(model_instance, pca_components=None)

        try:
            start_cv_time = time.time()
            y_pred_cv = cross_val_predict(pipeline_model, X, y, cv=cv_strategy, n_jobs=-1) # Set n_jobs=1 to help suppress warnings
            end_cv_time = time.time()
            print(f"  CV predictions for {name} took {end_cv_time - start_cv_time:.2f} seconds.")
            if np.isnan(y_pred_cv).any(): print(f"  Warning: NaN values found in CV predictions.")

            mse = mean_squared_error(y, y_pred_cv); rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred_cv); r2 = r2_score(y, y_pred_cv)
            neg_mse_eval = -mse
            results_dict = {'neg_mean_squared_error': neg_mse_eval, 'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred_cv': y_pred_cv }
            evaluation_results[name] = results_dict

            print(f"--- Results for {name} (Aggregated over {cv_name} folds with Tuned Params) ---")
            print(f"MSE:  {mse:.4f}"); print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}"); print(f"R²:   {r2:.4f}")

        except Exception as e:
            print(f"Error during CV evaluation for {name}: {type(e).__name__} - {e}")
            import traceback; traceback.print_exc(); evaluation_results[name] = None

    # 7. Analyze Results and Plot
    valid_results = {name: res for name, res in evaluation_results.items() if res is not None}
    if not valid_results: print("\nError: MLPRegressor evaluation failed."); return

    best_model_name = list(valid_results.keys())[0]
    metric_to_optimize = getattr(config, 'BEST_MODEL_METRIC_SCORE', 'neg_mean_squared_error')
    print(f"\n--- Results for Best Model ({best_model_name}) based on CV {metric_to_optimize} ---")
    best_result_metrics = valid_results[best_model_name]
    print(f"  Optimization Score ({metric_to_optimize}): {best_result_metrics[metric_to_optimize]:.4f}")
    print(f"  MSE:  {best_result_metrics['mse']:.4f}"); print(f"  RMSE: {best_result_metrics['rmse']:.4f}")
    print(f"  MAE:  {best_result_metrics['mae']:.4f}"); print(f"  R²:   {best_result_metrics['r2']:.4f}")

    # --- Plotting Actual vs Predicted (Conditional) ---
    save_actual_pred_plot = getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', False)
    actual_pred_plot_path = getattr(config, 'ACTUAL_VS_PREDICTED_PLOT_SAVE_PATH', None)
    plot_title = f'Actual vs. Predicted - {best_model_name} (Tuned, {cv_name}-CV Predictions)'

    if save_actual_pred_plot and actual_pred_plot_path:
        print(f"\nAttempting to save Actual vs Predicted plot to: {actual_pred_plot_path}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(actual_pred_plot_path), exist_ok=True)
        save_path_arg = actual_pred_plot_path
    else:
        print("\nDisplaying Actual vs Predicted plot (not saving).")
        save_path_arg = None # Pass None to display instead of saving

    try:
        if len(y) == len(best_result_metrics['y_pred_cv']):
            plotting.plot_actual_vs_predicted(
                y_true=y,
                y_pred=best_result_metrics['y_pred_cv'],
                title=plot_title,
                save_path=save_path_arg # Pass the determined save path or None
            )
        else: print(f"Warning: Length mismatch for plotting Actual vs Predicted.")
    except Exception as e: print(f"Error plotting Actual vs Predicted: {e}")

    # 8. Final Model Training, Saving, and Analysis
    print(f"\n--- Training final {best_model_name} regressor (tuned) on all data ---")
    try:
        final_model_instance = get_regressors()[best_model_name]
        best_params_raw = all_best_params.get(best_model_name, {})
        best_params_final = {k.replace('model__', ''): v for k,v in best_params_raw.items()}
        if best_params_final:
            print(f"  Applying Tuned Parameters: {best_params_final}")
            final_model_instance.set_params(**best_params_final)

        final_pipeline = modeling.build_pipeline(final_model_instance, pca_components=None)
        final_pipeline.fit(X, y)
        print("Final model training complete.")

        # --- Feature Analysis ---
        print(f"\n--- Analyzing Features for Final {best_model_name} Model ---")
        # ...(existing MLP analysis message)...
        if 'model' in final_pipeline.named_steps:
            fitted_model = final_pipeline.named_steps['model']
            if isinstance(fitted_model, MLPRegressor):
                 print(f"Feature analysis for {type(fitted_model).__name__}: Standard coefficients/importances not directly available.")
                 print("Consider permutation importance (separate implementation).")
                 print(f"  Model Architecture: hidden_layer_sizes={fitted_model.hidden_layer_sizes}, activation='{fitted_model.activation}', solver='{fitted_model.solver}'")
            else: print(f"Model type '{type(fitted_model).__name__}' unknown for feature analysis.")
        else: print("Error: Could not find 'model' step in pipeline.")


         # --- Plotting Training Loss Curve (Conditional) ---
        save_loss_plot = getattr(config, 'SAVE_LOSS_CURVE_PLOT', False)
        loss_plot_dir = getattr(config, 'LOSS_CURVE_PLOT_SAVE_DIR', '.') # Default to current dir

        if save_loss_plot:
            print(f"\n--- Plotting Training Loss Curve for Final {best_model_name} Model ---")
            if 'model' in final_pipeline.named_steps:
                fitted_model = final_pipeline.named_steps['model']
                if isinstance(fitted_model, MLPRegressor) and hasattr(fitted_model, 'loss_curve_'):
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(10, 5))
                        plt.plot(fitted_model.loss_curve_)
                        plt.title(f'{best_model_name} Training Loss Curve')
                        plt.xlabel('Epochs (Iterations)')
                        plt.ylabel('Loss')
                        plt.grid(True)
                        # Ensure directory exists and construct path
                        os.makedirs(loss_plot_dir, exist_ok=True)
                        loss_plot_save_path = os.path.join(loss_plot_dir, f"final_{best_model_name}_loss_curve.png")
                        plt.savefig(loss_plot_save_path)
                        plt.close()
                        print(f"Training loss curve saved to: {loss_plot_save_path}")
                    except ImportError:
                         print("Warning: matplotlib not found. Cannot save loss curve plot. Install with 'pip install matplotlib'")
                    except Exception as plot_err:
                         print(f"Error plotting loss curve: {plot_err}")
                else:
                    print("Final model is not MLPRegressor or loss curve unavailable.")
            else:
                print("Error: Could not find 'model' step for loss curve plot.")
        else:
             print("\nSkipping saving of training loss curve plot (SAVE_LOSS_CURVE_PLOT is False).")


        # --- Save Final Model ---
        model_save_path = getattr(config, 'MODEL_SAVE_PATH', 'final_model.joblib')
        # Ensure directory exists for model saving
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(final_pipeline, model_save_path)
        print(f"\nFinal {best_model_name} (tuned) pipeline saved to: {model_save_path}")

    except Exception as e:
        print(f"Error during final model training/saving: {type(e).__name__} - {e}")
        import traceback; traceback.print_exc()

    print("\n--- Experiment Finished ---")

if __name__ == "__main__":
    run_experiment()
# --- END OF FILE main.py ---