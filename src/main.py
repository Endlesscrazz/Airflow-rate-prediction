# main.py
"""Main script to run the airflow prediction regression experiment."""

import os
import pandas as pd
import joblib
import numpy as np
import time
import warnings

# Import project modules
import config           # Ensure config has FRAME_INTERVAL_SIZE, MAX_FRAME_INTERVALS
import data_utils
import feature_engineering # Uses multi-interval grad & std (FRAME BASED)
import modeling          # Regressors
import plotting          # Regression plots
import tuning            # Regression tuning
import data_preprocessing
import vis

from modeling import get_regressors # Get regressors
# Import sklearn components
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
# Import regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import FitFailedWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*SettingWithCopyWarning*")
warnings.filterwarnings("ignore", category=FutureWarning)
# Keep the UserWarning for imputation for now, as it might still be useful
# warnings.filterwarnings("ignore", category=UserWarning, message=".*Skipping features without any observed values.*")

try:
    _project_root = config._project_root # If defined in config
except AttributeError:
    # Redefine if necessary (assuming main.py is in src)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    print("Note: Redefined _project_root in main.py")

def run_experiment():
    """Loads data, extracts multi-interval temporal features (frame-based),
       tunes hyperparameters for regressors, evaluates models using regression metrics,
       analyzes features, and saves the best regressor."""

    print("--- Starting Airflow Prediction Experiment (Regression) ---")
    print("--- Features: delta_T, Multi-Interval Bright Region Gradients, Brighr Region area (FRAME-BASED) ---")
    print("--- NOTE: PCA, CNN, Ensembles are DISABLED ---")


    # 1. Load Raw Data
    all_raw_data = data_utils.load_raw_data(config.DATASET_FOLDER)
    if not all_raw_data: print("Error: No data loaded. Exiting."); return

    # --- ADD Preprocessing Step ---
    all_processed_data = []
    print("\n--- Preprocessing Raw Video Frames ---")

      # --- Read settings from config with defaults ---
    apply_filter = getattr(config, 'PREPROCESSING_APPLY_FILTER', True)
    kernel_size = getattr(config, 'PREPROCESSING_FILTER_KERNEL_SIZE', 3)
    apply_scaling = getattr(config, 'PREPROCESSING_APPLY_SCALING', True)
    scale_range = getattr(config, 'PREPROCESSING_SCALE_RANGE', (0, 1))

    print(f"  Settings: Filter={apply_filter} (Kernel={kernel_size}), Scaling={apply_scaling} (Range={scale_range})")

    for i, sample in enumerate(all_raw_data):
        if "frames" in sample and isinstance(sample["frames"], np.ndarray):
            # Apply preprocessing using settings (can be controlled via config later)
            processed_frames = data_preprocessing.preprocess_video_frames(
                sample["frames"], 
                apply_filter=apply_filter,    # Enable/disable via config?
                filter_kernel_size=kernel_size, # Set kernel size via config?
                apply_scaling=apply_scaling,   # Enable/disable via config?
                scale_range=scale_range    # Set range via config?
            )
            # Create a new dictionary or update the existing one
            processed_sample = sample.copy() # Avoid modifying original raw data dict
            processed_sample["frames"] = processed_frames 
            all_processed_data.append(processed_sample)
        else:
            print(f"  Skipping preprocessing for sample {i+1} due to missing/invalid frames.")
            all_processed_data.append(sample) # Append original if frames are bad
    # --- END Preprocessing Step ---


    # 2. Extract Specific Features (Multi-Interval Bright Region Gradient FRAME BASED)
    feature_list = []

    if not hasattr(config, 'FRAME_INTERVAL_SIZE') or not hasattr(config, 'MAX_FRAME_INTERVALS'):
        print("Error: config.py missing FRAME_INTERVAL_SIZE or MAX_FRAME_INTERVALS. Exiting.")
        return
        
    interval_frames = config.FRAME_INTERVAL_SIZE
    max_intervals = config.MAX_FRAME_INTERVALS          # Use the new config for gradients
    calculate_std = getattr(config, 'CALCULATE_STD_FEATURES', True)

    # Generate gradient feature names based on intervals
    expected_grad_features = []
    expected_std_features = []
    for i in range(max_intervals):
        start_frame = i * interval_frames
        end_frame = (i + 1) * interval_frames
        expected_grad_features.append(f"bright_region_grad_{start_frame}_{end_frame}")
        if calculate_std: # Only add std names if flag is True
            expected_std_features.append(f"bright_region_std_{start_frame}_{end_frame}")

    expected_area_feature = ["bright_region_area"] 

    # Combine the expected feature lists
    expected_feature_names = expected_area_feature + expected_grad_features + expected_std_features
    
    print(f"\nExpecting features: delta_T, {expected_feature_names}")

    feature_columns = ["delta_T"] + expected_feature_names
    
    print("\n--- Extracting Bright Region Features (Area, Grad, Optional Std) ---") 
    start_time = time.time()
    processed_samples = 0
    skipped_samples = 0

    for i, sample in enumerate(all_processed_data):
        # print(f"Processing sample {i+1}/{len(all_raw_data)}: {os.path.basename(sample['filepath'])}") # Verbose
        
        if "frames" not in sample or not isinstance(sample["frames"], np.ndarray) or sample["frames"].ndim != 3:
             print(f"  Skipping feature extraction for sample {i+1} due to missing/invalid 'frames' data post-preprocessing.")
             combined_features = {name: np.nan for name in expected_feature_names}
             combined_features["delta_T"] = sample.get("delta_T", np.nan)
             combined_features["airflow_rate"] = sample.get("airflow_rate", np.nan)
             feature_list.append(combined_features)
             skipped_samples += 1
             continue 

        interval_features = feature_engineering.extract_bright_region_features(sample["frames"]) 

        combined_features = {name: np.nan for name in expected_feature_names} 
        combined_features["delta_T"] = sample["delta_T"]
        combined_features["airflow_rate"] = sample["airflow_rate"] 

        if interval_features and isinstance(interval_features, dict): 
            updated_count = 0
            for key, value in interval_features.items():
                 if key in combined_features: 
                     combined_features[key] = value
                     updated_count += 1
                 else: 
                     print(f"  Warning: Feature '{key}' generated for sample {i+1} but not in expected list: {expected_feature_names}. Ignoring.")

            if updated_count > 0:
                feature_list.append(combined_features)
                processed_samples += 1
            else:
                # This case means the returned dict was empty or contained no expected keys
                print(f"  Sample {i+1} processed but generated no *expected* features (returned {list(interval_features.keys())}). Appending NaNs.")
                feature_list.append(combined_features) 
                skipped_samples += 1
        else:
            print(f"  Skipping sample {i+1} (feature extraction failed or returned invalid result: {type(interval_features)}). Appending NaNs.")
            feature_list.append(combined_features) 
            skipped_samples += 1

    end_time = time.time()
    print(f"\nFeature extraction finished. Samples providing >=1 expected feature: {processed_samples}, Samples providing 0 expected features (NaNs appended): {skipped_samples}.")
    print(f"Feature extraction took: {end_time - start_time:.2f} seconds.")

    if not feature_list: print("Error: No feature rows generated (not even NaN rows)."); return


    # 3. Create DataFrame and Prepare X, y for Regression
    df = pd.DataFrame(feature_list)

    # DEBUGGING 
    # print("\n--- Feature DataFrame Head (Frame-Based Intervals + DeltaT) ---")
    # # Increase display options for better inspection
    # with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
    #     print(df.head())
    #     print("\n--- Feature DataFrame Tail ---")
    #     print(df.tail())
    # print(f"\nDataFrame shape: {df.shape}")
    # print("\nDataFrame Info:")
    # df.info() # Good for checking dtypes and non-null counts
    # print("\nDataFrame Null Counts (Before potential column drop):")
    # print(df.isnull().sum()) # Crucial check!

    target_column = "airflow_rate"

    run_visualizations = getattr(config, 'RUN_VISUALIZATIONS', True)
    vis_save_dir = getattr(config, 'VIS_SAVE_DIR', os.path.join(_project_root, "plots"))

    if run_visualizations:
        print("\n--- Running Data Analysis Visualizations ---")

        os.makedirs(vis_save_dir, exist_ok=True) 
        # Ensure target column exists before proceeding
        if target_column in df.columns:
             
             # Prepare list of features to visualize (exclude target)
             features_to_vis = [col for col in feature_columns if col in df.columns] # Use only existing cols
            
             # Dyanmic file names
             scatter_save_path = os.path.join(vis_save_dir, "scatter_features_vs_target.png")
             boxplot_save_path = os.path.join(vis_save_dir, "boxplots_features_by_target.png")
             # Include target in correlation columns
             corr_columns = features_to_vis + [target_column] 
             corr_save_path = os.path.join(vis_save_dir, "correlation_matrix.png")

             # Call visualization functions
             vis.plot_feature_vs_target(df, features_to_vis, target_column, save_path=scatter_save_path)
             vis.plot_feature_distributions_by_target(df, features_to_vis, target_column, save_path=boxplot_save_path)
             vis.plot_correlation_matrix(df, corr_columns, save_path=corr_save_path) 
        else:
             print(f"Warning: Target column '{target_column}' not found in DataFrame. Skipping visualizations.")


    feature_columns = ["delta_T"] + expected_feature_names

    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Expected columns missing from DataFrame: {missing_cols}. Exiting.")
        # Exiting will be safer than trying to add them, as it suggests a fundamental issue.
        return

    # Handling Nan in y
    if df[target_column].isnull().any():
         print(f"\nWarning: NaN values detected in '{target_column}'.")
         print(f"  Dropping {df[target_column].isnull().sum()} rows with NaN target.")
         df = df.dropna(subset=[target_column])
         if df.empty:
              print("Error: All rows dropped due to NaN target.")
              return
         print(f"  DataFrame shape after dropping NaN target rows: {df.shape}")

    # Ensure columns exist before selection
    actual_feature_columns = [col for col in feature_columns if col in df.columns]
    if len(actual_feature_columns) != len(feature_columns):
         print(f"Warning: Not all expected feature columns found in final DataFrame. Using: {actual_feature_columns}")

    y = df["airflow_rate"].astype(float)
    X = df[feature_columns].copy() 

    # --- Check and Handle All-NaN Columns in X ---
    cols_before_drop = X.shape[1]
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        print(f"\nWarning: The following feature columns contain only NaN values and will be dropped: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
        print(f"  X shape changed from {len(y)}x{cols_before_drop} to {X.shape}")
        # Update feature_columns list to reflect actual columns used
        feature_columns_after_drop = X.columns.tolist()
        print(f"  Actual features used for modeling: {feature_columns_after_drop}")
    else:
        print("\nNo all-NaN feature columns detected in X.")
        feature_columns_after_drop = feature_columns # No columns dropped


    # Checking for other NaNs that need imputation
    if X.isnull().any().any():
        print("\nWarning: NaN values detected in feature matrix X before imputation:")
        
        print(X.isnull().sum()[X.isnull().sum() > 0])
    else:
        print("\nNo NaNs detected in feature matrix X before imputation (or they were in dropped columns).")

    if X.empty or y.empty: print("Error: X or y is empty after processing."); return
    if X.shape[1] == 0: print("Error: X has no columns left after dropping all-NaN columns."); return
    if X.shape[0] != len(y): # Sanity check after potential target NaN drop
        print(f"Error: Mismatch between X rows ({X.shape[0]}) and y length ({len(y)}).")
        return

    print(f"\nFinal Feature matrix X shape: {X.shape}")
    print(f"Final Target vector y shape: {y.shape}")
    print("\nFinal X Head (first 5 rows):")
    print(X.head())
    print("\ny Description:")
    print(y.describe())

    # 4. Hyperparameter Tuning (for Regressors)
    print("\n--- Running Hyperparameter Tuning for All Regressors ---")
    all_best_params, all_best_scores = tuning.run_grid_search_all_models(X, y)

    print("\n--- Tuning Results Summary (Score: neg_mean_squared_error) ---")
    for name in all_best_params:
         score = all_best_scores[name]
         mse = -score if score != -np.inf else np.inf
         rmse = np.sqrt(mse) if mse != np.inf else np.inf
         print(f"{name}: Best Score={score:.4f} (MSE={mse:.4f}, RMSE={rmse:.4f}), Best Params={all_best_params[name]}")


    # 5. Setup Cross-Validation Strategy for Evaluation
    # Ensure n_splits calculation uses the final length of X or y
    num_samples_final = len(y)
    if config.CV_METHOD == 'LeaveOneOut':
        cv_strategy = LeaveOneOut()
        n_splits = num_samples_final
        cv_name = "LeaveOneOut"
    elif config.CV_METHOD == 'KFold':
        n_splits = min(config.K_FOLDS, num_samples_final)
        if n_splits < 2:
             print(f"Warning: K_FOLDS ({config.K_FOLDS}) too large or dataset too small ({num_samples_final}). Falling back to LeaveOneOut.")
             cv_strategy = LeaveOneOut(); n_splits = num_samples_final; cv_name = "LeaveOneOut"
        else:
             cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE); cv_name = f"{n_splits}-Fold KFold"
    else:
        print(f"Warning: Invalid CV_METHOD '{config.CV_METHOD}'. Defaulting to LeaveOneOut.")
        cv_strategy = LeaveOneOut(); n_splits = num_samples_final; cv_name = "LeaveOneOut"
    print(f"\nUsing {cv_name} Cross-Validation with {n_splits} splits.")

    # 6. Evaluate individual regressors USING TUNED parameters with Cross-Validation
    print(f"\n--- Evaluating Regressors using Best Parameters found during Tuning ---")
    # print(f"--- Using {cv_name} Cross-Validation ({n_splits} splits) ---") # Already printed above

    evaluation_results = {}
    all_regressors_to_eval = get_regressors()

    for name, model_instance in all_regressors_to_eval.items():
        print(f"\nEvaluating model: {name} with tuned parameters...")
        tuned_params_raw = all_best_params.get(name, {})
        # Remove 'model__' prefix if present from grid search results
        tuned_params = {k.replace('model__', ''): v for k,v in tuned_params_raw.items()}

        if tuned_params:
             print(f"  Applying Tuned Parameters: {tuned_params}")
             try:
                 # Apply params directly to the model instance before building pipeline
                 model_instance.set_params(**tuned_params)
             except ValueError as e:
                 print(f"  Warning: Could not set parameters {tuned_params} for {name}. Using defaults. Error: {e}")
                 # Reset to default instance if params failed
                 model_instance = get_regressors()[name]
        else:
             print("  No tuned parameters found or applied. Using default parameters.")

        # Build the pipeline (Imputer + Scaler + Model)
        pipeline_model = modeling.build_pipeline(model_instance, pca_components=None)

        try:
            start_cv_time = time.time()
            # Use the final X and y for cross-validation
            y_pred_cv = cross_val_predict(pipeline_model, X, y, cv=cv_strategy, n_jobs=-1)
            end_cv_time = time.time()
            print(f"  CV predictions for {name} took {end_cv_time - start_cv_time:.2f} seconds.")

            # Ensure y_pred_cv and y have compatible shapes and no NaNs introduced unexpectedly
            if np.isnan(y_pred_cv).any():
                 print(f"  Warning: NaN values found in cross-val predictions for {name}. Evaluation metrics might be affected.")

            mse = mean_squared_error(y, y_pred_cv)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred_cv)
            r2 = r2_score(y, y_pred_cv)
            neg_mse_eval = -mse # Use this for comparison if optimizing neg_mse

            results_dict = {'neg_mean_squared_error': neg_mse_eval, 'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred_cv': y_pred_cv }
            evaluation_results[name] = results_dict

            print(f"--- Results for {name} (Aggregated over {cv_name} folds with Tuned Params) ---")
            print(f"MSE:  {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}")
            print(f"R²:   {r2:.4f}")

        except Exception as e:
            print(f"Error during CV evaluation for {name}: {type(e).__name__} - {e}")
            import traceback; traceback.print_exc()
            evaluation_results[name] = None


    # 7. Analyze Results and Plot Best Individual Regressor
    valid_results = {name: res for name, res in evaluation_results.items() if res is not None}
    if not valid_results: print("\nError: No models evaluated successfully."); return

    metric_to_optimize = getattr(config, 'BEST_MODEL_METRIC_SCORE', 'neg_mean_squared_error') # Default if not in config
    higher_is_better = getattr(config, 'METRIC_COMPARISON_HIGHER_IS_BETTER', True) # Default if not in config

    try:
        # Find best model based on the chosen metric
        best_model_name = max(valid_results, key=lambda n: valid_results[n][metric_to_optimize]) if higher_is_better \
                     else min(valid_results, key=lambda n: valid_results[n][metric_to_optimize])
        
        plot_save_path = getattr(config, 'PLOT_SAVE_PATH', os.path.join(_project_root, "output"))

        print(f"\n--- Best Individual Regressor based on CV {metric_to_optimize}: {best_model_name} ---")
        best_result_metrics = valid_results[best_model_name]
        print(f"  Optimization Score ({metric_to_optimize}): {best_result_metrics[metric_to_optimize]:.4f}")
        print(f"  MSE:  {best_result_metrics['mse']:.4f}")
        print(f"  RMSE: {best_result_metrics['rmse']:.4f}")
        print(f"  MAE:  {best_result_metrics['mae']:.4f}")
        print(f"  R²:   {best_result_metrics['r2']:.4f}")

        best_model_plot_filename = f"actual_vs_predicted_{best_model_name}.png"
        best_model_plot_save_path = os.path.join(plot_save_path, best_model_plot_filename)
        

        # Plotting using the final y and the cross-validated predictions for the best model
        # Ensure y and y_pred_cv have the same length
        if len(y) == len(best_result_metrics['y_pred_cv']):
            plotting.plot_actual_vs_predicted(y_true=y,
                                             y_pred=best_result_metrics['y_pred_cv'],
                                             title=f'Actual vs. Predicted - {best_model_name} (Tuned, {cv_name}-CV Predictions)')
                                             # save_path=config.PLOT_SAVE_PATH) # Add save path if needed from config
        else:
             print(f"Warning: Length mismatch between y_true ({len(y)}) and y_pred_cv ({len(best_result_metrics['y_pred_cv'])}) for plotting.")

    except KeyError as e:
        print(f"Error: Metric '{metric_to_optimize}' not found in evaluation results for key {e}. Check metric name and evaluation output.")
    except Exception as e:
        print(f"Error determining or plotting best model: {type(e).__name__} - {e}")


    # 8. Final Model Training, Saving, and Feature Analysis
    if 'best_model_name' not in locals():
        print("\nSkipping final model training as best model could not be determined.")
    else:
        print(f"\n--- Training final {best_model_name} regressor (tuned) on all data ---")
        try:
            final_model_instance = get_regressors()[best_model_name]
            best_params_raw = all_best_params.get(best_model_name, {})
            best_params_final = {k.replace('model__', ''): v for k,v in best_params_raw.items()}
            if best_params_final:
                print(f"  Applying Tuned Parameters: {best_params_final}")
                final_model_instance.set_params(**best_params_final)

            # Build the final pipeline (using the same steps as in CV)
            final_pipeline = modeling.build_pipeline(final_model_instance, pca_components=None)
            # Train on the FINAL X and y (after dropping NaNs, etc.)
            final_pipeline.fit(X, y)
            print("Final model training complete.")

            # --- Feature Importance/Coefficient Analysis ---
            try:
                print(f"\n--- Analyzing Features for Final {best_model_name} Model ---")
                # Access the fitted model step from the pipeline
                if 'model' in final_pipeline.named_steps:
                    fitted_model = final_pipeline.named_steps['model']
                    # Get feature names from the final X DataFrame (after potential column drops)
                    feature_names_for_analysis = X.columns.tolist() # Use columns from the actual X used for fitting

                    if hasattr(fitted_model, 'coef_'):
                        print("Model Coefficients (Linear Model):")
                        coefficients = fitted_model.coef_
                        # Adjust shape handling (common for linear models)
                        if coefficients.ndim == 2 and coefficients.shape[0] == 1:
                            coefficients = coefficients.flatten()

                        if coefficients.ndim == 1 and len(coefficients) == len(feature_names_for_analysis):
                             coef_df = pd.DataFrame({'Feature': feature_names_for_analysis, 'Coefficient': coefficients})
                             coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
                             coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])
                             print(coef_df.to_string(index=False))
                        else: print(f"  Coefficient array shape {coefficients.shape} not directly mappable to features {len(feature_names_for_analysis)}.")

                    elif hasattr(fitted_model, 'feature_importances_'):
                         print("Feature Importances (Tree-Based Model):")
                         importances = fitted_model.feature_importances_
                         if len(importances) == len(feature_names_for_analysis):
                             imp_df = pd.DataFrame({'Feature': feature_names_for_analysis, 'Importance': importances})
                             imp_df = imp_df.sort_values('Importance', ascending=False)
                             print(imp_df.to_string(index=False))
                         else: print(f"  Importance array length {len(importances)} mismatch with features {len(feature_names_for_analysis)}.")
                    else: print(f"Model type '{type(fitted_model).__name__}' does not have standard .coef_ or .feature_importances_.")
                else:
                    print("Error: Could not find 'model' step in the final pipeline for feature analysis.")

            except Exception as coef_err: print(f"Error during feature analysis: {coef_err}")
            # --- END Feature Analysis ---

            # Save the final pipeline
            model_save_path = getattr(config, 'MODEL_SAVE_PATH', 'final_model.joblib') # Get path from config or use default
            joblib.dump(final_pipeline, model_save_path)
            print(f"\nFinal {best_model_name} (tuned) pipeline saved to: {model_save_path}")

        except Exception as e:
            print(f"Error during final model training/saving: {type(e).__name__} - {e}")
            import traceback; traceback.print_exc()

    print("\n--- Experiment Finished ---")

if __name__ == "__main__":
    run_experiment()