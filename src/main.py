
"""
Main script to run the airflow prediction regression experiment using
pre-computed hotspot masks. Allows easy switching of feature sets.
"""

import os
import pandas as pd
import joblib
import numpy as np
import time
import warnings
import fnmatch
import traceback

# Import project modules
import config
import data_utils
import feature_engineering
import modeling
import plotting
import tuning

from modeling import get_regressors
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
# --- Import other models if needed for comparison ---
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor # Example if you add it later

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*SettingWithCopyWarning*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Constants ---
PRECOMPUTED_MASK_DIR = getattr(config, 'MASK_DIR', 'output_hotspot_mask') # Get from config or default

#PRECOMPUTED_MASK_DIR = getattr(config, 'MASK_DIR', 'output_hotspot_mask_5s') 

# --- Feature Selection Control ---
# Define ALL possible features that feature_engineering.py might return
ALL_POSSIBLE_FEATURE_NAMES = [
    'hotspot_area',
    'hotspot_temp_change_rate',
    'hotspot_temp_change_magnitude',
    'activity_mean', # Added
    'activity_median',# Added
    'activity_std', # Added
    'activity_max', # Added
    'activity_sum', # Added
]

# === CHOOSE YOUR FEATURE SET HERE ===
# (delta_T is added separately later)
SELECTED_FEATURE_NAMES_FROM_MASK = [
    # 'hotspot_area', # Still seems detrimental, keep commented
    'hotspot_temp_change_rate',      # Keep for normalization
    'hotspot_temp_change_magnitude', # Keep for normalization
    'activity_mean',                 # Add new activity features
    'activity_median',
    'activity_std',
    'activity_max',
    #'activity_sum',
]
# --- Baseline Feature Set  ---
# SELECTED_FEATURE_NAMES_FROM_MASK = [
#     'hotspot_temp_change_rate',
#     'hotspot_temp_change_magnitude' 
# ]


# Parameters for feature calculation (should match mask generation)
FPS_FOR_FEATURES = getattr(config, 'FPS', 5.0)
FOCUS_DURATION_SEC_FOR_FEATURES = getattr(config, 'FOCUS_DURATION_SEC', 10.0)
SMOOTH_WINDOW = getattr(config, 'SMOOTH_WINDOW', 1)
ENVIR_PARA = getattr(config, 'ENVIR_PARA', -1)
ERRORWEIGHT = getattr(config, 'ERRORWEIGHT', 0.5)
AUGMENT = getattr(config, 'AUGMENT', 1.0)
FUSELEVEL = getattr(config, 'FUSELEVEL', 0)
NORMALIZET = getattr(config, 'NORMALIZET', 0)
ROI_BORDER_PERCENT = getattr(config, 'ROI_BORDER_PERCENT', None) # Use None if ROI wasn't used for masks


def run_experiment():
    """Loads data, extracts features using masks, selects features,
       trains/evaluates models."""

    print("--- Starting Airflow Prediction Experiment (Regression) ---")
    print("--- Using Pre-computed Masks & Recalculated Activity Maps ---")
    print(f"--- Mask Directory: {PRECOMPUTED_MASK_DIR} ---")
    print(f"--- Feature/ActivityMap Params: FPS={FPS_FOR_FEATURES}, Focus Duration={FOCUS_DURATION_SEC_FOR_FEATURES}s, Smooth={SMOOTH_WINDOW}, etc. ---")
    print(f"--- Selected Mask Features: {SELECTED_FEATURE_NAMES_FROM_MASK} ---")
    print(f"--- Models: {list(modeling.get_regressors().keys())} ---")

    # 1. Find Data Files and Corresponding Masks
    all_samples_info = []
    input_dataset_dir = config.DATASET_FOLDER
    # ... (Keep the file scanning logic exactly as before) ...
    if not input_dataset_dir or not os.path.isdir(input_dataset_dir): print(f"Error: Dataset folder not found: {input_dataset_dir}"); return
    print(f"\nScanning for .mat files in: {input_dataset_dir}")
    file_count, skipped_meta_count, skipped_mask_count = 0, 0, 0
    for root, _, files in os.walk(input_dataset_dir):
        folder_name = os.path.basename(root)
        for mat_filename in sorted(fnmatch.filter(files, '*.mat')):
            file_count += 1; mat_filepath = os.path.join(root, mat_filename)
            relative_dir = os.path.relpath(root, input_dataset_dir)
            mat_filename_no_ext = os.path.splitext(mat_filename)[0]
            mask_filename = mat_filename_no_ext + '_mask.npy'
            mask_path = os.path.join(PRECOMPUTED_MASK_DIR, relative_dir, mask_filename)
            if not os.path.exists(mask_path): skipped_mask_count += 1; continue
            try:
                airflow_rate = data_utils.parse_airflow_rate(folder_name)
                delta_T = data_utils.parse_delta_T(mat_filename)
                if delta_T is None: raise ValueError("Could not parse delta_T")
            except ValueError as e: skipped_meta_count += 1; continue
            all_samples_info.append({"mat_filepath": mat_filepath, "mask_path": mask_path, "delta_T": float(delta_T), "airflow_rate": float(airflow_rate),"source_folder_name": folder_name, "mat_filename_no_ext": mat_filename_no_ext})
    print(f"\nFound {file_count} .mat files. Skipped Meta: {skipped_meta_count}, Skipped Mask: {skipped_mask_count}. Proceeding with {len(all_samples_info)} samples.")
    if not all_samples_info: print("Error: No valid samples found."); return

    # 2. Extract Features using Masks
    feature_list = []
    print("\n--- Extracting Features (Recalculating Activity Map) ---")
    start_time = time.time()
    processed_samples, feature_error_count = 0, 0

    for i, sample_info in enumerate(all_samples_info):
        # Call the updated feature extraction function
        # Pass all necessary parameters for activity map recalculation
        extracted_features_dict = feature_engineering.extract_features_with_mask(
            frames_or_path=sample_info["mat_filepath"],
            mask_path=sample_info["mask_path"],
            # Parameters for activity map recalc & feature calc:
            fps=FPS_FOR_FEATURES,
            focus_duration_sec=FOCUS_DURATION_SEC_FOR_FEATURES,
            smooth_window=SMOOTH_WINDOW,
            envir_para=ENVIR_PARA,
            errorweight=ERRORWEIGHT,
            augment=AUGMENT,
            fuselevel=FUSELEVEL,
            normalizeT=NORMALIZET,
            roi_border_percent=ROI_BORDER_PERCENT,
            # Other args:
            source_folder_name=sample_info["source_folder_name"],
            mat_filename_no_ext=sample_info["mat_filename_no_ext"]
        )

        # --- Combine features ---
        # Initialize with ALL POSSIBLE feature names + metadata
        combined_features = {name: np.nan for name in ALL_POSSIBLE_FEATURE_NAMES}
        combined_features["delta_T"] = sample_info["delta_T"]
        combined_features["airflow_rate"] = sample_info["airflow_rate"]

        if extracted_features_dict and isinstance(extracted_features_dict, dict):
             # Populate with extracted values
             has_valid_feature = False
             for key, value in extracted_features_dict.items():
                 if key in combined_features: combined_features[key] = value
                 # Check if any *selected* feature is valid
                 if key in SELECTED_FEATURE_NAMES_FROM_MASK and np.isfinite(value): has_valid_feature = True
             # Add special check if only area is selected? No, area is calculated first.

             # Add row only if at least one selected feature is valid
             if has_valid_feature or 'hotspot_area' in SELECTED_FEATURE_NAMES_FROM_MASK: # Allow if only area is selected
                  feature_list.append(combined_features)
                  processed_samples += 1
             else:
                  feature_error_count += 1; print(f"Warning: Skipping {sample_info['mat_filename_no_ext']} - No valid *selected* features.")
        else:
            feature_error_count += 1; print(f"Error: Feature extraction failed for {sample_info['mat_filename_no_ext']}.")

    end_time = time.time()
    print(f"\nFeature extraction finished. Processed: {processed_samples}, Errors/Skipped: {feature_error_count}. Took: {end_time - start_time:.2f}s.")
    if not feature_list: print("Error: No feature rows generated."); return

    # 3. Create DataFrame and Prepare X, y
    df = pd.DataFrame(feature_list)
    print(f"\nDataFrame shape: {df.shape}")
    # print("\nDataFrame Null Counts (Before Selection/Transforms):"); print(df.isnull().sum()) # Less useful now

    # --- Define BASE feature columns (metadata + selected mask features) ---
    current_feature_columns = ["delta_T"] + SELECTED_FEATURE_NAMES_FROM_MASK
    #print(f"\nUsing BASE features: {current_feature_columns}")

    # --- Prepare y (Target Variable) ---
    if df["airflow_rate"].isnull().any():
         print(f"Warning: Dropping {df['airflow_rate'].isnull().sum()} rows with NaN target.")
         df = df.dropna(subset=["airflow_rate"])
         if df.empty: print("Error: All rows dropped."); return
    y = df["airflow_rate"].astype(float)

    # --- Select initial X based on chosen columns ---
    # Ensure columns exist before selecting
    available_cols = [col for col in current_feature_columns if col in df.columns]
    if "delta_T" not in available_cols: print("Error: delta_T column missing!"); return
    print(f"Available selected features in DataFrame: {available_cols}")
    X = df[available_cols].copy()

    # --- Calculate Normalized Features (if base features exist) ---
    add_normalized_rate = True # Default to True if rate is selected
    add_normalized_magnitude = True # Default to True if magnitude is selected
    add_normalized_max_inst_rate = False # Default to False unless explicitly selected later
    add_normalized_activity_mean = True # Add norm activity? Maybe useful
    add_normalized_activity_median = True
    add_normalized_activity_std = False # Std probably less useful normalized
    add_normalized_activity_max = True
    add_normalized_activity_sum = True

    if add_normalized_rate and 'hotspot_temp_change_rate' in X.columns:
         X['hotspot_temp_change_rate_norm'] = df.apply(lambda r: r['hotspot_temp_change_rate'] / r['delta_T'] if r['delta_T']!=0 else np.nan, axis=1)
         if 'hotspot_temp_change_rate_norm' not in current_feature_columns: current_feature_columns.append('hotspot_temp_change_rate_norm')
    if add_normalized_magnitude and 'hotspot_temp_change_magnitude' in X.columns:
         X['hotspot_temp_change_magnitude_norm'] = df.apply(lambda r: r['hotspot_temp_change_magnitude'] / r['delta_T'] if r['delta_T']!=0 else np.nan, axis=1)
         if 'hotspot_temp_change_magnitude_norm' not in current_feature_columns: current_feature_columns.append('hotspot_temp_change_magnitude_norm')
    # Add normalization for NEW activity features
    if add_normalized_activity_mean and 'activity_mean' in X.columns:
        X['activity_mean_norm'] = df.apply(lambda r: r['activity_mean'] / r['delta_T'] if r['delta_T']!=0 else np.nan, axis=1)
        if 'activity_mean_norm' not in current_feature_columns: current_feature_columns.append('activity_mean_norm')
    if add_normalized_activity_median and 'activity_median' in X.columns:
        X['activity_median_norm'] = df.apply(lambda r: r['activity_median'] / r['delta_T'] if r['delta_T']!=0 else np.nan, axis=1)
        if 'activity_median_norm' not in current_feature_columns: current_feature_columns.append('activity_median_norm')
    # if add_normalized_activity_std and 'activity_std' in X.columns: # Std might not make sense normalized by delta_T
    #     X['activity_std_norm'] = df.apply(lambda r: r['activity_std'] / r['delta_T'] if r['delta_T']!=0 else np.nan, axis=1)
    #     if 'activity_std_norm' not in current_feature_columns: current_feature_columns.append('activity_std_norm')
    if add_normalized_activity_max and 'activity_max' in X.columns:
        X['activity_max_norm'] = df.apply(lambda r: r['activity_max'] / r['delta_T'] if r['delta_T']!=0 else np.nan, axis=1)
        if 'activity_max_norm' not in current_feature_columns: current_feature_columns.append('activity_max_norm')
    if add_normalized_activity_sum and 'activity_sum' in X.columns:
        X['activity_sum_norm'] = df.apply(lambda r: r['activity_sum'] / r['delta_T'] if r['delta_T']!=0 else np.nan, axis=1)
        if 'activity_sum_norm' not in current_feature_columns: current_feature_columns.append('activity_sum_norm')


    # --- Apply Log Transform ---
    # Apply only to selected features if they exist
    print("\nApplying Log Transform (log1p)...")
    # Flag to control if log transform is applied
    apply_log_transform_deltaT = True
    apply_log_transform_area = True # Only applies if area is selected

    if apply_log_transform_deltaT and 'delta_T' in X.columns:
        if (df['delta_T'] <= 0).any(): print("Warning: Non-positive delta_T values.")
        X['delta_T_log'] = np.log1p(df['delta_T'].astype(float)) # Use original delta_T from df
        current_feature_columns.append('delta_T_log') # Add new name
        if 'delta_T' in current_feature_columns: current_feature_columns.remove('delta_T') # Remove old name
        print("  Applied to delta_T (created delta_T_log)")

    if apply_log_transform_area and 'hotspot_area' in X.columns and 'hotspot_area' in current_feature_columns:
        if (df['hotspot_area'] < 0).any(): print("Warning: Negative hotspot_area values.")
        X['hotspot_area_log'] = np.log1p(df['hotspot_area'].astype(float).clip(lower=0))
        current_feature_columns.append('hotspot_area_log') # Add new name
        if 'hotspot_area' in current_feature_columns: current_feature_columns.remove('hotspot_area') # Remove old name
        print("  Applied to hotspot_area (created hotspot_area_log)")

    # --- Final Feature Selection for X ---
    # Select only the final desired columns (including transformed/normalized)
    current_feature_columns.remove('hotspot_temp_change_rate')
    current_feature_columns.remove('hotspot_temp_change_magnitude')
    #current_feature_columns.remove('hotspot_max_inst_rate')
    current_feature_columns.remove('activity_mean')
    current_feature_columns.remove('activity_median')
    current_feature_columns.remove('activity_max')
    
    final_feature_columns = [col for col in current_feature_columns if col in X.columns]
    X = X[final_feature_columns]
    print(f"\nFinal Features for Model: {list(X.columns)}")


    # --- Handle NaNs/Infs and Final Checks ---
    if X.isnull().any().any() or np.isinf(X.values).any():
         print("\nWarning: NaN or Inf values detected in X BEFORE imputation.")
         # print("NaN Counts:\n", X.isnull().sum()[X.isnull().sum() > 0]) # Less verbose
         # print(f"Inf count: {np.isinf(X.values).sum()}")
    # Drop columns that are ALL NaN
    cols_before_drop = X.shape[1]
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        print(f"\nWarning: Dropping all-NaN feature columns: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
        print(f"  X shape changed from {len(y)}x{cols_before_drop} to {X.shape}")
    # Final checks
    if X.empty or y.empty: print("Error: X or y is empty."); return
    if X.shape[1] == 0: print("Error: X has no columns left."); return
    if X.shape[0] != len(y): print(f"Error: Mismatch X rows ({X.shape[0]}) vs y length ({len(y)})."); return

    print(f"Final Feature matrix X shape: {X.shape}")
    print(f"Final Target vector y shape: {y.shape}")

    # 4. Hyperparameter Tuning
    print("\n--- Running Hyperparameter Tuning ---")
    # Pass final X to tuning
    all_best_params, all_best_scores = tuning.run_grid_search_all_models(X, y)
    print("\n--- Tuning Results Summary ---")
    for name in all_best_params:
         score = all_best_scores[name]; mse = -score if np.isfinite(score) else np.inf
         rmse = np.sqrt(mse) if np.isfinite(mse) else np.inf
         # Handle cases where tuning might have failed (score is -inf)
         if np.isfinite(score):
              print(f"{name}: Best Score={score:.4f} (MSE={mse:.4f}, RMSE={rmse:.4f}), Best Params={all_best_params[name]}")
         else:
              print(f"{name}: Tuning failed or skipped.")


    # 5. Setup Cross-Validation Strategy
    # (Keep CV logic as before)
    num_samples_final = len(y); cv_strategy=None; cv_name="Unknown"
    if config.CV_METHOD == 'LeaveOneOut': cv_strategy=LeaveOneOut(); n_splits=num_samples_final; cv_name="LeaveOneOut"
    elif config.CV_METHOD == 'KFold':
        k_folds_config=getattr(config,'K_FOLDS',5); n_splits=min(k_folds_config,num_samples_final)
        if n_splits<2: cv_strategy=LeaveOneOut(); n_splits=num_samples_final; cv_name="LeaveOneOut(Fallback)"
        else: cv_strategy=KFold(n_splits=n_splits,shuffle=True,random_state=config.RANDOM_STATE); cv_name=f"{n_splits}-Fold KFold"
    else: cv_strategy=LeaveOneOut(); n_splits=num_samples_final; cv_name="LeaveOneOut(Default)"
    print(f"\nUsing {cv_name} Cross-Validation ({n_splits} splits) for evaluation.")


    # 6. Evaluate Models using TUNED parameters
    print(f"\n--- Evaluating ALL Models using Best Parameters found during Tuning (or defaults) ---")
    evaluation_results = {}
    all_regressors_to_eval = modeling.get_regressors() # Get all models

    for name, model_instance in all_regressors_to_eval.items():
        print(f"\nEvaluating model: {name}...")
        # Use parameters found during tuning, or defaults if tuning skipped/failed
        tuned_params_raw = all_best_params.get(name, model_instance.get_params()) # Get tuned or default params
        # Ensure params don't have 'model__' prefix if applying directly
        tuned_params = {k.replace('model__', ''): v for k, v in tuned_params_raw.items() if k.startswith('model__') or k in model_instance.get_params()}


        if tuned_params and name in all_best_params and np.isfinite(all_best_scores[name]): # Check if tuning was done and successful
             print(f"  Applying Tuned Parameters: {tuned_params}")
             try: model_instance.set_params(**tuned_params)
             except ValueError as e: print(f"Warning: Error setting params: {e}. Using defaults."); model_instance = modeling.get_regressors()[name]
        else:
             print(f"  Using default parameters for {name}.")
             model_instance = modeling.get_regressors()[name] # Ensure fresh default instance


        pipeline_model = modeling.build_pipeline(model_instance, pca_components=None)

        try:
            start_cv_time = time.time()
            y_pred_cv = cross_val_predict(pipeline_model, X, y, cv=cv_strategy, n_jobs=-1)
            end_cv_time = time.time()
            # print(f"  CV predictions took {end_cv_time - start_cv_time:.2f} seconds.")
            if np.isnan(y_pred_cv).any(): print(f"  Warning: NaN values found in CV predictions for {name}.")

            # Calculate metrics based on CV predictions
            mse = mean_squared_error(y, y_pred_cv); rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred_cv); r2 = r2_score(y, y_pred_cv)
            # Store results WITH the model name as key
            results_dict = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred_cv': y_pred_cv }
            evaluation_results[name] = results_dict # Store under model name

            print(f"--- CV Results for {name} ---")
            print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

        except Exception as e:
            print(f"Error during CV evaluation for {name}: {e}")
            traceback.print_exc(); evaluation_results[name] = None # Mark as failed

    # 7. Analyze Results and Plot
    valid_results = {name: res for name, res in evaluation_results.items() if res is not None}
    if not valid_results: print("\nError: All model evaluations failed."); return

    # --- MODIFIED: Select best model based on CV R² ---
    best_model_name = max(valid_results, key=lambda name: valid_results[name]['r2']) # Find model with highest R²
    best_result_metrics = valid_results[best_model_name]
    print(f"\n--- Best Model based on CV R²: {best_model_name} ---")
    print(f"MSE: {best_result_metrics['mse']:.4f} | RMSE: {best_result_metrics['rmse']:.4f} | MAE: {best_result_metrics['mae']:.4f} | R²: {best_result_metrics['r2']:.4f}")


    # --- Plotting Actual vs Predicted for the BEST model ---
    save_actual_pred_plot = getattr(config, 'SAVE_ACTUAL_VS_PREDICTED_PLOT', False)
    actual_pred_plot_path = getattr(config, 'ACTUAL_VS_PREDICTED_PLOT_SAVE_PATH', None)
    # Update title to reflect the best model selected by CV R^2
    plot_title = f'Actual vs. Predicted - {best_model_name} (Tuned*, {cv_name}-CV)'
    save_path_arg = None
    if save_actual_pred_plot and actual_pred_plot_path:
        # Ensure filename reflects the best model
        base, ext = os.path.splitext(actual_pred_plot_path)
        plot_save_path = f"{base}_{best_model_name}{ext}"
        print(f"\nSaving Actual vs Predicted plot for {best_model_name} to: {plot_save_path}")
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        save_path_arg = plot_save_path
    else: print("\nDisplaying Actual vs Predicted plot (not saving).")

    try:
        if len(y) == len(best_result_metrics['y_pred_cv']):
            plotting.plot_actual_vs_predicted(y, best_result_metrics['y_pred_cv'], plot_title, save_path_arg)
        else: print("Warning: Length mismatch for Actual vs Predicted plot.")
    except Exception as e: print(f"Error plotting Actual vs Predicted: {e}")


    # 8. Final Model Training, Saving, and Analysis (for the BEST model)
    print(f"\n--- Training final {best_model_name} regressor on all data ---")
    try:
        # Get a fresh instance of the best model type
        final_model_instance = modeling.get_regressors()[best_model_name]
        # Get the best parameters found during tuning for this model
        best_params_raw = all_best_params.get(best_model_name, {})
        # Ensure correct parameter format (remove 'model__' prefix if present from GridSearchCV)
        best_params_final = {k.replace('model__', ''): v for k, v in best_params_raw.items() if k in final_model_instance.get_params() or k.replace('model__','') in final_model_instance.get_params() }

        if best_params_final and best_model_name in all_best_params and np.isfinite(all_best_scores[best_model_name]):
            print(f"  Applying Tuned Parameters: {best_params_final}")
            final_model_instance.set_params(**best_params_final)
        else:
            print(f"  Using default parameters for final {best_model_name} model.")

        final_pipeline = modeling.build_pipeline(final_model_instance, pca_components=None)
        final_pipeline.fit(X, y)
        print("Final model training complete.")

        # --- Feature Analysis ---
        # (Keep analysis logic as before, it will now apply to the actual best model)
        print(f"\n--- Analyzing Features for Final {best_model_name} Model ---")
        if 'model' in final_pipeline.named_steps:
             fitted_model = final_pipeline.named_steps['model']
             if isinstance(fitted_model, MLPRegressor):
                  print(f"MLPRegressor: Standard coefficients/importances not directly available.")
                  # ... (rest of MLP analysis)
             elif isinstance(fitted_model, LinearRegression):
                  print("LinearRegression Coefficients:")
                  # Ensure feature names match columns *after* potential drops/transforms in X
                  coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': fitted_model.coef_})
                  print(coef_df.round(4))
                  print(f"Intercept: {fitted_model.intercept_:.4f}")
             elif isinstance(fitted_model, SVR):
                  print(f"SVR: Feature importances not directly available from coefficients.")
                  if fitted_model.kernel == 'linear':
                       print("  (Linear Kernel) Consider magnitude of `dual_coef_` or permutation importance.")
                       # print(f"  Dual Coefficients shape: {fitted_model.dual_coef_.shape}") # Example
                  else:
                       print("  (Non-Linear Kernel) Consider permutation importance.")
             else: print(f"Model type '{type(fitted_model).__name__}' analysis not implemented.")
        else: print("Error: Could not find 'model' step in pipeline.")

        # --- Plotting Training Loss Curve (Only if best model is MLP) ---
        save_loss_plot = getattr(config, 'SAVE_LOSS_CURVE_PLOT', False)
        if save_loss_plot and isinstance(final_model_instance, MLPRegressor):
             print(f"\n--- Plotting Training Loss Curve for {best_model_name} ---")
             # (Keep loss curve plotting logic as before)
             if 'model' in final_pipeline.named_steps:
                 fitted_model = final_pipeline.named_steps['model']
                 if hasattr(fitted_model, 'loss_curve_'):
                     # ... (plotting code) ...
                     pass # Keep plotting code here
                 else: print("Loss curve unavailable.")
             else: print("Error: Could not find 'model' step.")
        elif save_loss_plot: print(f"\nSkipping loss curve plot (Best model is not MLPRegressor).")
        else: print("\nSkipping saving of training loss curve plot.")

        # --- Save Final Model ---
        # Ensure filename reflects the best model
        model_base, model_ext = os.path.splitext(getattr(config, 'MODEL_SAVE_PATH', 'final_model.joblib'))
        final_model_save_path = f"{model_base}_{best_model_name}{model_ext}"
        os.makedirs(os.path.dirname(final_model_save_path), exist_ok=True)
        joblib.dump(final_pipeline, final_model_save_path)
        print(f"\nFinal {best_model_name} pipeline saved to: {final_model_save_path}")

    except Exception as e:
        print(f"Error during final {best_model_name} training/saving: {e}")
        traceback.print_exc()

    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    run_experiment()
# --- END OF FILE main.py ---