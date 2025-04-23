# --- START OF FILE tuning.py ---

# tuning.py
"""Hyperparameter tuning using GridSearchCV."""

import config
import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold

# --- Import only MLPRegressor and necessary modules ---
from sklearn.neural_network import MLPRegressor
from modeling import build_pipeline # To build pipeline for grid search
from sklearn.metrics import mean_squared_error # For scoring if needed

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Define ONLY the parameter grid for MLPRegressor ---
def get_param_grids():
    """Returns parameter grids specifically for MLPRegressor,
       adjusted for small dataset size."""
    param_grids = {
        "MLPRegressor": {
            # Note: Parameter names must start with 'model__' due to the pipeline

            # --- MODIFIED: Focus on simpler architectures ---
            'model__hidden_layer_sizes': [
                (10,),        # Very simple single layer
                (20,),
                (50,),
                (10, 5),      # Simple two-layer
                (20, 10),
                (50, 20),
                # (100,),     # Optionally keep slightly larger ones if needed
                # (100, 50)   # but prioritize smaller ones for N=22
                ],

            # --- Keep common activation functions ---
            'model__activation': ['relu', 'tanh'],

            'model__batch_size': [8, 16, 'auto'],

            # --- Keep Adam optimizer ---
            'model__solver': ['adam'],

            # --- MODIFIED: Increase regularization range ---
            'model__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0], # L2 regularization strength

            # --- MODIFIED: Refine learning rate options ---
            'model__learning_rate_init': [0.0005, 0.001, 0.005, 0.01] # Added lower, kept common range
        }
    }
    # --- Grids for other models REMOVED ---
    return param_grids


def run_grid_search(X, y, model_name, model_instance, param_grid, cv_strategy):
    """Runs GridSearchCV for a single model."""
    print(f"\n--- Running Grid Search for {model_name} ---")

    if not param_grid:
        print(f"Warning: Parameter grid for {model_name} is empty. GridSearch will only fit the default pipeline.")
        # Fit default pipeline once to get a score
        pipeline = build_pipeline(model_instance, pca_components=config.PCA_N_COMPONENTS)
        try:
             # Need cross_val_score for scoring default pipeline
             from sklearn.model_selection import cross_val_score
             # Use neg_mean_squared_error for scoring
             scores = cross_val_score(pipeline, X, y, cv=cv_strategy, scoring='neg_mean_squared_error', n_jobs=-1)
             best_score = np.mean(scores)
             best_params = {} # No params tuned
             print(f"Fitted default pipeline for {model_name}. Average Score (neg_mean_squared_error): {best_score:.4f}")
             return best_params, best_score
        except Exception as e:
             print(f"Error fitting default pipeline for {model_name}: {e}")
             return {}, -np.inf # Return worst score on error
    else:
         # Build pipeline for GridSearch
         pipeline = build_pipeline(model_instance, pca_components=config.PCA_N_COMPONENTS)
         # Use neg_mean_squared_error for scoring (consistent with config)
         scoring_metric = getattr(config, 'BEST_MODEL_METRIC_SCORE', 'neg_mean_squared_error') # Get metric from config
         print(f"Using {type(cv_strategy).__name__} with '{scoring_metric}' scoring for GridSearch CV.")

         # Set verbose=0 to reduce excessive grid search output, or keep verbose=1 if you want fold details
         grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring=scoring_metric, n_jobs=-1, verbose=0)

         try:
             grid_search.fit(X, y)
             best_score = grid_search.best_score_
             best_params = grid_search.best_params_

             # Calculate corresponding MSE/RMSE from the best score
             # This assumes the score is neg_mean_squared_error, adjust if metric changes
             if scoring_metric == 'neg_mean_squared_error':
                 best_mse = -best_score if best_score != -np.inf else np.inf
                 best_rmse = np.sqrt(best_mse) if best_mse != np.inf else np.inf
             else:
                 # Set to NaN if we don't know how to convert the score
                 best_mse = np.nan
                 best_rmse = np.nan
                 print(f"Note: MSE/RMSE calculation assumes scoring='neg_mean_squared_error'. Current score: {scoring_metric}")


             print(f"GridSearch completed for {model_name}. Best Score ({scoring_metric}): {best_score:.4f}")
             # Only print MSE/RMSE if calculated
             if not np.isnan(best_mse):
                 print(f"  Corresponding Best MSE: {best_mse:.4f}")
                 print(f"  Corresponding Best RMSE: {best_rmse:.4f}")
             print(f"Best Parameters for {model_name}: {best_params}")
             return best_params, best_score

         except Exception as e:
             print(f"Error during GridSearchCV for {model_name}: {e}")
             import traceback
             traceback.print_exc()
             return {}, -np.inf # Return worst score on error

def run_grid_search_all_models(X, y):
    """Runs GridSearchCV for all models defined in modeling.get_regressors()."""
    from modeling import get_regressors # Get the updated regressors list (should be just MLP)

    all_regressors = get_regressors()
    all_param_grids = get_param_grids() # Get updated grids
    all_best_params = {}
    all_best_scores = {}

    # Setup CV Strategy (same as in main.py for consistency)
    num_samples = len(y)
    if config.CV_METHOD == 'LeaveOneOut':
        cv_strategy_tuning = LeaveOneOut()
    elif config.CV_METHOD == 'KFold':
        k_folds_config = getattr(config, 'K_FOLDS', 5)
        n_splits_tuning = min(k_folds_config, num_samples)
        if n_splits_tuning < 2:
             print(f"Warning (Tuning): K_FOLDS too large ({k_folds_config} vs {num_samples} samples). Falling back to LeaveOneOut.")
             cv_strategy_tuning = LeaveOneOut()
        else:
             cv_strategy_tuning = KFold(n_splits=n_splits_tuning, shuffle=True, random_state=config.RANDOM_STATE)
    else:
        print(f"Warning (Tuning): Invalid CV_METHOD '{config.CV_METHOD}'. Defaulting to LeaveOneOut.")
        cv_strategy_tuning = LeaveOneOut()

    for name, model_instance in all_regressors.items(): # This loop will only run for MLPRegressor now
        param_grid = all_param_grids.get(name, {}) # Get MLP grid
        best_params, best_score = run_grid_search(X, y, name, model_instance, param_grid, cv_strategy_tuning)
        all_best_params[name] = best_params
        all_best_scores[name] = best_score

    return all_best_params, all_best_scores

# --- END OF FILE tuning.py ---