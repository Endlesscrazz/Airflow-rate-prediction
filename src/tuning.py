# tuning.py
"""Hyperparameter tuning using GridSearchCV."""

import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error
import config
import modeling # Import your modeling script
import time
import traceback

# --- Parameter Grids ---

# Define grids for the models included in modeling.get_regressors()
# These are EXAMPLES - adjust ranges based on your understanding and initial results!

# Linear Regression usually doesn't have hyperparameters tuned in this way
# (Tuning often involves feature selection/regularization handled differently)
# param_grid_lr = {} # Empty grid

# SVR parameter grid (EXAMPLE - C, epsilon, kernel are common)
param_grid_svr = {
    'model__C': [0.1, 1, 10, 50], # Regularization parameter
    'model__epsilon': [0.01, 0.1, 0.2], # Epsilon in epsilon-SVR
    'model__kernel': ['rbf',] # Kernel type
    # 'model__gamma': ['scale', 'auto'] # Only for 'rbf', 'poly', 'sigmoid'
}

# MLP Regressor parameter grid (Refined based on simplified architecture)
param_grid_mlp = {
    'model__hidden_layer_sizes': [(10,), (30, 15), (15, 5), (20,10,5)], # Try different depths/widths
    'model__activation': ['relu', 'tanh'],
    'model__solver': ['adam'], # Adam is generally good
    'model__alpha': [0.001, 0.01, 0.5], # Regularization strength
    'model__learning_rate_init': [0.001, 0.01],
    'model__batch_size': [4, 8] # Smaller batch sizes for small dataset
}

# Combine grids into a dictionary
# Add only grids for models returned by modeling.get_regressors()
param_grids = {
    # "LinearRegression": param_grid_lr, # Exclude LR from grid search
    "SVR": param_grid_svr,
    "MLPRegressor": param_grid_mlp
}

# --- Tuning Function ---

def run_grid_search_all_models(X, y):
    """
    Performs GridSearchCV for all models defined in modeling.py for which
    a parameter grid is provided in tuning.py.
    """
    all_regressors = modeling.get_regressors()
    best_params = {}
    best_scores = {}

    # Define CV strategy based on config (as in main.py)
    num_samples = len(y)
    if config.CV_METHOD == 'LeaveOneOut':
        # GridSearchCV with LOO can be VERY slow, especially for MLP/SVR. Consider KFold.
        cv_strategy = LeaveOneOut()
        n_splits = num_samples
        cv_name = "LeaveOneOut"
        print(f"Warning: Using GridSearchCV with LeaveOneOut ({n_splits} splits). This can be very time-consuming.")
    elif config.CV_METHOD == 'KFold':
        k_folds_config = getattr(config, 'K_FOLDS', 5)
        n_splits = min(k_folds_config, num_samples)
        if n_splits < 2:
            cv_strategy = LeaveOneOut(); n_splits = num_samples; cv_name = "LeaveOneOut (Fallback)"
        else:
            cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
            cv_name = f"{n_splits}-Fold KFold"
    else: # Default to LOO if config is unclear
        cv_strategy = LeaveOneOut(); n_splits = num_samples; cv_name = "LeaveOneOut (Default)"
        print(f"Warning: Unknown CV_METHOD '{config.CV_METHOD}'. Defaulting to LeaveOneOut for GridSearchCV.")

    print(f"\n--- Running Grid Search using {cv_name} ({n_splits} splits) ---")

    for name, model in all_regressors.items():
        if name not in param_grids:
            print(f"\n--- Skipping Grid Search for {name} (no param_grid defined) ---")
            # Store default parameters and a placeholder score (e.g., NaN or -inf)
            best_params[name] = model.get_params() # Get default params
            best_scores[name] = -np.inf # Or np.nan
            continue

        print(f"\n--- Running Grid Search for {name} ---")
        start_time = time.time()
        pipeline = modeling.build_pipeline(model, pca_components=None) # Build pipeline for the current model
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=cv_strategy,
            scoring='neg_mean_squared_error', # Optimize for lower MSE
            n_jobs=-1, # Use all available CPU cores
            verbose=1 # Show some progress
        )

        try:
            grid_search.fit(X, y)
            end_time = time.time()
            print(f"GridSearch completed for {name} in {end_time - start_time:.2f} seconds.")
            print(f"  Best Score (neg_mean_squared_error): {grid_search.best_score_:.4f}")
            # Convert score back to positive MSE/RMSE for easier interpretation
            best_mse = -grid_search.best_score_
            best_rmse = np.sqrt(best_mse)
            print(f"  Corresponding Best MSE: {best_mse:.4f}")
            print(f"  Corresponding Best RMSE: {best_rmse:.4f}")
            print(f"  Best Parameters Found: {grid_search.best_params_}")
            best_params[name] = grid_search.best_params_
            best_scores[name] = grid_search.best_score_
        except Exception as e:
            print(f"!!! Error during GridSearchCV for {name}: {e}")
            traceback.print_exc()
            best_params[name] = {} # Indicate failure
            best_scores[name] = -np.inf # Indicate failure

    return best_params, best_scores