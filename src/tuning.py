# tuning.py
"""Functions for hyperparameter tuning for regression models."""

from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold # Use KFold for regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestRegressor # Example estimator used below
from modeling import get_regressors # Import regressors
import config
import pandas as pd
import numpy as np

def run_grid_search(X, y, regressor_name="RandomForestRegressor"):
    """
    Performs GridSearchCV for a specific regressor.
    The pipeline includes Imputer -> Scaler -> Model.
    Hyperparameters for the model are tuned based on the regressor.

    Args:
        X: Feature matrix.
        y: Target vector (continuous).
        regressor_name: Name of the regressor from get_regressors() to use.

    Returns:
        tuple: (best_params, best_score) from the GridSearchCV. Score is neg_mean_squared_error.
    """
    print(f"\n--- Running Grid Search for {regressor_name} ---")
    try:
        regressor = get_regressors()[regressor_name]
    except KeyError:
        print(f"Error: Regressor '{regressor_name}' not found in get_regressors(). Using RandomForestRegressor.")
        regressor = get_regressors()["RandomForestRegressor"]

    # Basic pipeline: Imputer -> Scaler -> Model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', regressor)
    ])

    # Define parameter grid based on regressor
    param_grid = {} # Default empty
    if regressor_name == "Ridge":
        param_grid = {
            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0] # Regularization strength
        }
    elif regressor_name == "Lasso":
         param_grid = {
            'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0] # Regularization strength
        }
    elif regressor_name == "SVR":
        param_grid = {
            'model__C': [0.1, 1.0, 10.0, 100.0], # Regularization parameter
            'model__kernel': ['linear', 'rbf'],
            'model__epsilon': [0.01, 0.1, 0.5] # Margin of tolerance
            # 'model__gamma': ['scale', 'auto'] # Only for 'rbf', 'poly', 'sigmoid'
        }
    elif regressor_name == "RandomForestRegressor":
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_split': [2, 5, 10]
        }
    elif regressor_name == "GradientBoostingRegressor":
         param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7]
         }
    # Add other regressors if needed

    if not param_grid:
        print(f"Warning: Parameter grid for {regressor_name} is empty. GridSearch will only fit the default pipeline.")

    # Determine CV strategy based on config
    if config.CV_METHOD == 'LeaveOneOut':
        cv_strategy = LeaveOneOut()
        cv_name = "LeaveOneOut"
    elif config.CV_METHOD == 'KFold':
        # Use KFold for regression (StratifiedKFold is for classification)
        n_splits = config.K_FOLDS
        cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
        cv_name = f"{n_splits}-Fold KFold"
    else:
        print(f"Warning: Invalid CV_METHOD '{config.CV_METHOD}' for regression. Defaulting to 5-Fold KFold.")
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
        cv_name = "5-Fold KFold"

    # Use neg_mean_squared_error (higher is better, less negative) or r2
    # MSE is often preferred for model selection as it directly penalizes large errors
    scoring_metric = 'neg_mean_squared_error'
    print(f"Using {cv_name} with '{scoring_metric}' scoring for GridSearch CV.")

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring=scoring_metric, n_jobs=-1)

    try:
        grid_search.fit(X, y)
        # Best score is neg_mse, convert back to positive MSE for reporting if desired
        best_neg_mse = grid_search.best_score_
        best_mse = -best_neg_mse if scoring_metric == 'neg_mean_squared_error' else None # Handle other metrics if used

        print(f"GridSearch completed for {regressor_name}. Best Score ({scoring_metric}): {best_neg_mse:.4f}")
        if best_mse is not None:
             print(f"  Corresponding Best MSE: {best_mse:.4f}")
             print(f"  Corresponding Best RMSE: {np.sqrt(best_mse):.4f}") # Also show RMSE
        if grid_search.best_params_:
            print(f"Best Parameters for {regressor_name}: {grid_search.best_params_}")
        else:
            print("No hyperparameters were tuned (param_grid was empty).")
        # Return best params and the NEGATIVE MSE score (as GridSearchCV returns)
        return grid_search.best_params_, best_neg_mse
    except Exception as e:
        print(f"Error during GridSearchCV for {regressor_name}: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        # Return default/error values, ensure score is comparable (e.g., large negative number for neg_mse)
        return {}, -np.inf

# Keep run_grid_search_all_models function (it calls the modified run_grid_search)
def run_grid_search_all_models(X, y):
    """
    Performs grid search for multiple regressors.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        tuple: (all_best_params, all_best_scores) where scores are neg_mean_squared_error.
    """
    all_best_params = {}
    all_best_scores = {}
    regressors = get_regressors()
    for name in regressors.keys():
        best_params, best_score = run_grid_search(X, y, regressor_name=name)
        all_best_params[name] = best_params
        all_best_scores[name] = best_score # Store the neg_mse score
    return all_best_params, all_best_scores