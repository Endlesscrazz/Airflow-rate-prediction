# modeling.py
"""Builds machine learning models and pipelines."""

import config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --- Import Only MLPRegressor ---
from sklearn.neural_network import MLPRegressor

# --- Removed other model imports ---
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# --- Updated function to return only MLPRegressor ---
def get_regressors():
    """Returns a dictionary containing only the MLPRegressor."""
    # Increase max_iter for convergence, adjust other defaults if needed
    # Add early_stopping=True, validation_fraction=0.1, n_iter_no_change=10 for robustness?
    regressors = {
        "MLPRegressor": MLPRegressor(
            random_state=config.RANDOM_STATE,
            max_iter=3000, # Increase max iterations
            early_stopping=True, # Stop training when validation score is not improving
            validation_fraction=0.1, # Use 10% of training data for validation
            n_iter_no_change=10, # How many iterations with no improvement to wait
            warm_start=False # Start fresh each fit
        )
    }
    return regressors

def build_pipeline(model, pca_components=None):
    """Builds a standard pipeline: Imputer -> Scaler -> Model."""
    steps = []
    # 1. Impute missing values (essential)
    steps.append(('imputer', SimpleImputer(strategy='mean'))) # Mean imputation is common

    # 2. Scale features (CRITICAL for Neural Networks)
    steps.append(('scaler', StandardScaler()))

    # 3. PCA (Currently Disabled in config)
    if pca_components is not None and isinstance(pca_components, int) and pca_components > 0:
        from sklearn.decomposition import PCA
        steps.append(('pca', PCA(n_components=pca_components, random_state=config.RANDOM_STATE)))
        print(f"--- Pipeline: PCA Enabled with n_components={pca_components} ---")
    else:
         print("--- Pipeline: PCA Disabled ---")


    # 4. Add the regressor model
    steps.append(('model', model))

    return Pipeline(steps)