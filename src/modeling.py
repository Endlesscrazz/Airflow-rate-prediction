# modeling.py
"""Functions for defining models and building the ML pipeline for regression."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
# Import Regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

import config

def build_pipeline(model, pca_components=None):
    """
    Builds the standard ML pipeline with imputation, scaling, optional PCA, and model.
    Suitable for both regression and classification models.

    Args:
        model: The scikit-learn estimator instance (regressor or classifier).
        pca_components: Number of PCA components or variance ratio. If None, PCA is skipped.

    Returns:
        sklearn.pipeline.Pipeline: The configured pipeline.
    """
    steps = []
    steps.append(('imputer', SimpleImputer(strategy='mean'))) # Handle NaNs
    steps.append(('scaler', StandardScaler()))               # Scale features

    # --- PCA Logic (same as before, controlled by config/argument) ---
    if pca_components is not None:
        valid_n = False
        if isinstance(pca_components, int) and pca_components > 0: valid_n = True
        elif isinstance(pca_components, float) and 0 < pca_components <= 1.0: valid_n = True
        if valid_n:
            print(f"  Adding PCA step with n_components={pca_components} (from argument)")
            steps.append(('pca', PCA(n_components=pca_components, random_state=config.RANDOM_STATE)))
        else: print(f"  Warning: Invalid PCA n_components ({pca_components}) passed as argument. Disabling PCA step.")
    elif config.PCA_N_COMPONENTS is not None:
        valid_n = False
        n_components_config = config.PCA_N_COMPONENTS
        if isinstance(n_components_config, int) and n_components_config > 0: valid_n = True
        elif isinstance(n_components_config, float) and 0 < n_components_config <= 1.0: valid_n = True
        if valid_n:
            print(f"  Adding PCA step with n_components={n_components_config} (from config.py)")
            steps.append(('pca', PCA(n_components=n_components_config, random_state=config.RANDOM_STATE)))
        else: print(f"  Warning: Invalid PCA n_components ({n_components_config}) in config.py. Disabling PCA step.")
    # --- End PCA Logic ---

    steps.append(('model', model)) # The regressor
    return Pipeline(steps)

# Removed build_ensemble_pipeline as it was classification-focused

def get_regressors():
    """
    Returns a dictionary of configured regressor instances.

    Returns:
        dict: Dictionary where keys are model names and values are regressor objects.
    """
    regressors = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=config.RANDOM_STATE),
        "Lasso": Lasso(random_state=config.RANDOM_STATE),
        "SVR": SVR(), # Kernels (linear, rbf) and C/epsilon are key tuning params
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            n_jobs=-1 # Use all cores
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE
        )
    }
    return regressors