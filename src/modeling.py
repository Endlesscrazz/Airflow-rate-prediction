# modeling.py
"""Builds machine learning models and pipelines."""

import config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --- Import Regressors ---
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso,Ridge # Added
from sklearn.svm import SVR                     # Added
# Removed other imports if they existed

# --- Updated function to return selected regressors ---
def get_regressors():
    """Returns a dictionary containing selected regressors."""
    regressors = {}

    # a) Linear Regression (Simple Baseline)
    #regressors["LinearRegression"] = LinearRegression()
   #regressors["Lasso"] = Lasso()

    #regressors["Ridge"] = Ridge()

    # b) Support Vector Regression (Handles non-linearity)
    # Default parameters are often a good starting point, tune C and epsilon later
    regressors["SVR"] = SVR() # Can add kernel='rbf', C=1.0, epsilon=0.1 etc.

    # c) MLP Regressor (Simplified Architecture)
    regressors["MLPRegressor"] = MLPRegressor(
        # --- Simplified Architecture ---
        hidden_layer_sizes=(10, 5), # Reduced layers/neurons
        # --- Other Parameters ---
        activation='relu', # ReLU is often a good default
        solver='adam',
        alpha=0.001, # Slightly increased regularization default
        batch_size='auto',
        learning_rate_init=0.001, # Standard default
        max_iter=3000, # Keep higher iterations
        early_stopping=True,
        validation_fraction=0.15, # Slightly larger validation set for small data
        n_iter_no_change=20, # Wait a bit longer
        random_state=config.RANDOM_STATE,
        warm_start=False
    )

    print(f"Selected Regressors: {list(regressors.keys())}")
    return regressors

def build_pipeline(model, pca_components=None):
    """Builds a standard pipeline: Imputer -> Scaler -> Model."""
    steps = []
    # 1. Impute missing values
    steps.append(('imputer', SimpleImputer(strategy='mean')))

    # 2. Scale features (Important for SVR and MLP)
    steps.append(('scaler', StandardScaler()))

    # 3. PCA (Optional)
    if pca_components is not None and isinstance(pca_components, int) and pca_components > 0:
        from sklearn.decomposition import PCA
        steps.append(('pca', PCA(n_components=pca_components, random_state=config.RANDOM_STATE)))
        # print(f"--- Pipeline: PCA Enabled with n_components={pca_components} ---") # Reduce verbosity
    # else:
        # print("--- Pipeline: PCA Disabled ---") # Reduce verbosity


    # 4. Add the regressor model
    steps.append(('model', model))

    return Pipeline(steps)