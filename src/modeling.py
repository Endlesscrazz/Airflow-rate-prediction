# modeling.py
"""Functions for defining models and building the ML pipeline."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

import config # Import configuration

def build_pipeline(model):
    """
    Builds the standard ML pipeline with imputation, scaling, optional PCA, and model.

    Args:
        model: The scikit-learn classifier instance.

    Returns:
        sklearn.pipeline.Pipeline: The configured pipeline.
    """
    steps = []
    steps.append(('imputer', SimpleImputer(strategy='mean'))) # Handle NaNs
    steps.append(('scaler', StandardScaler()))               # Scale features

    # Optional: PCA for dimensionality reduction
    if config.PCA_N_COMPONENTS is not None:
         # Ensure n_components is valid (less than min(n_samples, n_features))
         # This will be handled dynamically during CV fit, but good practice
         pca_components = config.PCA_N_COMPONENTS
         # If it's a float (variance ratio), leave it as is.
         # If it's an int, ensure it's positive.
         if isinstance(pca_components, int) and pca_components <= 0:
             print(f"Warning: PCA n_components ({pca_components}) is not positive. Disabling PCA.")
             pca_components = None
         if pca_components is not None:
            steps.append(('pca', PCA(n_components=pca_components, random_state=config.RANDOM_STATE)))

    steps.append(('model', model)) # The classifier
    return Pipeline(steps)

def get_classifiers():
    """
    Returns a dictionary of configured classifier instances.

    Returns:
        dict: Dictionary where keys are model names and values are classifier objects.
    """
    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            class_weight='balanced' # Good for small/potentially imbalanced datasets
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE
            # Note: GB doesn't have simple 'class_weight'. Requires sample weighting if needed.
        ),
        "SVC": SVC(
            probability=True, # Useful for some analyses, slightly slower
            random_state=config.RANDOM_STATE,
            class_weight='balanced' # Good for small/potentially imbalanced datasets
        )
    }
    return classifiers