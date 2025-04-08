# modeling.py
"""Functions for defining models and building the ML pipeline."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression # Added
from sklearn.feature_selection import RFECV

import config

def build_pipeline(model, pca_components=None):
    """
    Builds the standard ML pipeline with imputation, scaling, optional PCA, and model.

    Args:
        model: The scikit-learn classifier instance.
        pca_components: Number of PCA components or variance ratio. If None, PCA is skipped.
                        NOTE: pca_components from config is checked *if* this arg is None.
                        Best practice is to pass None explicitly if PCA is not desired.

    Returns:
        sklearn.pipeline.Pipeline: The configured pipeline.
    """
    steps = []
    steps.append(('imputer', SimpleImputer(strategy='mean'))) # Handle NaNs
    steps.append(('scaler', StandardScaler()))               # Scale features

    # Determine if PCA should be used based on argument or config (but arg takes precedence)
    use_pca = False
    n_components = None
    if pca_components is not None:
    # Validate n_components if PCA is explicitly requested via argument
        valid_n = False
        if isinstance(pca_components, int) and pca_components > 0:
            valid_n = True
        elif isinstance(pca_components, float) and 0 < pca_components <= 1.0:
            valid_n = True # Variance ratio

        if valid_n:
            print(f"  Adding PCA step with n_components={pca_components} (from argument)")
            steps.append(('pca', PCA(n_components=pca_components, random_state=config.RANDOM_STATE)))
        else:
            print(f"  Warning: Invalid PCA n_components ({pca_components}) passed as argument. Disabling PCA step.")
    # Only check config IF pca_components argument was None (i.e., not explicitly set)
    elif config.PCA_N_COMPONENTS is not None:
        # Validate n_components from config
        valid_n = False
        n_components_config = config.PCA_N_COMPONENTS
        if isinstance(n_components_config, int) and n_components_config > 0:
            valid_n = True
        elif isinstance(n_components_config, float) and 0 < n_components_config <= 1.0:
            valid_n = True # Variance ratio

        if valid_n:
            print(f"  Adding PCA step with n_components={n_components_config} (from config.py)")
            steps.append(('pca', PCA(n_components=n_components_config, random_state=config.RANDOM_STATE)))
        else:
            print(f"  Warning: Invalid PCA n_components ({n_components_config}) in config.py. Disabling PCA step.")
    # else: # Optional: Add confirmation that PCA is off if both are None
    #    print("  PCA step is disabled (pca_components=None and config.PCA_N_COMPONENTS=None).")

    steps.append(('model', model)) # The classifier
    return Pipeline(steps)

def build_ensemble_pipeline(ensemble_model, pca_components=None, use_rfecv=False, rfecv_estimator=None):
    """
    Builds an ML pipeline that integrates imputation, scaling, optional recursive feature selection (RFECV),
    optional PCA, and an ensemble classifier.
    NOTE: RFECV is now OFF by default (use_rfecv=False).

    Args:
        ensemble_model: The ensemble classifier (e.g., VotingClassifier or StackingClassifier).
        pca_components: The number (or variance ratio) of components for PCA. If None, check config.
        use_rfecv: Boolean flag to include RFECV for feature selection (Default: False).
        rfecv_estimator: The base estimator to use for RFECV. If None, defaults to RandomForestClassifier.

    Returns:
        sklearn.pipeline.Pipeline: The configured ensemble pipeline.
    """
    steps = []
    steps.append(('imputer', SimpleImputer(strategy='mean')))
    steps.append(('scaler', StandardScaler()))

    if use_rfecv:
        print("  Adding RFECV step (Warning: computationally expensive and potentially unstable on small N)")
        # Use a default estimator for RFECV if none is provided.
        if rfecv_estimator is None:
            # Use a simpler estimator for RFECV by default maybe? Or keep RF?
            rfecv_estimator = RandomForestClassifier(n_estimators=50, # Fewer estimators might speed up RFECV
                                                     random_state=config.RANDOM_STATE,
                                                     class_weight='balanced')
            print(f"  Using default RFECV estimator: {type(rfecv_estimator).__name__}")
        # Consider adjusting cv within RFECV, maybe 3 folds if outer CV is 5? Or keep 5?
        rfecv = RFECV(estimator=rfecv_estimator, step=1, cv=3, scoring='f1_weighted', n_jobs=-1) # Use more cores if available
        steps.append(('feature_selection', rfecv))
    else:
         print("  Skipping RFECV step.")

    # Determine if PCA should be used based on argument or config (but arg takes precedence)
    use_pca = False
    n_components = None
    if pca_components is not None:
        use_pca = True
        n_components = pca_components
    elif config.PCA_N_COMPONENTS is not None:
        use_pca = True
        n_components = config.PCA_N_COMPONENTS
        print("Info: Using PCA from config file as pca_components argument was None.")

    # Optional: PCA step
    if use_pca:
        valid_n = False
        if isinstance(n_components, int) and n_components > 0:
            valid_n = True
        elif isinstance(n_components, float) and 0 < n_components <= 1.0:
            valid_n = True # Variance ratio

        if valid_n:
            print(f"  Adding PCA step with n_components={n_components}")
            steps.append(('pca', PCA(n_components=n_components, random_state=config.RANDOM_STATE)))
        else:
            print(f"  Warning: Invalid PCA n_components ({n_components}). Disabling PCA step.")


    steps.append(('ensemble', ensemble_model))

    return Pipeline(steps)


def get_classifiers():
    """
    Returns a dictionary of configured classifier instances.
    Includes RandomForest, GradientBoosting, SVC, and LogisticRegression.

    Returns:
        dict: Dictionary where keys are model names and values are classifier objects.
    """
    classifiers = {
        "LogisticRegression": LogisticRegression(
            random_state=config.RANDOM_STATE,
            class_weight='balanced', # Good for small/potentially imbalanced datasets
            max_iter=1000 # Increase max_iter for convergence if needed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            class_weight='balanced' # Good for small/potentially imbalanced datasets
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE
            # Note: GB doesn't have simple 'class_weight'. Requires sample weighting if needed.
            # Might be more prone to overfitting on N=20 than RF or LogReg.
        ),
        "SVC": SVC(
            probability=True, # Useful for some analyses, slightly slower. Needed for soft voting if re-enabled.
            random_state=config.RANDOM_STATE,
            class_weight='balanced' # Good for small/potentially imbalanced datasets
        )
    }
    return classifiers