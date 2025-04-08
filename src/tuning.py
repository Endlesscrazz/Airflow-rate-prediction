# tuning.py
"""Functions for hyperparameter tuning for multiple models at once."""

from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from modeling import get_classifiers  # To get classifier instances
import config
import pandas as pd

def run_grid_search(X, y, classifier_name="RandomForest"):
    """
    Performs GridSearchCV for a specific classifier.
    The pipeline includes Imputer -> Scaler -> Model.
    Hyperparameters for the model are tuned based on the classifier.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        classifier_name: Name of the classifier from get_classifiers() to use.
        
    Returns:
        tuple: (best_params, best_score) from the GridSearchCV.
    """
    print(f"\n--- Running Grid Search for {classifier_name} ---")
    try:
        classifier = get_classifiers()[classifier_name]
    except KeyError:
        print(f"Error: Classifier '{classifier_name}' not found in get_classifiers(). Using RandomForest.")
        classifier = get_classifiers()["RandomForest"]

    # Basic pipeline: Imputer -> Scaler -> Model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', classifier)
    ])

    # Define parameter grid based on classifier
    if classifier_name == "RandomForest":
        param_grid = {
            'model__n_estimators': [50, 100, 150],
            'model__max_depth': [3, 5, None],
            'model__min_samples_split': [2, 5]
        }
    elif classifier_name == "GradientBoosting":
        param_grid = {
            'model__n_estimators': [50, 100, 150],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, None]
        }
    elif classifier_name == "SVC":
        param_grid = {
            'model__C': [1.0, 5.0, 10.0, 15.0, 20.0],
            'model__kernel': ['linear'],
            #'model__gamma': ['scale', 'auto']
        }
    else:
        param_grid = {}
    
    if not param_grid:
        print("Warning: Parameter grid is empty. GridSearch will only fit the default pipeline.")
    
    # Determine CV strategy based on config
    if config.CV_METHOD == 'LeaveOneOut':
        cv_strategy = LeaveOneOut()  # Use LeaveOneOut instance
    elif config.CV_METHOD == 'StratifiedKFold':
        min_class_count = pd.Series(y).value_counts().min()
        n_splits = max(2, min(config.K_FOLDS, min_class_count))
        cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    else:
        print(f"Invalid CV_METHOD '{config.CV_METHOD}'. Defaulting to 5-fold StratifiedKFold.")
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    print(f"Using {config.CV_METHOD} for GridSearch CV.")
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)
    
    try:
        grid_search.fit(X, y)
        print(f"GridSearch completed for {classifier_name}. Best Score (f1_weighted): {grid_search.best_score_:.4f}")
        if grid_search.best_params_:
            print(f"Best Parameters for {classifier_name}: {grid_search.best_params_}")
        else:
            print("No hyperparameters were tuned (param_grid was empty).")
        return grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error during GridSearchCV for {classifier_name}: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return {}, -1.0

def run_grid_search_all_models(X, y):
    """
    Performs grid search for multiple models at once.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        
    Returns:
        dict: A dictionary mapping each classifier name to its best parameters and score.
              Example: {"RandomForest": {"best_params": {...}, "best_score": 0.45}, ...}
    """
    results = {}
    classifiers = get_classifiers()
    for name in classifiers.keys():
        best_params, best_score = run_grid_search(X, y, classifier_name=name)
        results[name] = {"best_params": best_params, "best_score": best_score}
    return results
