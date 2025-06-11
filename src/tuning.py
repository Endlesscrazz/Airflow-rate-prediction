# tuning.py
"""Hyperparameter tuning using GridSearchCV.
Includes function for Nested Cross-Validation.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, RepeatedKFold
from sklearn.base import clone
import config
import modeling # Import your modeling script
import time
import traceback
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # For nested CV scoring
from sklearn.exceptions import ConvergenceWarning # To selectively ignore for MLP
import warnings

# --- Parameter Grids (No feature_selector__k) ---
param_grid_svr = {
    'model__C': [1, 10, 50], # Reduced for faster testing, expand later
    'model__epsilon': [0.01, 0.1, 0.2],
    'model__kernel': ['rbf'],
    'model__gamma': ['scale', 'auto']
}
param_grid_mlp = {
    'model__hidden_layer_sizes': [(10,), (15,5), (5,5), (20,10,5)], # Reduced for faster testing
    'model__activation': ['relu', 'tanh'],
    #'model__solver': ['adam'],
    'model__alpha': [ 0.1, 0.5, 1.0],
    'model__learning_rate_init': [0.001, 0.01],
    'model__batch_size': [4, 8]
}
param_grids = {
    "SVR": param_grid_svr,
    "MLPRegressor": param_grid_mlp,
}

# --- Nested CV Function ---
def run_nested_cv_for_model_type(X_dev, y_dev, material_labels_dev, 
                                 model_name, model_prototype, param_grid_model,
                                 n_outer_folds=5, n_repeats_outer=5,
                                 n_inner_folds=3, random_state=None,
                                 scoring_metric='neg_mean_squared_error', pca_components=None):
    
    outer_cv_strategy = RepeatedKFold(
        n_splits=n_outer_folds, n_repeats=n_repeats_outer, random_state=random_state
    )
    
    outer_fold_metrics = []
    best_hyperparams_from_inner_folds = []
    
    y_true_all_outer_folds_list = [] # Use lists to append
    y_pred_all_outer_folds_list = []
    material_labels_all_outer_folds_list = [] # To store material labels for outer test sets

    print(f"    Running Nested CV for {model_name}: Outer {n_repeats_outer}x{n_outer_folds}-Fold, Inner {n_inner_folds}-Fold GridSearch")
    
    for i, (train_idx_outer, test_idx_outer) in enumerate(outer_cv_strategy.split(X_dev, y_dev)): # Pass y_dev for splitting if y has groups for StratifiedKFold
        X_train_outer, y_train_outer = X_dev.iloc[train_idx_outer], y_dev.iloc[train_idx_outer]
        X_test_outer, y_test_outer = X_dev.iloc[test_idx_outer], y_dev.iloc[test_idx_outer]
        material_labels_test_outer = material_labels_dev.iloc[test_idx_outer] # Get material labels for this test fold

        # ... (Inner loop GridSearchCV logic as before) ...
        pipeline_for_grid = modeling.build_pipeline(clone(model_prototype), pca_components=pca_components)
        inner_cv_strategy = KFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
        grid_search_inner = GridSearchCV(pipeline_for_grid, param_grid_model, cv=inner_cv_strategy, scoring=scoring_metric, n_jobs=-1, verbose=0)
        try:
            with warnings.catch_warnings():
                if "MLP" in model_prototype.__class__.__name__: warnings.filterwarnings("ignore", category=ConvergenceWarning)
                grid_search_inner.fit(X_train_outer, y_train_outer)
        except Exception as e_grid_inner:
            # ... (error handling)
            outer_fold_metrics.append({'r2': np.nan, 'rmse': np.nan, 'mae': np.nan}); best_hyperparams_from_inner_folds.append({}); continue
        current_best_params_inner = grid_search_inner.best_params_
        best_hyperparams_from_inner_folds.append(current_best_params_inner)
        
        model_for_outer_eval_proto_outer = clone(model_prototype)
        model_for_outer_eval_proto_outer.set_params(**{k.replace('model__',''): v for k,v in current_best_params_inner.items() if k.startswith('model__')})
        pipeline_for_outer_test_run = modeling.build_pipeline(model_for_outer_eval_proto_outer, pca_components=pca_components)
        pipeline_for_outer_test_run.fit(X_train_outer, y_train_outer)
        y_pred_on_outer_test_run = pipeline_for_outer_test_run.predict(X_test_outer)
        
        y_true_all_outer_folds_list.extend(y_test_outer.tolist())
        y_pred_all_outer_folds_list.extend(y_pred_on_outer_test_run.tolist())
        material_labels_all_outer_folds_list.extend(material_labels_test_outer.tolist()) # Store material labels

        r2_fold_outer = r2_score(y_test_outer, y_pred_on_outer_test_run)
        # ... (calculate rmse, mae) ...
        rmse_fold_outer = np.sqrt(mean_squared_error(y_test_outer, y_pred_on_outer_test_run))
        mae_fold_outer = mean_absolute_error(y_test_outer, y_pred_on_outer_test_run)
        outer_fold_metrics.append({'r2': r2_fold_outer, 'rmse': rmse_fold_outer, 'mae': mae_fold_outer})

    return (outer_fold_metrics, best_hyperparams_from_inner_folds, 
            np.array(y_true_all_outer_folds_list), 
            np.array(y_pred_all_outer_folds_list), 
            np.array(material_labels_all_outer_folds_list))

# Keep your existing run_grid_search_all_models for the final tuning on the full dev set
def run_grid_search_for_final_tuning(X, y, model_prototype, param_grid_model, cv_strategy, scoring_metric='neg_mean_squared_error', pca_components=None):
    """Performs GridSearchCV for a single model type on the full development set."""
    print(f"  Running final GridSearchCV for {model_prototype.__class__.__name__}...")
    pipeline = modeling.build_pipeline(clone(model_prototype), pca_components=pca_components)
    
    grid_search = GridSearchCV(
        pipeline, param_grid_model, cv=cv_strategy, scoring=scoring_metric, n_jobs=-1, verbose=1
    )
    with warnings.catch_warnings():
        if "MLP" in model_prototype.__class__.__name__:
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.neural_network._multilayer_perceptron")
        grid_search.fit(X, y)
        
    print(f"    Best final score ({scoring_metric}): {grid_search.best_score_:.4f}")
    print(f"    Best final parameters: {grid_search.best_params_}")
    return grid_search.best_params_, grid_search.best_score_