# tuning.py
"""Hyperparameter tuning using GridSearchCV.
(Corrected Augmentation Logic)
- Implements data augmentation within the nested CV loop.
- Correctly augments the 'groups' array to match the augmented data size.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold, GroupShuffleSplit
from sklearn.base import clone
import config
import modeling
import time
import traceback
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning
import warnings
import pandas as pd

# --- Parameter Grids ---
param_grid_svr = {
    'model__C': [1, 10, 50], 
    'model__epsilon': [0.01, 0.1, 0.2, 0.5],
    'model__kernel': ['rbf'],
    'model__gamma': ['scale', 'auto']
}
param_grid_mlp = {
    'model__hidden_layer_sizes': [(10,), (15,5), (30,15,5), (20,10,5)], 
    'model__activation': ['relu', 'tanh'],
    'model__solver': ['adam'],
    'model__alpha': [ 0.1, 0.5, 1.0],
    'model__learning_rate_init': [0.001, 0.01],
    'model__batch_size': [4, 8]
}
param_grid_xgb = {
    'model__n_estimators': [50, 100, 150],        
    'model__max_depth': [2, 3, 4],               
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],    
    'model__subsample': [0.7, 0.8, 0.9, 1.0],              
    'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],       
    'model__gamma': [0, 0.1, 0.5],               
    'model__reg_alpha': [0, 0.1, 0.5],           
    'model__reg_lambda': [1, 1.5, 2]             
}
param_grids = {
    "SVR": param_grid_svr,
    "MLPRegressor": param_grid_mlp,
    "XGBRegressor": param_grid_xgb
}

# --- CORRECTED Data Augmentation Helper Function ---
def augment_features(X_df, y_series, groups_series, num_augment_copies=1, noise_level=0.01):
    """
    Augments features, targets, AND group labels together.
    """
    augmented_X_rows = []
    augmented_y_values = []
    augmented_groups = [] # <-- NEW list for group labels
    
    numerical_cols = X_df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols = [col for col in numerical_cols if not col.startswith('material_')]

    for index, row in X_df.iterrows():
        for _ in range(num_augment_copies):
            augmented_row = row.copy()
            for col_name in numerical_cols:
                original_value = augmented_row[col_name]
                if np.isfinite(original_value):
                    if original_value != 0:
                        noise = np.random.normal(0, noise_level) * original_value
                        augmented_row[col_name] = original_value + noise
                    else:
                        pass # Keep zeros as zeros
                        
            augmented_X_rows.append(augmented_row)
            augmented_y_values.append(y_series.loc[index])
            # --- NEW: Append the group label for the new augmented sample ---
            augmented_groups.append(groups_series.loc[index])

    if not augmented_X_rows: # If no rows were augmented
        return X_df, y_series, groups_series

    augmented_X = pd.DataFrame(augmented_X_rows, columns=X_df.columns)
    augmented_y = pd.Series(augmented_y_values, name=y_series.name)
    augmented_groups_series = pd.Series(augmented_groups, name=groups_series.name) # <-- NEW Series for groups

    # Concatenate original with augmented
    X_augmented_final = pd.concat([X_df, augmented_X], ignore_index=True)
    y_augmented_final = pd.concat([y_series, augmented_y], ignore_index=True)
    groups_augmented_final = pd.concat([groups_series, augmented_groups_series], ignore_index=True) # <-- NEW final groups
    
    return X_augmented_final, y_augmented_final, groups_augmented_final

# --- Nested CV Function ---
def run_nested_cv_for_model_type(X_dev, y_dev, material_labels_dev, 
                                 video_id_groups, 
                                 model_name, model_prototype, param_grid_model,
                                 n_outer_folds=5, n_repeats_outer=1, 
                                 n_inner_folds=3, random_state=None,
                                 scoring_metric='neg_mean_squared_error', pca_components=None):
    
    outer_cv_strategy = GroupKFold(n_splits=n_outer_folds)
    
    # (Initialize lists as before)
    outer_fold_metrics, best_hyperparams_from_inner_folds = [], []
    y_true_all_outer_folds_list, y_pred_all_outer_folds_list = [], []
    material_labels_all_outer_folds_list, video_ids_all_outer_folds_list = [], []

    print(f"    Running Group-Aware Nested CV for {model_name}...")
    
    for i, (train_idx_outer, test_idx_outer) in enumerate(outer_cv_strategy.split(X_dev, y_dev, groups=video_id_groups)):
        X_train_outer_original, y_train_outer_original = X_dev.iloc[train_idx_outer], y_dev.iloc[train_idx_outer]
        X_test_outer, y_test_outer = X_dev.iloc[test_idx_outer], y_dev.iloc[test_idx_outer]
        
        material_labels_test_outer = material_labels_dev.iloc[test_idx_outer]
        groups_train_outer_original = video_id_groups.iloc[train_idx_outer] # <-- Get original groups for this fold
        video_ids_test_outer = video_id_groups.iloc[test_idx_outer]

        # --- CORRECTED AUGMENTATION CALL ---
        X_train_augmented, y_train_augmented, groups_train_augmented = augment_features(
            X_train_outer_original, y_train_outer_original, groups_train_outer_original,
            num_augment_copies=2, noise_level=0.01
        )
        print(f"      Fold {i+1}: Augmented training set from {len(X_train_outer_original)} to {len(X_train_augmented)} samples.")
        
        pipeline_for_grid = modeling.build_pipeline(clone(model_prototype), pca_components=pca_components)
        inner_cv_strategy = GroupShuffleSplit(n_splits=n_inner_folds, test_size=0.25, random_state=random_state)
        grid_search_inner = GridSearchCV(pipeline_for_grid, param_grid_model, cv=inner_cv_strategy, scoring=scoring_metric, n_jobs=-1, verbose=0)
        
        try:
            with warnings.catch_warnings():
                if "MLP" in model_prototype.__class__.__name__: warnings.filterwarnings("ignore", category=ConvergenceWarning)
                # Fit the inner GridSearchCV on the AUGMENTED data and AUGMENTED groups
                grid_search_inner.fit(X_train_augmented, y_train_augmented, groups=groups_train_augmented)
        except Exception as e_grid_inner:
            print(f"      Inner GridSearchCV failed for fold {i+1}: {e_grid_inner}")
            outer_fold_metrics.append({'r2': np.nan, 'rmse': np.nan, 'mae': np.nan}); best_hyperparams_from_inner_folds.append({}); continue

        current_best_params_inner = grid_search_inner.best_params_
        best_hyperparams_from_inner_folds.append(current_best_params_inner)
        
        model_for_outer_eval_proto_outer = clone(model_prototype)
        model_for_outer_eval_proto_outer.set_params(**{k.replace('model__',''): v for k,v in current_best_params_inner.items() if k.startswith('model__')})
        pipeline_for_outer_test_run = modeling.build_pipeline(model_for_outer_eval_proto_outer, pca_components=pca_components)
        
        # Fit the outer pipeline on the AUGMENTED training data
        pipeline_for_outer_test_run.fit(X_train_augmented, y_train_augmented)
        # But evaluate on the ORIGINAL, UNSEEN test data
        y_pred_on_outer_test_run = pipeline_for_outer_test_run.predict(X_test_outer)
        
        y_true_all_outer_folds_list.extend(y_test_outer.tolist())
        y_pred_all_outer_folds_list.extend(y_pred_on_outer_test_run.tolist())
        material_labels_all_outer_folds_list.extend(material_labels_test_outer.tolist())
        video_ids_all_outer_folds_list.extend(video_ids_test_outer.tolist())

        r2_fold_outer = r2_score(y_test_outer, y_pred_on_outer_test_run)
        rmse_fold_outer = np.sqrt(mean_squared_error(y_test_outer, y_pred_on_outer_test_run))
        mae_fold_outer = mean_absolute_error(y_test_outer, y_pred_on_outer_test_run)
        outer_fold_metrics.append({'r2': r2_fold_outer, 'rmse': rmse_fold_outer, 'mae': mae_fold_outer})

    return (outer_fold_metrics, best_hyperparams_from_inner_folds, 
            np.array(y_true_all_outer_folds_list), 
            np.array(y_pred_all_outer_folds_list), 
            np.array(material_labels_all_outer_folds_list),
            np.array(video_ids_all_outer_folds_list))

def run_grid_search_for_final_tuning(X, y, model_prototype, param_grid_model, cv_strategy, scoring_metric='neg_mean_squared_error', pca_components=None):
    """Performs GridSearchCV for a single model type on the full development set."""
    # Note: Augmentation is not applied here, as this is for the final model fit on the original dev data.
    # If you wanted to augment here too, you'd need to pass the groups and augment before fitting.
    print(f"  Running final GridSearchCV for {model_prototype.__class__.__name__}...")
    pipeline = modeling.build_pipeline(clone(model_prototype), pca_components=pca_components)
    grid_search = GridSearchCV(pipeline, param_grid_model, cv=cv_strategy, scoring=scoring_metric, n_jobs=-1, verbose=1)
    with warnings.catch_warnings():
        if "MLP" in model_prototype.__class__.__name__:
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.neural_network._multilayer_perceptron")
        grid_search.fit(X, y)
    print(f"    Best final score ({scoring_metric}): {grid_search.best_score_:.4f}")
    print(f"    Best final parameters: {grid_search.best_params_}")
    return grid_search.best_params_, grid_search.best_score_