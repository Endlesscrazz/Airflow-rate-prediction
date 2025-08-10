# src_feature_based/modeling.py
"""
Builds machine learning models and pipelines for the feature-based approach.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from src_feature_based import config as cfg

def asymmetric_mse_objective(y_true, y_pred):
    """
    Custom XGBoost objective function for asymmetric MSE.
    This must be a top-level function to be pickleable.
    """
    errors = y_pred - y_true
    grad = np.where(errors > 0, cfg.ASYMMETRIC_LOSS_OVER_WEIGHT * errors, errors)
    hess = np.full_like(y_true, 1.0)
    hess = np.where(errors > 0, cfg.ASYMMETRIC_LOSS_OVER_WEIGHT * hess, hess)
    return grad, hess

def get_models_and_grids():
    """
    Returns dictionaries of model instances and their corresponding hyperparameter grids.
    The models and grids are defined in the central config file.
    """
    models = {
        'XGBoost': XGBRegressor(random_state=cfg.RANDOM_STATE, n_jobs=-1),
        'RandomForest': RandomForestRegressor(random_state=cfg.RANDOM_STATE, n_jobs=-1),
        'SVR': SVR(),
        'MLPRegressor': MLPRegressor(
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=cfg.RANDOM_STATE
        )
    }
    
    # If asymmetric loss is enabled
    if cfg.ENABLE_ASYMMETRIC_LOSS:
        models['XGBoost'].set_params(objective=asymmetric_mse_objective)

    param_grids = cfg.PARAM_GRIDS
    
    active_models = {name: model for name, model in models.items() if name in param_grids}
    
    print(f"Active models for tuning: {list(active_models.keys())}")
    return active_models, param_grids

def build_pipeline(model):
    """Builds a standard pipeline: Imputer -> Scaler -> Model."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])