# evaluation_utils.py
"""Utility functions for model evaluation and reporting."""
import pandas as pd
import numpy as np

def aggregate_nested_cv_scores(model_name, outer_fold_scores_list):
    """Calculates mean and std for metrics from nested CV outer folds."""
    if not outer_fold_scores_list:
        print(f"Warning: No scores provided for {model_name} to aggregate.")
        return None, None, None # Or raise error

    df_scores = pd.DataFrame(outer_fold_scores_list)
    results = {}
    for metric in ['r2', 'rmse', 'mae']:
        if metric in df_scores.columns:
            results[f'mean_{metric}'] = df_scores[metric].mean()
            results[f'std_{metric}'] = df_scores[metric].std()
        else:
            results[f'mean_{metric}'] = np.nan
            results[f'std_{metric}'] = np.nan
            
    print(f"  Aggregated Nested CV Results for {model_name}:")
    print(f"    Mean R²: {results.get('mean_r2', np.nan):.4f} +/- {results.get('std_r2', np.nan):.4f}")
    print(f"    Mean RMSE: {results.get('mean_rmse', np.nan):.4f} +/- {results.get('std_rmse', np.nan):.4f}")
    print(f"    Mean MAE: {results.get('mean_mae', np.nan):.4f} +/- {results.get('std_mae', np.nan):.4f}")
    return results

def analyze_hyperparameter_frequency(model_name, list_of_best_params_dicts):
    """Analyzes frequency of chosen hyperparameters across outer folds."""
    if not list_of_best_params_dicts:
        print(f"No hyperparameter sets to analyze for {model_name}.")
        return {}
    
    print(f"\n  Hyperparameter Selection Frequency for {model_name} (from {len(list_of_best_params_dicts)} outer folds):")
    param_counts = {}
    for params_dict in list_of_best_params_dicts:
        for p_name, p_val in params_dict.items():
            p_name_clean = p_name.replace('model__', '') # Clean prefix
            if p_name_clean not in param_counts:
                param_counts[p_name_clean] = {}
            
            # Convert mutable types like tuples (for hidden_layer_sizes) to strings for counting
            p_val_key = str(p_val) if isinstance(p_val, (list, tuple, dict)) else p_val
            param_counts[p_name_clean][p_val_key] = param_counts[p_name_clean].get(p_val_key, 0) + 1
            
    for p_name_clean, val_counts in param_counts.items():
        print(f"    Parameter '{p_name_clean}':")
        for val_key, count in sorted(val_counts.items(), key=lambda item: item[1], reverse=True):
            print(f"      - {val_key}: {count} times")
    return param_counts
