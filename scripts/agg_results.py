# scripts/agg_results.py
"""
Aggregates and summarizes the results from a cross-validation run.
"""
import pandas as pd
import os
import argparse
import sys

def main():
    # 1. Setup & Parse Arguments
    parser = argparse.ArgumentParser(description="Aggregate cross-validation results from a specific experiment.")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'avg'],
                        help="The model type used in the experiment ('lstm' or 'avg').")
    parser.add_argument("--in_channels", type=int, required=True,
                        help="The number of input channels used (1, 2, or 3).")
    parser.add_argument("--total_folds", type=int, default=5,
                        help="The total number of folds that were run.")
    parser.add_argument("--optuna_tuned", action='store_true',
                        help="Specify if aggregating results from an Optuna-tuned run.")
    args = parser.parse_args()

    # 2. Dynamically construct the results directory path
    model_name_tag = f"{args.model_type}_{args.in_channels}ch"
    if args.optuna_tuned:
        model_name_tag += "_optuna"
    RESULTS_DIR = f"results_{model_name_tag}_CV"
    
    print(f"--- Aggregating results from: '{RESULTS_DIR}' ---")

    if not os.path.isdir(RESULTS_DIR):
        print(f"\nFATAL ERROR: Results directory not found at '{RESULTS_DIR}'")
        sys.exit(1)

    # 3. Load results from each fold
    all_dfs = []
    for i in range(args.total_folds):
        path = os.path.join(RESULTS_DIR, f"fold_{i}_results.csv")
        if os.path.exists(path):
            all_dfs.append(pd.read_csv(path))
        else:
            print(f"Warning: Result file not found for fold {i} at '{path}'")

    if not all_dfs:
        print("\nERROR: No result files were found. Cannot generate summary.")
        return

    # 4. Aggregate and print summary
    final_results = pd.concat(all_dfs, ignore_index=True)
    
    print("\n--- Individual Fold Results ---")
    print(final_results.to_string())

    mean_r2 = final_results['r2'].mean()
    std_r2 = final_results['r2'].std()
    mean_rmse = final_results['rmse'].mean()
    std_rmse = final_results['rmse'].std()
    
    print("\n--- Final Summary ---")
    print(f"Average R²   : {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"Average RMSE : {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"(Based on {len(final_results)} completed folds out of {args.total_folds})")

if __name__ == "__main__":
    main()

# python -m scripts.agg_results --model_type lstm --in_channels 1