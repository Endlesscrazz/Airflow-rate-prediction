import pandas as pd
import os

N_SPLITS = 5
RESULTS_DIR = "results_lstm_1ch_CV"

all_dfs = []
for i in range(N_SPLITS):
    path = os.path.join(RESULTS_DIR, f"fold_{i}_results.csv")
    if os.path.exists(path):
        all_dfs.append(pd.read_csv(path))
    else:
        print(f"Warning: Result file not found for fold {i}")

if all_dfs:
    final_results = pd.concat(all_dfs, ignore_index=True)
    print("--- Cross-Validation Results ---")
    print(final_results)

    mean_r2 = final_results['r2'].mean()
    std_r2 = final_results['r2'].std()
    mean_rmse = final_results['rmse'].mean()
    std_rmse = final_results['rmse'].std()
    
    print("\n--- Summary ---")
    print(f"Average R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"Average RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

# python -m src_cnn.agg_results