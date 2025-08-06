# src_cnn/agg_classifier_results.py
"""
Aggregates and summarizes the results from all cross-validation folds for the classifier.
"""
import os
import pandas as pd

RESULTS_DIR = "results_classifier"

def main():
    all_dfs = []
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        return
        
    for filename in os.listdir(RESULTS_DIR):
        if filename.startswith("fold_") and filename.endswith("_results.csv"):
            filepath = os.path.join(RESULTS_DIR, filename)
            df = pd.read_csv(filepath)
            all_dfs.append(df)

    if not all_dfs:
        print("No fold result files found.")
        return

    results_df = pd.concat(all_dfs, ignore_index=True)
    
    print("--- Cross-Validation Results ---")
    print(results_df)

    # Calculate and print summary statistics
    mean_acc = results_df['accuracy'].mean()
    std_acc = results_df['accuracy'].std()
    mean_f1 = results_df['f1_score'].mean()
    std_f1 = results_df['f1_score'].std()

    print("\n--- Summary ---")
    print(f"Average Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Average F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")

if __name__ == "__main__":
    main()

# python -m src_cnn.agg_classifier_result