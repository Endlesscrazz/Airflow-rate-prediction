# src_cnn_v2/check_split_distribution.py
"""
Utility script to visualize and verify the distribution of the target variable
(airflow_rate) across Training, Validation, and Test splits.

Use this to prove that your splits are statistically fair and representative.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v2 import config_v2 as cfg

def main():
    print(f"--- Analyzing Data Split Distribution ---")
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    print(f"Seed: {cfg.RANDOM_STATE}")

    # 1. Load the Split Files
    try:
        train_df = pd.read_csv(cfg.TRAIN_SPLIT_PATH)
        val_df = pd.read_csv(cfg.VAL_SPLIT_PATH)
        test_df = pd.read_csv(cfg.TEST_SPLIT_PATH)
    except FileNotFoundError as e:
        sys.exit(f"FATAL: Split file not found. Error: {e}\nRun split_data_v2.py first.")

    # 2. Print Statistical Summary
    print("\n" + "="*60)
    print(f"{'Split':<10} | {'Count':<8} | {'Mean':<8} | {'Std Dev':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 60)
    
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        stats = df['airflow_rate'].describe()
        print(f"{name:<10} | {int(stats['count']):<8} | {stats['mean']:.4f}   | {stats['std']:.4f}   | {stats['min']:.4f}   | {stats['max']:.4f}")
    print("="*60 + "\n")

    # 3. Generate Visualization Plots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Airflow Rate Distribution by Split (Seed {cfg.RANDOM_STATE})', fontsize=16)

    # Plot A: Kernel Density Estimate (KDE) - Shows the "Shape" of the data
    # This is the best way to show that the distributions overlap
    sns.kdeplot(data=train_df, x="airflow_rate", label='Train', fill=True, alpha=0.2, linewidth=2, ax=axes[0])
    sns.kdeplot(data=val_df, x="airflow_rate", label='Validation', fill=True, alpha=0.2, linewidth=2, ax=axes[0])
    sns.kdeplot(data=test_df, x="airflow_rate", label='Test', fill=True, alpha=0.2, linewidth=2, ax=axes[0])
    axes[0].set_title("Density Plot (Shape Comparison)")
    axes[0].set_xlabel("Airflow Rate (L/min)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Plot B: Box Plot - Shows the Spread and Outliers
    # Combine into one DF for easy plotting
    train_df['Split'] = 'Train'
    val_df['Split'] = 'Validation'
    test_df['Split'] = 'Test'
    combined_df = pd.concat([train_df, val_df, test_df])
    
    sns.boxplot(data=combined_df, x='Split', y='airflow_rate', ax=axes[1], palette="Set2")
    axes[1].set_title("Box Plot (Range & Quartile Comparison)")
    axes[1].set_ylabel("Airflow Rate (L/min)")

    # Save the plot
    save_path = os.path.join(cfg.OUTPUT_DIR, f"split_distribution_seed{cfg.RANDOM_STATE}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Distribution plot saved to: {save_path}")
    print("Check this image. If the curves overlap well and boxplots are aligned, the split is fair.")

if __name__ == "__main__":
    main()

# python -m src_cnn_v2.debug_files.check_split_distribution