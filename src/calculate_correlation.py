# calculate_correlation.py
"""
Calculates and visualizes the correlation matrix for the extracted features
and the target variable (airflow_rate).

Assumes you have run main.py at least once to generate the initial DataFrame
with all features before NaN handling specific to model input (X).
You might need to slightly modify main.py to save this intermediate DataFrame
or re-run the feature extraction part here.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Option 1: Load features from a saved CSV (Recommended)
# Modify main.py to save the 'df' DataFrame before extensive manipulation for X
FEATURES_CSV_PATH = "output/intermediate_features.csv" # Example path

# Option 2: Re-run feature extraction here (Slower, duplicates main.py logic)
# If using this option, you need to copy/adapt the feature extraction loop from main.py
# import config
# import data_utils
# import feature_engineering
# PRECOMPUTED_MASK_DIR = "output_hotspot_mask" # Or from config
# FPS_FOR_FEATURES = getattr(config, 'FPS', 5.0)
# FOCUS_DURATION_SEC_FOR_FEATURES = getattr(config, 'FOCUS_DURATION_SEC', 10.0)
# # ... (Add ALL other necessary parameters for feature_engineering call) ...

# Output path for the correlation plot
CORRELATION_PLOT_PATH = "output/feature_correlation_matrix.png"


def main():
    """Loads data, calculates correlations, and saves plot."""

    print("--- Feature Correlation Analysis ---")

    # --- Load Data ---
    df = None
    if os.path.exists(FEATURES_CSV_PATH):
        print(f"Loading features from: {FEATURES_CSV_PATH}")
        try:
            df = pd.read_csv(FEATURES_CSV_PATH)
        except Exception as e:
            print(f"Error loading CSV: {e}. Attempting recalculation.")
            # df = recalculate_features() # Uncomment to enable recalculation fallback
    else:
        print(f"Features CSV not found at {FEATURES_CSV_PATH}.")
        # df = recalculate_features() # Uncomment to enable recalculation fallback

    if df is None:
        print("Error: Could not load or recalculate features. Exiting.")
        return

    print(f"Loaded DataFrame shape: {df.shape}")
    print("Columns:", list(df.columns))

    # --- Select Columns for Correlation ---
    # Include the target variable and all potentially relevant features
    # (before normalization/transformation, as correlation is usually checked on raw relationships)
    # Or use the final transformed features if you prefer (adjust column names)
    cols_for_corr = [
        'airflow_rate', # Target
        'delta_T',      # Raw delta_T
        # --- Add ALL features present in your DataFrame ---
        'hotspot_area',
        'hotspot_temp_change_rate',
        'hotspot_temp_change_magnitude',
        'activity_mean',
        'activity_median',
        'activity_std',
        'activity_max',
        'activity_sum',
        'temp_mean_avg',
        'temp_std_avg',
        'temp_min_overall',
        'temp_max_overall',
        'avg_pixel_slope',
        'max_abs_pixel_slope',
        'std_pixel_slope',
        'avg_pixel_mag',
        'max_pixel_mag',
        # Add normalized/log versions if you want to see their correlations too
        # 'hotspot_temp_change_rate_norm',
        # 'avg_pixel_slope_norm',
        # 'delta_T_log',
    ]

    # Filter DataFrame to only include existing columns from the list
    existing_cols = [col for col in cols_for_corr if col in df.columns]
    df_corr = df[existing_cols].copy()

    # Handle potential NaNs before calculating correlation (e.g., drop rows with any NaN)
    rows_before = len(df_corr)
    df_corr.dropna(inplace=True)
    rows_after = len(df_corr)
    if rows_before != rows_after:
        print(f"Warning: Dropped {rows_before - rows_after} rows with NaN values before calculating correlation.")

    if len(df_corr) < 2:
        print("Error: Not enough valid data points remaining to calculate correlation.")
        return

    # --- Calculate Correlation Matrix ---
    correlation_matrix = df_corr.corr()

    print("\nCorrelation Matrix (Full):")
    print(correlation_matrix.round(3))

    print(f"\nCorrelations with Target Variable ('{df_corr.columns[0]}'):")
    print(correlation_matrix[df_corr.columns[0]].sort_values(ascending=False).round(3))

    # --- Visualize Correlation Matrix ---
    print(f"\nGenerating heatmap plot...")
    plt.figure(figsize=(16, 12)) # Adjust size as needed
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(CORRELATION_PLOT_PATH), exist_ok=True)
    try:
        plt.savefig(CORRELATION_PLOT_PATH, dpi=150)
        print(f"Correlation heatmap saved to: {CORRELATION_PLOT_PATH}")
    except Exception as e:
        print(f"Error saving correlation plot: {e}")
    # plt.show() # Uncomment to display plot interactively


if __name__ == "__main__":
    main()