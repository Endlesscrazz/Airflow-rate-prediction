# visualize_features_3d.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import argparse
import sys

try:
    import config
except ImportError:
    pass 

def load_features_from_csv(x_path="saved_features/model_input_features_X.csv",
                           y_path="saved_features/target_variable_y.csv",
                           all_features_path=None):
    """Loads features and target from CSV files."""
    df_X = None
    series_y = None
    df_all_processed = None

    if os.path.exists(x_path):
        try:
            df_X = pd.read_csv(x_path)
            print(f"Successfully loaded model input features (X) from: {x_path}")
        except Exception as e:
            print(f"Error loading X features from {x_path}: {e}")
            return None, None, None 
    else:
        print(f"Warning: Model input features file (X) not found at {x_path}")

    if os.path.exists(y_path):
        try:
            series_y = pd.read_csv(y_path).iloc[:, 0] # y is a single column
            print(f"Successfully loaded target variable (y) from: {y_path}")
        except Exception as e:
            print(f"Error loading y target from {y_path}: {e}")
            
    else:
        print(f"Warning: Target variable file (y) not found at {y_path}")


    if all_features_path and os.path.exists(all_features_path):
        try:
            df_all_processed = pd.read_csv(all_features_path)
            print(f"Successfully loaded all processed features from: {all_features_path}")
            # if df_all_processed contains 'airflow_rate' and the features for X,
            # it is used as the main DataFrame for plotting.
            if 'airflow_rate' not in df_all_processed.columns and series_y is not None:
                if len(df_all_processed) == len(series_y):
                    df_all_processed['airflow_rate'] = series_y.values
                else:
                    print("Warning: Length mismatch between all_processed_features and y. Cannot combine airflow_rate directly.")
            return df_X, series_y, df_all_processed # Return all three
        except Exception as e:
            print(f"Error loading all_processed_features from {all_features_path}: {e}")

    # If only X and y were loaded 
    if df_X is not None and series_y is not None:
        if len(df_X) == len(series_y):
            # Combine X and y into a single DataFrame for convenience if df_all_processed isn't used
            combined_df = df_X.copy()
            combined_df['airflow_rate'] = series_y.values
            return df_X, series_y, combined_df # df_all_processed is effectively this combined_df
        else:
            print("Warning: Length mismatch between loaded X and y. Cannot combine for plotting.")
            # Return X and y separately, df_all_processed will be None or the one from its path
            return df_X, series_y, None if not all_features_path else df_all_processed


    return df_X, series_y, df_all_processed


# plot_3d_scatter function 
def plot_3d_scatter(df, feature_cols_to_plot, color_col='airflow_rate', title="3D Feature Scatter Plot", save_path="3d_feature_plot.png"):
    if df is None or df.empty:
        print("DataFrame for plotting is empty or None. Cannot plot.")
        return
    if len(feature_cols_to_plot) != 3:
        print(f"Error: Exactly 3 feature columns are required for plotting. Got: {feature_cols_to_plot}")
        return
    
    missing_cols = [col for col in feature_cols_to_plot if col not in df.columns]
    if missing_cols:
        print(f"Error: Feature column(s) for plotting not found in DataFrame: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    if color_col not in df.columns:
        print(f"Error: Color column '{color_col}' not in DataFrame.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    df_plot = df.dropna(subset=feature_cols_to_plot + [color_col]).copy()
    if df_plot.empty:
        print("DataFrame is empty after dropping NaNs from selected plotting columns. Cannot plot.")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = df_plot[feature_cols_to_plot[0]]
    y = df_plot[feature_cols_to_plot[1]]
    z = df_plot[feature_cols_to_plot[2]]
    colors = df_plot[color_col]
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=60, alpha=0.75, edgecolors='k', linewidth=0.5)
    ax.set_xlabel(feature_cols_to_plot[0], fontsize=10)
    ax.set_ylabel(feature_cols_to_plot[1], fontsize=10)
    ax.set_zlabel(feature_cols_to_plot[2], fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='z', labelsize=8)
    cbar = fig.colorbar(scatter, shrink=0.7)
    cbar.set_label(color_col, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    try:
        plt.savefig(save_path, dpi=150)
        print(f"3D scatter plot saved to {save_path}")
        plt.show()
    except Exception as e:
        print(f"Error saving/showing plot: {e}")
    finally:
        plt.close(fig)


def main_features_vis_from_csv():
    parser = argparse.ArgumentParser(description="Visualize features in 3D from saved CSV files.")
    parser.add_argument("--x_file", default="saved_features/model_input_features_X.csv",
                        help="Path to the CSV file containing model input features (X).")
    parser.add_argument("--y_file", default="saved_features/target_variable_y.csv",
                        help="Path to the CSV file containing the target variable (y).")
    parser.add_argument("--all_features_file", default="saved_features/all_processed_features_with_target.csv", # Default to the comprehensive file
                        help="Optional: Path to CSV with all processed features (raw, transformed) and target. If provided, this is preferred for plotting.")
    parser.add_argument("--mode", choices=['manual', 'pca'], default='manual',
                        help="Feature selection mode: 'manual' or 'pca'.")
    parser.add_argument("--features_to_plot", nargs='+',
                        default=['delta_T_log','hotspot_area_log','hotspot_avg_temp_change_rate_initial_norm',], # Example names from your main.py's X
                        help="For 'manual' mode: list of 3 feature column names from the loaded CSV to plot.")
    parser.add_argument("--features_for_pca_input", nargs='+',
                        default=['delta_T_log', 'hotspot_area_log',
                                 'hotspot_avg_temp_change_rate_initial_norm',
                                 'temp_max_overall_initial', # Assuming these are column names in your saved CSV
                                 'overall_std_deltaT'],
                        help="For 'pca' mode: list of feature column names from the loaded CSV to input into PCA.")

    args = parser.parse_args()

    # Load data
    _, _, df_for_plotting = load_features_from_csv(args.x_file, args.y_file, args.all_features_file)
    # df_for_plotting will be the combined X and y, or the all_features_df if successfully loaded and contains target.

    if df_for_plotting is None or df_for_plotting.empty:
        print("No data loaded or DataFrame is empty. Exiting.")
        if not os.path.exists(args.all_features_file) and not os.path.exists(args.x_file):
             print(f"Hint: Ensure you have run main.py to generate '{args.all_features_file}' or '{args.x_file}'.")
        sys.exit(1)

    print("\n--- DataFrame Head for Plotting (loaded from CSV) ---")
    print(df_for_plotting.head())
    print(f"DataFrame shape: {df_for_plotting.shape}")
    print(f"Available columns for plotting: {df_for_plotting.columns.tolist()}")

    output_dir = "visualized_thermal_data/feature_visualizations_output_from_csv"
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == 'manual':
        plot_title = f"3D Scatter (Manual): {', '.join(args.features_to_plot)} vs Airflow"
        save_file = os.path.join(output_dir, f"3d_manual_{'_'.join(args.features_to_plot)}.png")
        plot_3d_scatter(df_for_plotting,
                        feature_cols_to_plot=args.features_to_plot,
                        color_col='airflow_rate', # Assumes 'airflow_rate' is in df_for_plotting
                        title=plot_title,
                        save_path=save_file)

    elif args.mode == 'pca':
        print(f"\n--- Performing PCA on features: {args.features_for_pca_input} ---")
        
        pca_input_cols_present = [col for col in args.features_for_pca_input if col in df_for_plotting.columns]
        missing_pca_inputs = set(args.features_for_pca_input) - set(pca_input_cols_present)
        if missing_pca_inputs:
            print(f"Warning: Requested PCA input features not found in loaded CSV: {missing_pca_inputs}")
        if len(pca_input_cols_present) < 3:
            print(f"Error: Need at least 3 available features for PCA. Found: {len(pca_input_cols_present)}")
            print(f"Available from your request and loaded CSV: {pca_input_cols_present}")
            sys.exit(1)
        
        print(f"Using these features from CSV for PCA: {pca_input_cols_present}")
        X_for_pca = df_for_plotting[pca_input_cols_present].copy()

        if X_for_pca.isnull().values.any():
            print("NaNs found in PCA input columns from CSV. Imputing with mean...")
            for col in X_for_pca.columns[X_for_pca.isnull().any()]: # Iterate only over columns that have NaNs
                X_for_pca[col] = X_for_pca[col].fillna(X_for_pca[col].mean())
        
        if X_for_pca.empty:
            print("PCA input DataFrame is empty after selection/NaN handling from CSV.")
            sys.exit(1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_for_pca)

        pca_random_state = getattr(config, 'RANDOM_STATE', 42) if 'config' in sys.modules else 42
        pca = PCA(n_components=3, random_state=pca_random_state)
        X_pca_transformed = pca.fit_transform(X_scaled)

        print(f"Explained variance ratio by 3 PCs: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")

        pca_plot_df = pd.DataFrame(data=X_pca_transformed, columns=['PC1', 'PC2', 'PC3'], index=X_for_pca.index)
        # Add airflow_rate back for coloring, ensuring correct alignment from the original loaded df
        pca_plot_df['airflow_rate'] = df_for_plotting.loc[X_for_pca.index, 'airflow_rate']

        plot_3d_scatter(pca_plot_df,
                        feature_cols_to_plot=['PC1', 'PC2', 'PC3'],
                        color_col='airflow_rate',
                        title="3D Scatter of Principal Components (from CSV) vs Airflow",
                        save_path=os.path.join(output_dir, "3d_pca_features_plot_from_csv.png"))

if __name__ == "__main__":
    main_features_vis_from_csv()