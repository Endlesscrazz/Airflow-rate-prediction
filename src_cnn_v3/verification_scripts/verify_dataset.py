# src_cnn_v3/verification_scripts/verify_dataset.py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src_cnn_v3 import config_v3 as cfg

# Style setup
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

def load_npy(filename):
    path = os.path.join(cfg.DATASET_DIR, filename)
    return np.load(path)

def plot_spatial_gallery(df_train, n_samples, save_dir):
    """
    Creates a side-by-side comparison of Original vs Augmented samples.
    Demonstrates the 'Squash & Stretch' and Noise injection.
    """
    print(f"  - Generating Spatial Gallery ({n_samples} samples)...")
    
    # Get N unique original samples
    originals = df_train[df_train['aug_type'] == 'original'].sample(n_samples)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2.5, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle(f'Input Tensor Visualization (32x32)\nTop: Original | Bottom: Augmented (Noise + Jitter)', fontsize=14)

    # Handle case where n_samples=1 (axes is 1D array)
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, (_, row) in enumerate(originals.iterrows()):
        # 1. Load Original
        orig_data = load_npy(row['file_path'])
        # Dynamic middle frame selection
        mid_idx_orig = orig_data.shape[0] // 2
        orig_img = orig_data[mid_idx_orig, :, :]
        
        # 2. Find a corresponding Augmentation
        sample_prefix = row['file_path'].replace('_orig.npy', '')
        aug_rows = df_train[
            (df_train['file_path'].str.contains(sample_prefix)) & 
            (df_train['aug_type'] == 'augmented')
        ]
        
        if not aug_rows.empty:
            aug_row = aug_rows.sample(1).iloc[0]
            aug_data = load_npy(aug_row['file_path'])
            mid_idx_aug = aug_data.shape[0] // 2
            aug_img = aug_data[mid_idx_aug, :, :]
        else:
            # Fallback if no augmentation found (shouldn't happen in training set)
            aug_img = np.zeros_like(orig_img)

        # Plot Original
        ax_top = axes[0, i] if n_samples > 1 else axes[0][0]
        ax_top.imshow(orig_img, cmap='inferno')
        ax_top.set_title(f"ID {row['hole_id']}\nRate: {row['airflow_rate']:.1f}", fontsize=9)
        ax_top.axis('off')
        
        # Plot Augmented
        ax_bot = axes[1, i] if n_samples > 1 else axes[1][0]
        ax_bot.imshow(aug_img, cmap='inferno')
        ax_bot.set_title(f"Augmented", fontsize=9)
        ax_bot.axis('off')

    save_path = os.path.join(save_dir, "verify_1_spatial_gallery.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    -> Saved: {os.path.basename(save_path)}")

def plot_temporal_consistency(df_train, n_samples, save_dir):
    """
    Plots the mean intensity over time for N samples.
    Demonstrates that Temporal Normalization worked (lines should be stable/flat).
    """
    print(f"  - Generating Temporal Consistency Plot...")
    
    samples = df_train.sample(n_samples)
    
    plt.figure(figsize=(10, 5))
    
    for _, row in samples.iterrows():
        data = load_npy(row['file_path'])
        # Calculate mean of each frame over time axis
        # data shape: (Time, H, W) -> mean over (1, 2)
        temporal_profile = np.mean(data, axis=(1, 2))
        
        label = f"ID {row['hole_id']} ({row['aug_type'][:3]})"
        plt.plot(temporal_profile, label=label, alpha=0.8, linewidth=1.5)
        
    plt.axhline(1.0, color='k', linestyle='--', alpha=0.5, label="Ideal Norm Baseline (1.0)")
    plt.title("Temporal Stability Check\n(Should be flat lines near 1.0)", fontsize=12)
    plt.xlabel("Frame Index (Time)")
    plt.ylabel("Mean Intensity (Normalized)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "verify_2_temporal_stability.png")
    plt.savefig(save_path)
    plt.close()
    print(f"    -> Saved: {os.path.basename(save_path)}")

def plot_value_distribution(df_train, n_samples, save_dir):
    """
    Plots histogram of pixel values.
    Ensures data isn't clipped or vanishing.
    """
    print(f"  - Generating Data Distribution Plot...")
    
    samples = df_train.sample(n_samples * 2) 
    all_pixels = []
    
    for _, row in samples.iterrows():
        data = load_npy(row['file_path'])
        # Subsample pixels to save memory/time
        all_pixels.extend(data.flatten()[::10]) 
        
    plt.figure(figsize=(10, 5))
    sns.histplot(all_pixels, bins=50, kde=True, color='purple')
    plt.axvline(1.0, color='r', linestyle='--', label='Mean (1.0)')
    
    plt.title(f"Pixel Value Distribution (Sample of {len(samples)} tensors)", fontsize=12)
    plt.xlabel("Normalized Intensity Value")
    plt.ylabel("Count")
    plt.legend()
    
    save_path = os.path.join(save_dir, "verify_3_value_dist.png")
    plt.savefig(save_path)
    plt.close()
    print(f"    -> Saved: {os.path.basename(save_path)}")

def main():
    parser = argparse.ArgumentParser(description="Visualize V3 Dataset Tensors")
    parser.add_argument("--n_samples", type=int, default=6, help="Number of samples to visualize in gallery")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"--- V3 Dataset Verification (N={args.n_samples}) ---")
    
    # 1. Load Metadata
    train_meta_path = os.path.join(cfg.DATASET_DIR, "train_metadata.csv")
    if not os.path.exists(train_meta_path):
        sys.exit(f"FATAL: Metadata not found at {train_meta_path}")
        
    df_train = pd.read_csv(train_meta_path)
    print(f"Loaded Metadata: {len(df_train)} records found.")
    
    save_dir = cfg.EXPERIMENT_RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving visualizations to: {save_dir}")

    # 2. Generate Visuals
    try:
        plot_spatial_gallery(df_train, args.n_samples, save_dir)
        plot_temporal_consistency(df_train, args.n_samples, save_dir)
        plot_value_distribution(df_train, args.n_samples, save_dir)
    except Exception as e:
        print(f"\nERROR during visualization: {e}")
        import traceback
        traceback.print_exc()

    print("\nVerification Complete")

if __name__ == "__main__":
    main()

# python src_cnn_v3/verification_scripts/verify_dataset.py --n_samples 10