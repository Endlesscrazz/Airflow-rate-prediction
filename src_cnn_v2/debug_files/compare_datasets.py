# src_cnn_v2/compare_datasets.py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm

def find_colleague_match(user_filepath, colleague_files_list):
    """
    Matches User format: 'T1.4V_...__1_orig.npy' 
    To Colleague format: '2_T1.4V_...__spot1_original.npy'
    """
    try:
        user_filename = os.path.basename(user_filepath)
        
        if "__" not in user_filename:
            return None
            
        parts = user_filename.split("__")
        video_id = parts[0]  # "T1.4V_2025..."
        
        suffix = parts[1]
        hole_num = suffix.split("_")[0] 
        
        target_spot_str = f"spot{hole_num}"
        
        for c_path in colleague_files_list:
            c_name = os.path.basename(c_path)
            
            if (video_id in c_name) and \
               (target_spot_str in c_name) and \
               ("original" in c_name) and \
               ("noise" not in c_name):
                return c_path
                
    except Exception as e:
        print(f"Error matching {user_filename}: {e}")
        return None
    return None

def analyze_pair(user_path, colleague_path, output_dir):
    user_filename = os.path.basename(user_path)
    colleague_filename = os.path.basename(colleague_path)
    
    # 1. Load Data
    u_data = np.load(user_path).astype(np.float32)
    c_data = np.load(colleague_path).astype(np.float32)
    
    # 2. Standardize Shapes to (Time, H, W)
    if u_data.shape[0] != 150 and u_data.shape[-1] == 150:
        u_data = u_data.transpose(2, 0, 1)
    
    if c_data.shape[0] != 150 and c_data.shape[-1] == 150:
        c_data = c_data.transpose(2, 0, 1)
        
    # 3. Calc Stats
    diff = np.abs(u_data - c_data)
    mean_diff = diff.mean()
    max_diff = diff.max()
    
    u_stats = {'min': u_data.min(), 'max': u_data.max(), 'mean': u_data.mean()}
    c_stats = {'min': c_data.min(), 'max': c_data.max(), 'mean': c_data.mean()}
    
    # 4. Determine Status and Folder
    is_match = mean_diff < 0.05
    status_str = "MATCH" if is_match else "MISMATCH"
    subfolder = "matches" if is_match else "mismatches"
    
    save_dir = os.path.join(output_dir, subfolder)
    os.makedirs(save_dir, exist_ok=True)

    # 5. Plotting
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    
    title_text = (
        f"[{status_str}] Mean Diff: {mean_diff:.4f} | Max Diff: {max_diff:.4f}\n"
        f"User: {user_filename}\n"
        f"Coll: {colleague_filename}"
    )
    fig.suptitle(title_text, fontsize=14, y=0.98, color='green' if is_match else 'red')
    
    # Spatial Heatmap
    u_img = np.mean(u_data, axis=0)
    c_img = np.mean(c_data, axis=0)
    
    vmin = min(u_img.min(), c_img.min())
    vmax = max(u_img.max(), c_img.max())
    
    im1 = ax[0,0].imshow(u_img, cmap='inferno', vmin=vmin, vmax=vmax)
    ax[0,0].set_title(f"User Spatial Mean\nRange: [{u_stats['min']:.2f}, {u_stats['max']:.2f}]")
    plt.colorbar(im1, ax=ax[0,0])
    
    im2 = ax[0,1].imshow(c_img, cmap='inferno', vmin=vmin, vmax=vmax)
    ax[0,1].set_title(f"Colleague Spatial Mean\nRange: [{c_stats['min']:.2f}, {c_stats['max']:.2f}]")
    plt.colorbar(im2, ax=ax[0,1])
    
    # Temporal Profile
    u_time = np.mean(u_data, axis=(1, 2))
    c_time = np.mean(c_data, axis=(1, 2))
    
    ax[1,0].plot(u_time, label='User', color='blue', alpha=0.8)
    ax[1,0].plot(c_time, label='Colleague', color='orange', linestyle='--', alpha=0.8)
    ax[1,0].set_title("Temporal Profile (Avg Intensity)")
    ax[1,0].legend()
    ax[1,0].grid(True, alpha=0.3)
    
    # Histogram
    ax[1,1].hist(u_data.flatten(), bins=50, alpha=0.5, label='User', color='blue', density=True)
    ax[1,1].hist(c_data.flatten(), bins=50, alpha=0.5, label='Colleague', color='orange', density=True)
    ax[1,1].set_title("Pixel Value Distribution")
    ax[1,1].legend()
    
    save_name = f"compare_{user_filename.replace('.npy', '')}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()

    # Return stats for logging
    return is_match, {
        'user_filename': user_filename,
        'colleague_filename': colleague_filename,
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'u_stats': u_stats,
        'c_stats': c_stats
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_dir", required=True, help="Path to your .npy files")
    parser.add_argument("--colleague_dir", required=True, help="Path to colleague's Final_data folder")
    parser.add_argument("--output_dir", default="comparison_results", help="Where to save plots")
    parser.add_argument("--limit", type=int, default=5, help="Number of files to compare (ignored if --all is set)")
    parser.add_argument("--all", action='store_true', help="Compare ALL matching files found")
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "mismatch_report.txt")
    
    print("--- Loading File Lists ---")
    user_files = glob.glob(os.path.join(args.user_dir, "*_orig.npy"))
    colleague_files = [str(p) for p in Path(args.colleague_dir).rglob("*.npy")]
    
    print(f"Found {len(user_files)} User files.")
    print(f"Found {len(colleague_files)} Colleague files (recursive).")
    
    matches_found = 0
    mismatches = []
    
    to_process = user_files if args.all else user_files[:args.limit]
    
    print(f"\n--- Starting Comparison (Processing {len(to_process)} files) ---")
    print(f"Plots will be saved to: {args.output_dir}/matches and /mismatches")
    
    # Open log file
    with open(log_path, 'w') as log_file:
        log_file.write("MISMATCH REPORT\n")
        log_file.write("==================================================\n")
        
        for u_file in tqdm(to_process):
            c_file = find_colleague_match(u_file, colleague_files)
            
            if c_file:
                is_match, stats = analyze_pair(u_file, c_file, args.output_dir)
                matches_found += 1
                
                if not is_match:
                    mismatches.append(stats)
                    # Log details immediately
                    log_file.write(f"\n[MISMATCH] {stats['user_filename']}\n")
                    log_file.write(f"  Matched with:   {stats['colleague_filename']}\n")
                    log_file.write(f"  Mean Diff:      {stats['mean_diff']:.4f}\n")
                    log_file.write(f"  Max Diff:       {stats['max_diff']:.4f}\n")
                    log_file.write(f"  User Range:     {stats['u_stats']['min']:.2f} to {stats['u_stats']['max']:.2f} (Mean: {stats['u_stats']['mean']:.2f})\n")
                    log_file.write(f"  Colleague Range:{stats['c_stats']['min']:.2f} to {stats['c_stats']['max']:.2f} (Mean: {stats['c_stats']['mean']:.2f})\n")
                    log_file.write("-" * 50 + "\n")
            else:
                pass

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"Total Pairs Compared: {matches_found}")
    print(f"Total Mismatches (>0.05 mean diff): {len(mismatches)}")
    print(f"Detailed report saved to: {log_path}")
    
    if len(mismatches) == 0 and matches_found > 0:
        print("\n🎉 ALL FILES MATCHED PERFECTLY!")

if __name__ == "__main__":
    main()
"""
python src_cnn_v2/debug_files/compare_datasets.py \
  --user_dir "CNN_dataset/hardyboard_all_dataset_v2/dataset_cs15_nf150_aug100" \
  --colleague_dir "/scratch/general/vast/u1527145/Maksym-code/Flow_rate_model_hardyboard/Final_data" \
  --output_dir "comparison_output" \
  --all

# BRICKCLADDING
python src_cnn_v2/debug_files/compare_datasets.py \
  --user_dir "CNN_dataset/brickcladding_all_dataset_v2/dataset_cs15_nf150_aug100" \
  --colleague_dir "/scratch/general/vast/u1527145/Maksym-code/Flow_rate_model_brickcladding/Final_data" \
  --output_dir "comparison_output/brickcladding" \
  --all


"""