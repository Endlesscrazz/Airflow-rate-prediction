import os
import sys
import numpy as np
import glob
import scipy.io
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src_cnn_v3 import config_v3 as cfg

def find_original_video_path(video_id):
    """Helper to find the source .mat file given a video ID."""
    for key, conf in cfg.DATASET_CONFIGS.items():
        base_dir = os.path.join(cfg.RAW_DATASET_PARENT_DIR, conf['dataset_subfolder'])
        search_pattern = os.path.join(base_dir, "**", f"{video_id}.mat")
        results = glob.glob(search_pattern, recursive=True)
        if results:
            return results[0]
    return None

def main():
    print(f"--- Verifying Dataset Consistency ---")
    print(f"Target Directory: {cfg.DATASET_DIR}")
    print(f"Expected Shape: ({cfg.NUM_FRAMES_PER_SAMPLE}, {cfg.RESIZE_DIM[0]}, {cfg.RESIZE_DIM[1]})")

    npy_files = glob.glob(os.path.join(cfg.DATASET_DIR, "*.npy"))
    print(f"Found {len(npy_files)} tensors to check.\n")

    anomalies = []
    
    for fpath in tqdm(npy_files, desc="Checking Shapes"):
        try:
            # Load in mmap mode is faster just to check shape
            data = np.load(fpath, mmap_mode='r')
            
            # Check Time Dimension (Axis 0)
            if data.shape[0] != cfg.NUM_FRAMES_PER_SAMPLE:
                filename = os.path.basename(fpath)
                
                # Extract Video ID from filename (e.g., "video_id_1_orig.npy")
                # Adjust splitting based on your exact naming convention
                # Usually: {video_id}_{hole_id}_{type}.npy
                # But video_id might contain underscores.
                # Heuristic: Remove the last two parts (_holeID_type.npy)
                parts = filename.replace('.npy', '').split('_')
                
                # Reconstruct video_id (This is a guess, might need adjustment depending on exact naming)
                # Assuming format: video_name_hole_1_orig.npy
                if 'aug' in filename:
                    # video_name_hole_1_aug_0.npy
                    video_id = "_".join(parts[:-3]) 
                else:
                    # video_name_hole_1_orig.npy
                    video_id = "_".join(parts[:-2])
                
                anomalies.append({
                    "file": filename,
                    "shape": data.shape,
                    "video_id": video_id
                })
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)

    if not anomalies:
        print("✅ SUCCESS: All files have the correct shape.")
    else:
        print(f"❌ FOUND {len(anomalies)} FILES WITH INCORRECT LENGTH\n")
        
        # Group by Video ID to see root causes
        unique_videos = {}
        for a in anomalies:
            vid = a['video_id']
            if vid not in unique_videos:
                unique_videos[vid] = a['shape']
        
        print("--- Root Cause Analysis (Source Video Check) ---")
        for vid, shape in unique_videos.items():
            print(f"\nVideo ID: {vid}")
            print(f"  - Generated Tensor Shape: {shape}")
            
            # Check Source File
            mat_path = find_original_video_path(vid)
            if mat_path:
                try:
                    mat_data = scipy.io.loadmat(mat_path)
                    frames = mat_data.get('TempFrames')
                    if frames is not None:
                        print(f"  - Original .mat Shape:    {frames.shape}")
                        print(f"  - Conclusion: Source video only has {frames.shape[2]} frames.")
                        if frames.shape[2] < cfg.NUM_FRAMES_PER_SAMPLE:
                             print(f"    -> CONFIRMED: Video is shorter than required {cfg.NUM_FRAMES_PER_SAMPLE} frames.")
                    else:
                        print("  - Error: 'TempFrames' not found in .mat file")
                except Exception as e:
                    print(f"  - Error reading .mat file: {e}")
            else:
                print("  - Original .mat file not found in configured paths.")

if __name__ == "__main__":
    main()

# python src_cnn_v3/debug_scripts/debug_datasets.py