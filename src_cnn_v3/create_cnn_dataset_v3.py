# src_cnn_v3/create_cnn_dataset_v3.py
import os
import sys
import pandas as pd
import numpy as np
import scipy.io
import random
from tqdm import tqdm
from joblib import Parallel, delayed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src_cnn_v3 import config_v3 as cfg
from src_cnn_v3.core.preprocessing import ThermalPreprocessor
from src_cnn_v3.logging_utils_v3 import log_section

# Initialize Processor ONCE
processor = ThermalPreprocessor(
    resize_dim=cfg.RESIZE_DIM,
    blur_kernel=cfg.V3_PREPROCESS_PARAMS["BLUR_KERNEL_SIZE"],
    enable_temp_norm=cfg.V3_PREPROCESS_PARAMS["ENABLE_TEMPORAL_NORM"]
)

def process_sample(row, output_dir, is_training, dataset_configs):
    """
    Smart worker: Checks if files exist before processing.
    """
    video_id = row['video_id']
    hole_id = row['hole_id']
    sample_id = f"{video_id}_{hole_id}"
    
    metadata_entries = []
    
    # --- 1. IDENTIFY REQUIRED FILES ---
    required_files = []
    
    # Original is always required
    required_files.append({
        'path': os.path.join(output_dir, f"{sample_id}_orig.npy"),
        'type': 'original',
        'idx': -1
    })
    
    # Augmentations are required only for training
    if is_training:
        for i in range(cfg.NUM_AUGMENTATIONS):
            required_files.append({
                'path': os.path.join(output_dir, f"{sample_id}_aug_{i}.npy"),
                'type': 'augmented',
                'idx': i
            })

    # --- 2. CHECK FOR MISSING FILES ---
    missing_files = [f for f in required_files if not os.path.exists(f['path'])]
    
    # --- 3. FAST PATH: EVERYTHING EXISTS ---
    if not missing_files:
        # Just generate metadata entries and return
        for item in required_files:
            meta = row.to_dict()
            meta['file_path'] = os.path.basename(item['path'])
            meta['aug_type'] = item['type']
            metadata_entries.append(meta)
        return metadata_entries

    # --- 4. SLOW PATH: GENERATE MISSING ---
    
    # Locate Video
    source_key = row['source_dataset_key']
    d_conf = dataset_configs[source_key]
    dataset_path = os.path.join(cfg.RAW_DATASET_PARENT_DIR, d_conf['dataset_subfolder'])
    
    mat_path = None
    possible_path = os.path.join(dataset_path, f"{video_id}.mat")
    if os.path.exists(possible_path):
        mat_path = possible_path
    else:
        for root, _, files in os.walk(dataset_path):
            if f"{video_id}.mat" in files:
                mat_path = os.path.join(root, f"{video_id}.mat")
                break
    if not mat_path: return None

    try:
        frames = scipy.io.loadmat(mat_path)['TempFrames'].astype(np.float32)
    except:
        return None

    limit_T = min(frames.shape[2], cfg.NUM_FRAMES_PER_SAMPLE)
    frames = frames[:, :, :limit_T]

    # Process items in the list of requirements
    # Note: We re-generate the stack even if only 1 aug is missing to ensure consistency,
    # but we only SAVE what is missing to save disk I/O.
    
    # A. Generate Base Stack
    # We always need this to generate augmentations, even if orig exists
    base_stack = processor.process_sequence(
        frames, 
        (row['obb_center_x'], row['obb_center_y']),
        (row['obb_width'], row['obb_height']),
        row['obb_angle']
    )

    for item in required_files:
        save_path = item['path']
        
        # If file exists, just add meta
        if os.path.exists(save_path):
            meta = row.to_dict()
            meta['file_path'] = os.path.basename(save_path)
            meta['aug_type'] = item['type']
            metadata_entries.append(meta)
            continue
            
        # If missing, generate and save
        data_to_save = None
        
        if item['type'] == 'original':
            data_to_save = base_stack
        else:
            # Generate specific augmentation
            # Use deterministic seeding based on sample_id + index for reproducibility
            jit_center = (row['obb_center_x'] + random.uniform(-2, 2), 
                          row['obb_center_y'] + random.uniform(-2, 2))
            jit_angle = row['obb_angle'] + random.uniform(-5, 5)
            
            aug_stack = processor.process_sequence(
                frames, jit_center, (row['obb_width'], row['obb_height']), jit_angle
            )
            noise = np.random.normal(0, 0.02, aug_stack.shape).astype(np.float32)
            data_to_save = aug_stack + noise
            
        np.save(save_path, data_to_save)
        
        meta = row.to_dict()
        meta['file_path'] = os.path.basename(save_path)
        meta['aug_type'] = item['type']
        metadata_entries.append(meta)
            
    return metadata_entries

def main():
    print("--- V3: Generating Dataset Tensors (Smart Skip Enabled) ---")
    
    os.makedirs(cfg.DATASET_DIR, exist_ok=True)
    
    df_train = pd.read_csv(cfg.TRAIN_SPLIT_PATH)
    df_val = pd.read_csv(cfg.VAL_SPLIT_PATH)
    df_test = pd.read_csv(cfg.TEST_SPLIT_PATH)
    
    def process_partition(df, is_train, name):
        print(f"\nProcessing {name} set ({len(df)} samples)...")
        # Parallel processing
        results = Parallel(n_jobs=16)(
            delayed(process_sample)(row, cfg.DATASET_DIR, is_train, cfg.DATASET_CONFIGS) 
            for _, row in tqdm(df.iterrows(), total=len(df))
        )
        final_meta = []
        for res in results:
            if res: final_meta.extend(res)
            
        out_path = os.path.join(cfg.DATASET_DIR, f"{name}_metadata.csv")
        pd.DataFrame(final_meta).to_csv(out_path, index=False)
        print(f"  - Saved {len(final_meta)} entries to {name} metadata.")

    process_partition(df_train, True, "train")
    process_partition(df_val, False, "val")
    process_partition(df_test, False, "test")
    
    # Log the completion
    log_data = {
        "Status": "Completed",
        "Note": "Files that already existed were skipped (Lazy Loading).",
        "Output Dir": cfg.DATASET_DIR
    }
    log_section("Dataset Generation", log_data)

if __name__ == "__main__":
    main()