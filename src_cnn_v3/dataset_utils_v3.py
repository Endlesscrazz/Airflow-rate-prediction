# src_cnn_v3/dataset_utils_v3.py
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src_cnn_v3 import config_v3 as cfg

class HybridDataset(Dataset):
    """
    Serves (Video, Features, Target) tuples.
    """
    def __init__(self, metadata_df, dataset_dir, feature_scaler=None, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = transform
        
        self.feature_cols = ['delta_T']
        if cfg.USE_HANDCRAFTED_FEATURES:
            self.feature_cols += ['feat_area', 'feat_aspect', 'feat_extent']
            
        if feature_scaler:
            self.features_scaled = feature_scaler.transform(self.metadata[self.feature_cols].values)
        else:
            self.features_scaled = self.metadata[self.feature_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 1. Load Video Tensor
        npy_path = os.path.join(self.dataset_dir, row['file_path'])
        video_data = np.load(npy_path).astype(np.float32)
        
        # --- FIX: PAD SEQUENCE TO FIXED LENGTH ---
        current_len = video_data.shape[0]
        target_len = cfg.NUM_FRAMES_PER_SAMPLE
        
        if current_len < target_len:
            # Pad with zeros at the end
            pad_len = target_len - current_len
            # video_data shape is (Time, H, W)
            padding = np.zeros((pad_len, video_data.shape[1], video_data.shape[2]), dtype=np.float32)
            video_data = np.concatenate([video_data, padding], axis=0)
        elif current_len > target_len:
            # Trim if too long (shouldn't happen with current creation script, but safe)
            video_data = video_data[:target_len]
            
        # 2. Instance Normalization (On-the-Fly)
        if cfg.V3_PREPROCESS_PARAMS['ENABLE_INSTANCE_NORM']:
            v_min, v_max = video_data.min(), video_data.max()
            if v_max - v_min > 1e-6:
                video_data = (video_data - v_min) / (v_max - v_min)
            else:
                video_data = np.zeros_like(video_data)

        # 3. Channel Dimension -> (Time, 1, H, W) for Conv3d or similar
        # Note: Your model expects (B, T, C, H, W) during forward, so here we return (T, 1, H, W)
        # But wait, your current model processes 2D frames in a loop or uses (B*T, C, H, W).
        # Let's stick to the shape your model expects. 
        # In `models_v3.py`, you reshape: `c_in = video.view(B * T, C, H, W)`
        # This implies input `video` should be `(B, T, C, H, W)`.
        # So `__getitem__` should return `(T, C, H, W)`.
        
        video_tensor = torch.from_numpy(video_data).unsqueeze(1) # (T, 1, H, W)
        
        # 4. Tabular Features
        feat_vec = torch.tensor(self.features_scaled[idx], dtype=torch.float32)
        
        # 5. Target Scaling
        raw_airflow = row['airflow_rate']
        scaled_target = raw_airflow / cfg.MAX_FLOW_RATE
        target = torch.tensor(scaled_target, dtype=torch.float32)
        
        return video_tensor, feat_vec, target