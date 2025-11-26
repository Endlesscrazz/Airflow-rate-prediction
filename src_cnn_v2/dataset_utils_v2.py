# src_cnn_v2/dataset_utils_v2.py
"""
Contains the custom PyTorch Dataset class for the V2 (cropped) pipeline.
"""
import os
import torch
import pandas as pd
import numpy as np

from src_cnn_v2 import config_v2 as cfg
from torch.utils.data import Dataset

class CroppedSequenceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading sequences of cropped image patches.
    Handles single-channel thermal data and tabular features (delta_T).
    """
    def __init__(self, metadata_df, cnn_dataset_dir, transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing sample metadata 
                                        (image_path, airflow_rate, delta_T).
            cnn_dataset_dir (str): The root directory where the .npy files are stored.
            transform (callable, optional): Optional transform to be applied on a sequence.
        """
        self.metadata = metadata_df
        self.cnn_dataset_dir = cnn_dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 1. Get sample metadata
        sample_row = self.metadata.iloc[idx]
        
        # 2. Load the cropped image sequence
        sequence_path = os.path.join(self.cnn_dataset_dir, sample_row['image_path'])
        data = np.load(sequence_path).astype(np.float32)

        # --- MATCH COLLEAGUE: MAX NORMALIZATION IN LOADER ---
        # Data comes in Frame-Normalized (centered around 1.0).
        # We now scale it to [0, 1] relative to the max value in this specific clip.
        max_val = np.max(data)
        if max_val > 1e-6:
            data = data / max_val
        
        # 4. Add a channel dimension: (Time, 1, Height, Width)
        data = np.expand_dims(data, axis=1)
        
        # 5. Convert to PyTorch tensor
        image_sequence_tensor = torch.from_numpy(data.copy())

        # 6. Apply transforms if any (e.g., for normalization if you were using mean/std)
        #    NOTE: Since we are doing instance normalization, this transform is likely not needed.
        if self.transform:
            processed_frames = [self.transform(frame) for frame in image_sequence_tensor]
            image_sequence_tensor = torch.stack(processed_frames)

        # 7. Load tabular data (delta_T)
        delta_t_tensor = torch.tensor(sample_row['delta_T'], dtype=torch.float32)

        # 8. Load the target variable (airflow_rate)
        target_tensor = torch.tensor(
            sample_row['airflow_rate'] / cfg.MAX_FLOW_RATE, 
            dtype=torch.float32
        )
        
        return image_sequence_tensor, delta_t_tensor, target_tensor