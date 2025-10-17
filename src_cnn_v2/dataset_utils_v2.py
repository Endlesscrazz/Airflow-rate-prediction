# src_cnn_v2/dataset_utils_v2.py
"""
Contains the custom PyTorch Dataset class for the V2 (cropped) pipeline.
"""
import os
import torch
import pandas as pd
import numpy as np
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
        # Sequence is saved as (Time, Height, Width)
        sequence_numpy = np.load(sequence_path).astype(np.float32)

        # 3. Add a channel dimension: (Time, 1, Height, Width)
        sequence_numpy = np.expand_dims(sequence_numpy, axis=1)
        
        # 4. Convert to PyTorch tensor
        image_sequence_tensor = torch.from_numpy(sequence_numpy.copy())

        # 5. Apply transforms if any (e.g., normalization)
        if self.transform:
            # We need to apply the transform to each frame in the sequence
            processed_frames = [self.transform(frame) for frame in image_sequence_tensor]
            image_sequence_tensor = torch.stack(processed_frames)

        # 6. Load tabular data (delta_T)
        delta_t_tensor = torch.tensor(sample_row['delta_T'], dtype=torch.float32)

        # 7. Load the target variable (airflow_rate)
        target_tensor = torch.tensor(sample_row['airflow_rate'], dtype=torch.float32)
        
        return image_sequence_tensor, delta_t_tensor, target_tensor