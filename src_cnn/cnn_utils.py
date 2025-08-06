# src_cnn/cnn_utils.py
"""
Contains utility classes for the CNN+LSTM training pipeline.
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class AirflowSequenceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading sequences of thermal ROI patches.
    """
    # --- START OF FIX ---
    def __init__(self, metadata_df, cnn_dataset_dir, context_feature_cols,
                 dynamic_feature_cols, transform=None, task='regression'): # <-- Add 'task' to the signature
    # --- END OF FIX ---
        self.metadata = metadata_df.copy() # Use .copy() to avoid SettingWithCopyWarning
        self.cnn_dataset_dir = cnn_dataset_dir
        self.transform = transform
        self.context_feature_cols = context_feature_cols
        self.dynamic_feature_cols = dynamic_feature_cols
        self.task = task

        if self.task == 'classification':
            bins = [0, 1.7, 2.3, float('inf')]
            labels = [0, 1, 2]
            self.metadata['airflow_category'] = pd.cut(self.metadata['airflow_rate'], bins=bins, labels=labels, right=False)
            self.metadata['airflow_category'] = self.metadata['airflow_category'].fillna(0).astype(int)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample_row = self.metadata.iloc[idx]
        
        sequence_path = os.path.join(self.cnn_dataset_dir, sample_row['image_path'])
        sequence = np.load(sequence_path).astype(np.float32)

        if sequence.ndim == 3: # This is for single-channel data (Seq, H, W)
            # Just add the channel dimension. Do NOT normalize here.
            sequence = np.expand_dims(sequence, axis=1) # -> (Seq, 1, H, W)
        
        # The PyTorch standard is (C, H, W). Our data is (Seq, ...).
        # So we need to process frame by frame anyway.
        # Let's handle the transpose inside the loop for clarity.
        # The input `sequence` is now either (Seq, 1, H, W) or (Seq, H, W, C)

        processed_frames = []
        for i in range(sequence.shape[0]):
            if sequence.ndim == 4 and sequence.shape[1] == 1: # (Seq, 1, H, W)
                frame_data = sequence[i] # Shape (1, H, W)
            elif sequence.ndim == 4: # (Seq, H, W, C)
                frame_data = np.transpose(sequence[i], (2, 0, 1)) # (H, W, C) -> (C, H, W)
            else:
                raise ValueError("Unsupported sequence shape")

            frame_tensor = torch.from_numpy(frame_data.copy())
            if self.transform:
                frame_tensor = self.transform(frame_tensor)
            processed_frames.append(frame_tensor)
        
        image_sequence_tensor = torch.stack(processed_frames)

        context_features = sample_row[self.context_feature_cols].values.astype(np.float32)
        context_tensor = torch.from_numpy(context_features)

        dynamic_features = sample_row[self.dynamic_feature_cols].values.astype(np.float32)
        dynamic_tensor = torch.from_numpy(dynamic_features)

        if self.task == 'classification':
            target = torch.tensor(sample_row['airflow_category'], dtype=torch.long)
        else: # Default to regression
            target = torch.tensor(sample_row['airflow_rate'], dtype=torch.float32)

        return image_sequence_tensor, context_tensor, dynamic_tensor, target