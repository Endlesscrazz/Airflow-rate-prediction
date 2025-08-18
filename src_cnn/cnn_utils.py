# src_cnn/cnn_utils.py
"""
Contains the custom PyTorch Dataset class for the deep learning pipeline.
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import random
from src_cnn import config as cfg

class AirflowSequenceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading sequences of image patches.
    Handles different numbers of channels and optional temporal augmentations.
    """
    def __init__(self, metadata_df, cnn_dataset_dir, context_feature_cols,
                 dynamic_feature_cols, transform=None, task='regression', is_train=False):
        self.metadata = metadata_df.copy()
        self.cnn_dataset_dir = cnn_dataset_dir
        self.transform = transform
        self.context_feature_cols = context_feature_cols
        self.dynamic_feature_cols = dynamic_feature_cols
        self.task = task
        self.is_train = is_train

        if self.task == 'classification':
            bins = [0, 1.7, 2.3, float('inf')]
            labels = [0, 1, 2]
            self.metadata['airflow_category'] = pd.cut(self.metadata['airflow_rate'], bins=bins, labels=labels, right=False).fillna(0).astype(int)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample_row = self.metadata.iloc[idx]
        sequence_path = os.path.join(self.cnn_dataset_dir, sample_row['image_path'])
        sequence = np.load(sequence_path).astype(np.float32)

        # --- Temporal Augmentation (Jitter) ---
        if self.is_train:
            max_start_offset = 5
            current_seq_len = sequence.shape[0]
            if current_seq_len > max_start_offset + cfg.NUM_FRAMES_PER_SAMPLE - 1:
                start_offset = random.randint(0, max_start_offset)
                sequence = sequence[start_offset:]

        # Trim to a consistent length for batching
        target_len = cfg.NUM_FRAMES_PER_SAMPLE
        if 'flow' in self.cnn_dataset_dir or 'hybrid' in self.cnn_dataset_dir:
            target_len -= 1
        sequence = sequence[:target_len]

        # --- Channel & Transform Logic (Now fully agnostic) ---
        processed_frames = []
        # Case 1: Grayscale-like data saved as (Seq, H, W)
        if sequence.ndim == 3:
            for i in range(sequence.shape[0]):
                frame_data = np.expand_dims(sequence[i], axis=0) # (H, W) -> (1, H, W)
                frame_tensor = torch.from_numpy(frame_data.copy())
                if self.transform:
                    frame_tensor = self.transform(frame_tensor)
                processed_frames.append(frame_tensor)
        # Case 2: Multi-channel data saved as (Seq, H, W, C)
        elif sequence.ndim == 4:
            for i in range(sequence.shape[0]):
                frame_data = np.transpose(sequence[i], (2, 0, 1)) # (H, W, C) -> (C, H, W)
                frame_tensor = torch.from_numpy(frame_data.copy())
                if self.transform:
                    frame_tensor = self.transform(frame_tensor)
                processed_frames.append(frame_tensor)
        else:
            raise ValueError(f"Unsupported sequence ndim: {sequence.ndim}")
        
        image_sequence_tensor = torch.stack(processed_frames)

        # --- Tabular Data Loading ---
        context_features = sample_row[self.context_feature_cols].values.astype(np.float32)
        context_tensor = torch.from_numpy(context_features)

        dynamic_features = sample_row[self.dynamic_feature_cols].values.astype(np.float32)
        dynamic_tensor = torch.from_numpy(dynamic_features)

        if self.task == 'classification':
            target = torch.tensor(sample_row['airflow_category'], dtype=torch.long)
        else:
            # log transformation on target variable
            log_target = np.log1p(sample_row['airflow_rate'])
            target = torch.tensor(log_target, dtype=torch.float32)

        return image_sequence_tensor, context_tensor, dynamic_tensor, target