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
    def __init__(self, metadata_df, cnn_dataset_dir,context_feature_cols,
                 dynamic_feature_cols, transform=None):
        self.metadata = metadata_df
        self.cnn_dataset_dir = cnn_dataset_dir
        self.transform = transform
        self.context_feature_cols = context_feature_cols
        self.dynamic_feature_cols = dynamic_feature_cols

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample_row = self.metadata.iloc[idx]
        
        # 1. Load Image Sequence and normalize (this logic is correct)
        sequence_path = os.path.join(self.cnn_dataset_dir, sample_row['image_path'])
        sequence = np.load(sequence_path).astype(np.float32)

         # For optical flow data, the loaded shape is (Seq, H, W, Channels)
        # We need to convert it to PyTorch's desired (Seq, Channels, H, W)
        # The np.transpose function reorders the axes.
        # 0->0 (Seq), 3->1 (Channels), 1->2 (H), 2->3 (W)
        sequence = np.transpose(sequence, (0, 3, 1, 2))

        # min_val, max_val = sequence.min(), sequence.max()
        # if max_val > min_val:
        #     sequence = (sequence - min_val) / (max_val - min_val)
        
        # processed_frames = []
        # for frame in sequence:
        #     # The frame is now normalized correctly in the [0, 1] range.
        #     frame = np.expand_dims(frame, axis=0) # Add channel dimension
        #     frame_tensor = torch.from_numpy(frame)
        
        #     # The advanced augmentations (flips, rotations, jitter, etc.)
        #     # and the final normalization (to [-1, 1]) will be applied here
        #     # by the transform pipeline passed from the training script.
        #     if self.transform:
        #         frame_tensor = self.transform(frame_tensor)
            
        #     processed_frames.append(frame_tensor)

        # image_sequence_tensor = torch.stack(processed_frames)

        processed_frames = []
        for frame in sequence:
            # frame is now a (2, 128, 128) numpy array
            frame_tensor = torch.from_numpy(frame)
            
            if self.transform:
                frame_tensor = self.transform(frame_tensor)
            
            processed_frames.append(frame_tensor)

        image_sequence_tensor = torch.stack(processed_frames)


        # 2. Prepare CONTEXTUAL tabular data
        context_features = sample_row[self.context_feature_cols].values.astype(np.float32)
        context_tensor = torch.from_numpy(context_features)

        # 3. Prepare DYNAMIC handcrafted feature data
        dynamic_features = sample_row[self.dynamic_feature_cols].values.astype(np.float32)
        dynamic_tensor = torch.from_numpy(dynamic_features)

        # 4. Get the target
        target = torch.tensor(sample_row['airflow_rate'], dtype=torch.float32)

        return image_sequence_tensor, context_tensor, dynamic_tensor, target