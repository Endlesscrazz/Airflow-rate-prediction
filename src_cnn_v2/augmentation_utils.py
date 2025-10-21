# src_cnn_v2/augmentation_utils.py
"""
Contains functions for applying augmentations to thermal video sequences.
"""
import numpy as np
import cv2
import random

def add_gaussian_noise(sequence, noise_level):
    """Adds Gaussian noise to a sequence of frames."""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def augment_geometric(sequence, rotation_degrees, translation_frac):
    """
    Applies a consistent random rotation and translation to all frames in a sequence.

    Args:
        sequence (np.array): Input sequence of shape (Time, Height, Width).
        rotation_degrees (float): Max rotation angle.
        translation_frac (float): Max translation fraction.

    Returns:
        np.array: Augmented sequence.
    """
    T, H, W = sequence.shape
    
    # 1. Determine a single, random transformation for the entire sequence
    angle = random.uniform(-rotation_degrees, rotation_degrees)
    trans_x = random.uniform(-translation_frac, translation_frac) * W
    trans_y = random.uniform(-translation_frac, translation_frac) * H

    # 2. Get the transformation matrix
    center = (W // 2, H // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Add translation to the rotation matrix
    rotation_matrix[0, 2] += trans_x
    rotation_matrix[1, 2] += trans_y

    # 3. Apply the *same* transformation to each frame in the sequence
    augmented_sequence = np.zeros_like(sequence)
    for i in range(T):
        frame = sequence[i]
        # Use bilinear interpolation and reflect padding to handle edges
        augmented_frame = cv2.warpAffine(
            frame, 
            rotation_matrix, 
            (W, H), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        augmented_sequence[i] = augmented_frame
        
    return augmented_sequence