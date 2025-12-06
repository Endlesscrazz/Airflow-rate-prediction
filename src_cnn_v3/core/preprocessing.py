# src_cnn_v3/core/preprocessing.py
import cv2
import numpy as np
from src_cnn_v3.core.image_processing import crop_rotate_resize

class ThermalPreprocessor:
    """
    The single source of truth for converting Raw Thermal Frames 
    into Model-Ready Tensors. Used by both Training and Inference.
    """
    def __init__(self, resize_dim=(32, 32), blur_kernel=(3,3), enable_temp_norm=True):
        self.resize_dim = resize_dim
        self.blur_kernel = blur_kernel
        self.enable_temp_norm = enable_temp_norm

    def process_frame(self, frame, obb_center, obb_size, obb_angle):
        """
        Input: Raw 2D Frame (float32) + OBB Info
        Output: Standardized 2D Patch (float32)
        """
        # 1. Temporal Normalization (Physics)
        # Removes ambient temp fluctuations. 
        if self.enable_temp_norm:
            mean_val = np.mean(frame)
            if mean_val > 1e-6:
                frame = frame / mean_val

        # 2. Spatial Smoothing (Sensor Noise Reduction)
        if self.blur_kernel:
            frame = cv2.blur(frame, self.blur_kernel)

        # 3. Geometric Standardization
        patch = crop_rotate_resize(
            frame, 
            obb_center, 
            obb_size, 
            obb_angle, 
            self.resize_dim
        )
        
        return patch

    def process_sequence(self, frames, obb_center, obb_size, obb_angle):
        """
        Process a stack of frames (Height, Width, Time) -> (Time, 32, 32)
        """
        # Input frames are usually (H, W, T) from .mat files
        # We need to iterate over T
        T = frames.shape[2]
        processed = []
        
        for i in range(T): 
            # Extract 2D frame
            frm = frames[:, :, i]
            p = self.process_frame(frm, obb_center, obb_size, obb_angle)
            processed.append(p)
            
        return np.array(processed) # Returns (Time, 32, 32)