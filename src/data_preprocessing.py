# data_preprocessing.py
"""Functions for preprocessing raw IR video frames."""

import numpy as np
import cv2

def apply_median_filter(frames, kernel_size=3):
    """Applies a median filter to each frame in the video sequence."""
    if frames is None or frames.ndim != 3:
        print("Warning: Invalid frames array for median filter.")
        return frames # Return original if invalid
        
    num_frames = frames.shape[2]
    processed_frames = np.zeros_like(frames, dtype=frames.dtype)
    
    print(f"Applying Median Filter (kernel={kernel_size})...")
    for i in range(num_frames):
        try:
            # Ensure frame is contiguous C-style array if needed by OpenCV
            frame_to_filter = np.ascontiguousarray(frames[:, :, i], dtype=np.float32) # Use float32 for filtering
            filtered_frame = cv2.medianBlur(frame_to_filter, kernel_size)
            processed_frames[:, :, i] = filtered_frame.astype(frames.dtype) # Convert back to original dtype
        except Exception as e:
            print(f"Error applying median filter to frame {i}: {e}. Using original frame.")
            processed_frames[:, :, i] = frames[:, :, i]
            
    return processed_frames

def scale_video_intensity(frames, scale_range=(0, 1)):
    """Scales the intensity of all frames in a video to a specified range [min_val, max_val]."""
    if frames is None or frames.size == 0:
        print("Warning: Invalid frames array for scaling.")
        return frames

    print(f"Scaling video intensity to range {scale_range}...")
    min_val, max_val = scale_range
    
    # Find global min/max across the whole video, ignoring NaNs
    global_min = np.nanmin(frames)
    global_max = np.nanmax(frames)
    
    if np.isnan(global_min) or np.isnan(global_max):
        print("Warning: Could not find valid global min/max (all NaNs?). Skipping scaling.")
        return frames
        
    value_range = global_max - global_min
    
    if value_range < 1e-6: # Avoid division by zero if video is constant
        print(f"Warning: Video has near-zero intensity range ({value_range}). Setting scaled output to {min_val}.")
        # Return an array of the minimum scale value, preserving shape
        scaled_frames = np.full(frames.shape, min_val, dtype=np.float32) 
    else:
        # Apply scaling: (frame - global_min) / range
        scaled_frames = (frames - global_min) / value_range
        # Scale to the target range [min_val, max_val]
        scaled_frames = scaled_frames * (max_val - min_val) + min_val
        # Clip to ensure values are strictly within the range (handles potential float inaccuracies)
        scaled_frames = np.clip(scaled_frames, min_val, max_val)

    # Recommend returning float32 for subsequent processing
    return scaled_frames.astype(np.float32) 

def spatial_blur_video_frames(frames, ksize=3):
    """
    Applies a Gaussian blur to each frame in the video sequence.
    Args:
        frames (np.ndarray): Video frames (H, W, NumFrames).
        ksize (int): Kernel size for GaussianBlur (must be odd). If < 3, no blur is applied.
    Returns:
        np.ndarray: Spatially blurred frames.
    """
    if frames is None or frames.ndim != 3:
        print("Warning: Invalid frames array for spatial blur.")
        return frames
    if not isinstance(ksize, int) or ksize < 3 or ksize % 2 == 0:
        # print(f"Info: Spatial blur ksize ({ksize}) invalid or too small. Skipping spatial blur.")
        return frames.copy() # Return a copy if no blur

    blurred_frames = np.empty_like(frames, dtype=frames.dtype)
    # print(f"Applying Spatial Gaussian Blur (kernel size: {ksize}x{ksize})...")
    for i in range(frames.shape[2]):
        try:
            # GaussianBlur handles different data types, but float32 is common
            blurred_frames[:, :, i] = cv2.GaussianBlur(frames[:, :, i].astype(np.float32), 
                                                      (ksize, ksize), sigmaX=0).astype(frames.dtype)
        except Exception as e:
            print(f"Error applying Gaussian blur to frame {i}: {e}. Using original frame.")
            blurred_frames[:, :, i] = frames[:, :, i]
    return blurred_frames

# --- Main Preprocessing Function ---
def preprocess_video_frames(frames, apply_filter=True, filter_kernel_size=3, 
                            apply_scaling=True, scale_range=(0, 1),
                            apply_spatial_blur=False, spatial_blur_ksize=3): # Added spatial blur args
    """Applies selected preprocessing steps to the raw video frames."""
    
    processed_frames = frames.copy() 

    # Apply spatial blur first if requested
    if apply_spatial_blur: # This flag would be for general preprocessing
        processed_frames = spatial_blur_video_frames(processed_frames, ksize=spatial_blur_ksize)

    if apply_filter: # Median filter
        processed_frames = apply_median_filter(processed_frames, kernel_size=filter_kernel_size)

    if apply_scaling:
        processed_frames = scale_video_intensity(processed_frames, scale_range=scale_range)
        
    return processed_frames