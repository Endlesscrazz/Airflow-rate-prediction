# feature_engineering.py
"""Functions for extracting features from raw video frames."""

import numpy as np
import config

# Attempt to import TensorFlow and handle import errors gracefully
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.layers import GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model
    # Use tf.image.resize which handles batches
    # from tensorflow.keras.preprocessing.image import img_to_array, load_img # Less efficient for arrays
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not found. CNN feature extraction will not be available.")
    TF_AVAILABLE = False
except Exception as e:
    print(f"Warning: Error importing TensorFlow components: {e}")
    TF_AVAILABLE = False

def load_cnn_base(input_shape=(224, 224, 3)):
    """
    Loads the MobileNetV2 base model pre-trained on ImageNet.

    Args:
        input_shape (tuple): The expected input shape for the model (height, width, channels).

    Returns:
        tensorflow.keras.models.Model: The loaded base model, or None if TF is not available.
    """
    if not TF_AVAILABLE:
        return None

    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                             input_shape=input_shape)
    # We only need the base for feature extraction
    base_model.trainable = False # Freeze the weights
    print("Loaded MobileNetV2 base model with frozen weights.")
    return base_model

def extract_cnn_features(frames, cnn_base_model, target_size=(224, 224)):
    """
    Extracts features from video frames using a pre-trained CNN base model.

    Args:
        frames (np.ndarray): Video frames (num_frames, height, width). Assumed single channel (IR).
        cnn_base_model (tf.keras.models.Model): The loaded pre-trained CNN base.
        target_size (tuple): The target spatial size (height, width) for CNN input.

    Returns:
        np.ndarray: A single feature vector representing the video, or None if error.
    """
    if not TF_AVAILABLE or cnn_base_model is None:
        print("Error: TensorFlow or CNN base model not available for feature extraction.")
        return None
    if not isinstance(frames, np.ndarray) or frames.ndim != 3 or frames.shape[0] == 0:
        print("Warning: Invalid frames shape for CNN feature extraction.")
        return None

    num_frames = frames.shape[0]
    preprocessed_frames = []

    try:
        # Preprocess each frame
        for frame in frames:
            # 1. Convert single channel IR frame to 3 channels by stacking
            if frame.ndim == 2:
                frame_rgb = np.stack((frame,) * 3, axis=-1)
            elif frame.ndim == 3 and frame.shape[-1] == 1:
                frame_rgb = np.concatenate([frame] * 3, axis=-1)
            elif frame.ndim == 3 and frame.shape[-1] == 3:
                 frame_rgb = frame # Assume already RGB if 3 channels
            else:
                 print(f"Warning: Unexpected frame shape {frame.shape}. Skipping frame.")
                 continue

            # 2. Resize frame using TensorFlow (handles potential non-standard sizes)
            # Add batch dimension, resize, remove batch dimension
            frame_resized = tf.image.resize(frame_rgb[np.newaxis, ...], target_size)[0]

            preprocessed_frames.append(frame_resized)

        if not preprocessed_frames:
             print("Error: No frames could be preprocessed.")
             return None

        # Stack frames into a batch
        batch_frames = np.stack(preprocessed_frames, axis=0)

        # 3. Apply MobileNetV2-specific preprocessing
        batch_preprocessed = preprocess_input(batch_frames)

        # 4. Extract features using the CNN base
        # Running predict in batches can be more memory efficient if needed
        # For N=20 samples with likely short videos, processing all frames might be ok
        print(f"  Extracting features from {batch_preprocessed.shape[0]} frames...")
        features = cnn_base_model.predict(batch_preprocessed, verbose=0) # verbose=0 to reduce console spam

        # 5. Aggregate features spatially (per frame) using Global Average Pooling
        # Output shape after base is (num_frames, H', W', C') -> need (num_frames, C')
        # We can do this manually after predict if the model doesn't include it
        if features.ndim == 4: # Output includes spatial dims
             pooled_features_per_frame = np.mean(features, axis=(1, 2))
        elif features.ndim == 2: # Output might already be pooled if model structure implies it
             pooled_features_per_frame = features
        else:
             print(f"Warning: Unexpected CNN output feature dimension: {features.ndim}. Returning None.")
             return None

        # 6. Aggregate features temporally (across frames) - Simple Averaging
        # pooled_features_per_frame shape: (num_frames, num_cnn_features)
        if pooled_features_per_frame.shape[0] > 0:
            video_feature_vector = np.mean(pooled_features_per_frame, axis=0)
        else:
            print("Warning: No frame features to aggregate temporally. Returning None.")
            return None

        return video_feature_vector.astype(np.float32)

    except Exception as e:
        print(f"Error during CNN feature extraction: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging TF issues
        return None


def extract_handcrafted_features(frames):
    """
    Extracts simple statistical features from video frames.
    Assumes frames is a numpy array (num_frames, height, width).

    Args:
        frames (np.ndarray): Video frames.

    Returns:
        dict: A dictionary of extracted features. Returns NaNs if input is invalid.
    """
    feature_names = [
        "mean_temp", "std_temp", "max_temp", "min_temp",
        "mean_frame_std", "std_frame_std",
        "mean_temp_grad", "std_temp_grad",
        "mean_spatial_grad", "std_spatial_grad"
    ]

    if not isinstance(frames, np.ndarray) or frames.ndim != 3 or frames.shape[0] == 0 or frames.shape[1] == 0 or frames.shape[2] == 0:
        print(f"Warning: Invalid frames shape {getattr(frames, 'shape', 'N/A')} for feature extraction. Returning NaNs.")
        return {name: np.nan for name in feature_names}

    try:
        features = {}

        # Overall intensity/temperature stats (mean over pixels, then stats over frames)
        frame_means = np.mean(frames, axis=(1, 2))
        frame_stds = np.std(frames, axis=(1, 2))
        frame_maxs = np.max(frames, axis=(1, 2))
        frame_mins = np.min(frames, axis=(1, 2))

        features["mean_temp"] = np.mean(frame_means) if len(frame_means) > 0 else np.nan
        features["std_temp"] = np.std(frame_means) if len(frame_means) > 1 else 0.0
        features["max_temp"] = np.max(frame_maxs) if len(frame_maxs) > 0 else np.nan
        features["min_temp"] = np.min(frame_mins) if len(frame_mins) > 0 else np.nan
        features["mean_frame_std"] = np.mean(frame_stds) if len(frame_stds) > 0 else np.nan
        features["std_frame_std"] = np.std(frame_stds) if len(frame_stds) > 1 else 0.0

        # Temporal gradient (frame difference) stats
        if frames.shape[0] > 1:
            temporal_diff = np.abs(np.diff(frames, axis=0))
            temporal_grad_mean_per_frame = np.mean(temporal_diff, axis=(1, 2))
            features["mean_temp_grad"] = np.mean(temporal_grad_mean_per_frame) if len(temporal_grad_mean_per_frame) > 0 else np.nan
            features["std_temp_grad"] = np.std(temporal_grad_mean_per_frame) if len(temporal_grad_mean_per_frame) > 1 else 0.0
        else:
            features["mean_temp_grad"] = 0.0 # No change if only one frame
            features["std_temp_grad"] = 0.0

        # Spatial gradient stats
        spatial_grad_mags_mean_per_frame = []
        for frame in frames:
            if frame.shape[0] > 1 and frame.shape[1] > 1:
                 gy, gx = np.gradient(frame)
                 grad_mag = np.sqrt(gx**2 + gy**2)
                 spatial_grad_mags_mean_per_frame.append(np.mean(grad_mag))
            else:
                 spatial_grad_mags_mean_per_frame.append(0.0) # No gradient for 1D

        features["mean_spatial_grad"] = np.mean(spatial_grad_mags_mean_per_frame) if len(spatial_grad_mags_mean_per_frame) > 0 else np.nan
        features["std_spatial_grad"] = np.std(spatial_grad_mags_mean_per_frame) if len(spatial_grad_mags_mean_per_frame) > 1 else 0.0

        # Check for any NaN/Inf values produced (should be handled by imputer later, but good to know)
        for k, v in features.items():
            if not np.isfinite(v):
                 print(f"Warning: Non-finite value {v} generated for feature '{k}'.")

        return features

    except Exception as e:
         print(f"Error during feature extraction: {type(e).__name__} - {e}. Returning NaNs.")
         return {name: np.nan for name in feature_names}