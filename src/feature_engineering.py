# feature_engineering.py
"""Functions for extracting features from raw video frames."""

import numpy as np
import warnings
from scipy.stats import skew, kurtosis # Import skew and kurtosis

# --- TensorFlow Import Block (remains the same) ---
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
except Exception as e:
    print(f"Warning: Error importing TensorFlow components: {e}")
    TF_AVAILABLE = False
# --- End TensorFlow Import Block ---

# --- load_cnn_base function (remains the same) ---
def load_cnn_base(input_shape=(224, 224, 3)):
    # ... (keep existing code) ...
    if not TF_AVAILABLE:
        print("Error: Cannot load CNN base model, TensorFlow not available.")
        return None
    try:
        base_model = MobileNetV2(weights='imagenet', include_top=False,
                                 input_shape=input_shape, pooling='avg')
        base_model.trainable = False
        print("Loaded MobileNetV2 base model with built-in Global Average Pooling and frozen weights.")
        return base_model
    except Exception as e:
        print(f"Error loading MobileNetV2 model: {type(e).__name__} - {e}")
        return None
# --- End load_cnn_base ---

# --- extract_cnn_features function (remains the same) ---
def extract_cnn_features(frames, cnn_base_model, target_size=(224, 224)):
    # ... (keep existing code) ...
    if not TF_AVAILABLE or cnn_base_model is None:
        print("Error: TensorFlow or CNN base model not available/loaded for feature extraction.")
        return None
    if not isinstance(frames, np.ndarray) or frames.ndim != 3 or frames.shape[0] == 0:
        print(f"Warning: Invalid frames shape {getattr(frames, 'shape', 'N/A')} for CNN features. Returning None.")
        return None

    num_frames, h, w = frames.shape
    try:
        # Frame Preprocessing
        normalized_frames = np.zeros_like(frames, dtype=np.float32)
        for i, frame in enumerate(frames):
            min_val, max_val = frame.min(), frame.max()
            if max_val > min_val:
                normalized_frames[i] = (frame - min_val) / (max_val - min_val) * 255.0
            else:
                normalized_frames[i] = np.zeros(frame.shape, dtype=np.float32)
        frames_rgb = np.stack([normalized_frames] * 3, axis=-1)
        frames_resized = tf.image.resize(tf.constant(frames_rgb, dtype=tf.float32), target_size)
        frames_mobilenet_preprocessed = preprocess_input(frames_resized)

        # Feature Extraction
        print(f"  Extracting CNN features from {num_frames} frames...")
        @tf.function
        def predict_features(data):
             return cnn_base_model(data, training=False)
        features_per_frame = predict_features(frames_mobilenet_preprocessed)
        features_per_frame_np = features_per_frame.numpy()

        # Temporal Aggregation
        if features_per_frame_np.shape[0] > 0:
            video_feature_vector = np.mean(features_per_frame_np, axis=0)
        else:
            print("Warning: No frame features to aggregate temporally. Returning None.")
            return None
        return video_feature_vector.astype(np.float32)

    except Exception as e:
        print(f"Error during CNN feature extraction: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return None
# --- End extract_cnn_features ---


def extract_handcrafted_features(frames):
    """
    Extracts simple statistical features from video frames, including skewness and kurtosis.
    Assumes frames is a numpy array (num_frames, height, width).

    Args:
        frames (np.ndarray): Video frames.

    Returns:
        dict: A dictionary of extracted features. Returns dict with NaNs if input is invalid.
    """
    # Define ALL feature names FIRST
    feature_names = [
        # Basic Stats
        "mean_temp", "std_temp", "max_temp", "min_temp", "median_temp",
        "q25_temp", "q75_temp",
        # Skewness & Kurtosis of Temp Means
        "skew_temp", "kurt_temp",
        # Stats of Intra-Frame Variation
        "mean_frame_std", "std_frame_std",
        "skew_frame_std", "kurt_frame_std", # Skew/Kurt of frame stds
        # Temporal Gradient Stats
        "mean_temp_grad", "std_temp_grad", "max_abs_temp_grad",
        "skew_temp_grad", "kurt_temp_grad", # Skew/Kurt of temporal gradients
        # Spatial Gradient Stats
        "mean_spatial_grad", "std_spatial_grad", "max_spatial_grad",
        "skew_spatial_grad", "kurt_spatial_grad" # Skew/Kurt of spatial gradients
    ]

    # Basic check for valid input
    if not isinstance(frames, np.ndarray) or frames.ndim != 3 or frames.shape[0] < 1 or frames.shape[1] < 1 or frames.shape[2] < 1:
        print(f"Warning: Invalid frames shape {getattr(frames, 'shape', 'N/A')} for handcrafted features. Returning NaNs.")
        # Return dict with NaN for all expected features
        return {name: np.nan for name in feature_names}

    try:
        features = {}
        num_frames = frames.shape[0]

        # --- Per-Frame Stat Calculation ---
        # Calculate per-frame stats first to get distributions over time
        frame_means = np.array([np.mean(f) for f in frames])
        frame_stds = np.array([np.std(f) for f in frames])
        frame_maxs = np.array([np.max(f) for f in frames])
        frame_mins = np.array([np.min(f) for f in frames])
        frame_medians = np.array([np.median(f) for f in frames])
        # Use nanpercentile to be robust to potential NaNs within a frame, though unlikely here
        frame_q25 = np.array([np.nanpercentile(f, 25) for f in frames])
        frame_q75 = np.array([np.nanpercentile(f, 75) for f in frames])

        # Temporal gradient stats
        if num_frames > 1:
            temporal_diff = np.abs(np.diff(frames, axis=0))
            temporal_grad_mean_per_diff = np.array([np.mean(diff) for diff in temporal_diff])
        else:
            # Assign empty array or placeholder that results in NaN/0 for stats below
            temporal_grad_mean_per_diff = np.array([])

        # Spatial gradient stats
        spatial_grad_mags_mean_per_frame = []
        spatial_grad_mags_max_per_frame = []
        for frame in frames:
             if frame.shape[0] > 1 and frame.shape[1] > 1:
                 gy, gx = np.gradient(frame)
                 grad_mag = np.sqrt(gx**2 + gy**2)
                 spatial_grad_mags_mean_per_frame.append(np.mean(grad_mag))
                 spatial_grad_mags_max_per_frame.append(np.max(grad_mag))
             else:
                 spatial_grad_mags_mean_per_frame.append(0.0)
                 spatial_grad_mags_max_per_frame.append(0.0)
        spatial_grad_mags_mean_per_frame = np.array(spatial_grad_mags_mean_per_frame)
        spatial_grad_mags_max_per_frame = np.array(spatial_grad_mags_max_per_frame)
        # --- End Per-Frame Stat Calculation ---


        # --- Aggregate Stats Over Time ---

        # Overall intensity/temperature stats
        features["mean_temp"] = np.mean(frame_means) if len(frame_means) > 0 else np.nan
        features["std_temp"] = np.std(frame_means) if len(frame_means) > 1 else 0.0
        features["max_temp"] = np.max(frame_maxs) if len(frame_maxs) > 0 else np.nan
        features["min_temp"] = np.min(frame_mins) if len(frame_mins) > 0 else np.nan
        features["median_temp"] = np.median(frame_medians) if len(frame_medians) > 0 else np.nan
        # Use mean instead of median for q25/q75 for consistency with mean_temp? Mean is fine.
        features["q25_temp"] = np.mean(frame_q25) if len(frame_q25) > 0 else np.nan
        features["q75_temp"] = np.mean(frame_q75) if len(frame_q75) > 0 else np.nan
        # Skewness & Kurtosis of frame means
        features["skew_temp"] = skew(frame_means) if len(frame_means) > 1 else 0.0
        features["kurt_temp"] = kurtosis(frame_means, fisher=True) if len(frame_means) > 1 else 0.0 # Fisher=True -> normal=0

        # Stats of the standard deviation within frames
        features["mean_frame_std"] = np.mean(frame_stds) if len(frame_stds) > 0 else np.nan
        features["std_frame_std"] = np.std(frame_stds) if len(frame_stds) > 1 else 0.0
        # Skewness & Kurtosis of frame stds
        features["skew_frame_std"] = skew(frame_stds) if len(frame_stds) > 1 else 0.0
        features["kurt_frame_std"] = kurtosis(frame_stds, fisher=True) if len(frame_stds) > 1 else 0.0

        # Temporal gradient (frame difference) stats
        if num_frames > 1 and len(temporal_grad_mean_per_diff) > 0:
            features["mean_temp_grad"] = np.mean(temporal_grad_mean_per_diff)
            features["std_temp_grad"] = np.std(temporal_grad_mean_per_diff) if len(temporal_grad_mean_per_diff) > 1 else 0.0
            features["max_abs_temp_grad"] = np.max(temporal_grad_mean_per_diff)
            # Skewness & Kurtosis of temporal gradients
            features["skew_temp_grad"] = skew(temporal_grad_mean_per_diff) if len(temporal_grad_mean_per_diff) > 1 else 0.0
            features["kurt_temp_grad"] = kurtosis(temporal_grad_mean_per_diff, fisher=True) if len(temporal_grad_mean_per_diff) > 1 else 0.0
        else:
            # Assign 0 or NaN for consistency if no temporal diff possible/calculated
            features["mean_temp_grad"] = 0.0
            features["std_temp_grad"] = 0.0
            features["max_abs_temp_grad"] = 0.0
            features["skew_temp_grad"] = 0.0 # Added default
            features["kurt_temp_grad"] = 0.0 # Added default

        # Aggregate spatial gradient stats over frames
        features["mean_spatial_grad"] = np.mean(spatial_grad_mags_mean_per_frame) if len(spatial_grad_mags_mean_per_frame) > 0 else np.nan
        features["std_spatial_grad"] = np.std(spatial_grad_mags_mean_per_frame) if len(spatial_grad_mags_mean_per_frame) > 1 else 0.0
        features["max_spatial_grad"] = np.max(spatial_grad_mags_max_per_frame) if len(spatial_grad_mags_max_per_frame) > 0 else np.nan
        # Skewness & Kurtosis of spatial gradients (mean magnitude per frame)
        features["skew_spatial_grad"] = skew(spatial_grad_mags_mean_per_frame) if len(spatial_grad_mags_mean_per_frame) > 1 else 0.0
        features["kurt_spatial_grad"] = kurtosis(spatial_grad_mags_mean_per_frame, fisher=True) if len(spatial_grad_mags_mean_per_frame) > 1 else 0.0
        # --- End Aggregate Stats Over Time ---


        # --- Final Checks ---
        # Check for any NaN/Inf values produced by calculations
        for k, v in features.items():
            if not np.isfinite(v):
                 # Use a more specific warning message
                 print(f"    Warning: Non-finite value ({v}) generated for feature '{k}'. Will be handled by imputer.")
                 # Ensure NaNs remain NaN, don't replace with 0 here. Let imputer do its job.
                 features[k] = np.nan # Explicitly set to NaN if non-finite

        # Ensure all expected features are present, even if NaN (safeguard)
        for name in feature_names:
            if name not in features:
                print(f"    Error: Feature '{name}' was not calculated. Setting to NaN.")
                features[name] = np.nan
        # --- End Final Checks ---

        return features

    except Exception as e:
         # Add more detail to error message
         print(f"Error during handcrafted feature extraction (calculating stats): {type(e).__name__} - {e}. Returning NaNs for all features.")
         import traceback
         traceback.print_exc() # Print traceback for debugging
         # Return dict with NaN for all expected features
         return {name: np.nan for name in feature_names}