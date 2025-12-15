# src_cnn_v3/core/leak_detection.py
import numpy as np
import cv2
from scipy import stats
from skimage.feature import peak_local_max
from joblib import Parallel, delayed

class FusedSignalDetector:
    """
    Detects thermal leak epicenters using Temporal Trend * Spatial Heat fusion.
    v3.1 Update: Aligned with V2 Debug Logic (No enforced normalization/smoothing).
    """
    def __init__(self, heat_power=1.4):
        self.heat_power = heat_power

    def _calculate_slope_row(self, row_data, t):
        """Helper for parallel processing."""
        W = row_data.shape[0]
        slopes = np.zeros(W, dtype=np.float32)
        for c in range(W):
            try:
                # Theil-Sen is robust to outliers
                res = stats.theilslopes(row_data[c, :], t)
                slopes[c] = res[0]
            except:
                slopes[c] = 0.0
        return slopes

    def get_score_map(self, frames):
        """
        Generates the Fused Score Map: Temporal Trend * (Local Z-Score ^ power).
        """
        # Ensure frames are float64 for precision
        frames = frames.astype(np.float64)
        H, W, T = frames.shape
        
        # --- PREPROCESSING (The Missing Link) ---
        # Normalize each frame by its spatial mean to handle ambient fluctuations
        # This matches src_cnn_v2/find_leaking_holes.py logic
        frame_means = frames.mean(axis=(0, 1), keepdims=True) # Shape (1, 1, T)
        frame_means[frame_means < 1e-6] = 1.0
        norm_frames = frames / frame_means
        
        # --- 1. Calculate Temporal Trend Map (Theil-Sen) ---
        t = np.arange(T)
        results = Parallel(n_jobs=-1)(
            delayed(self._calculate_slope_row)(norm_frames[r, :, :], t) 
            for r in range(H)
        )
        temporal_map = np.vstack(results)
        temporal_map = np.maximum(temporal_map, 0.0) 

        # --- 2. Calculate Spatial Heat Z-Score ---
        # Use Normalized frames for Z-Score too (V2 logic)
        temp_mean = np.mean(norm_frames, axis=2)
        
        # Calculate Local Z-Score
        loc_mu = cv2.GaussianBlur(temp_mean, (31, 31), 0)
        temp_mean_sq = cv2.GaussianBlur(temp_mean**2, (31, 31), 0)
        sigma_sq = np.maximum(temp_mean_sq - loc_mu**2, 0)
        loc_sd = np.sqrt(sigma_sq) + 1e-9
        
        heat_z = np.maximum((temp_mean - loc_mu) / loc_sd, 0.0)

        # --- 3. Fuse ---
        score_map = temporal_map * (heat_z ** self.heat_power)
        
        # Clean up borders
        border = 10
        score_map[:border, :] = 0
        score_map[-border:, :] = 0
        score_map[:, :border] = 0
        score_map[:, -border:] = 0
        
        return score_map.astype(np.float32)

    def find_peaks(self, score_map, min_distance=30):
        """
        Finds leak coordinates.
        Matches V2: Threshold is 5% of max.
        """
        global_max = score_map.max()
        threshold_abs = global_max * 0.05 
        
        coords = peak_local_max(score_map, min_distance=min_distance, threshold_abs=threshold_abs)
        
        if coords.size == 0:
            return []

        candidates = []
        for r, c in coords:
            candidates.append({
                'centroid': (c, r), # (x, y)
                'score': float(score_map[r, c])
            })
        
        # Sort descending by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates