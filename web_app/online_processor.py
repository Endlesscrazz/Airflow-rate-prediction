# web_app/online_processor.py
import numpy as np
import cv2

class OnlineOLSCalculator:
    """
    Lightweight OLS for Real-Time Visualization (CPU Optimized).
    """
    def __init__(self, H, W):
        self.H, self.W = H, W
        self.n = 0
        self.sum_x = np.float64(0)
        self.sum_y = np.zeros((H, W), dtype=np.float64)
        self.sum_xy = np.zeros((H, W), dtype=np.float64)
        self.sum_x2 = np.float64(0)

    def add_frame(self, frame):
        t = self.n
        self.sum_x += t
        self.sum_y += frame
        self.sum_xy += frame * t
        self.sum_x2 += t**2
        self.n += 1

    def calculate_slope_map(self):
        if self.n < 2:
            return np.zeros((self.H, self.W), dtype=np.float32)
        
        numerator = self.n * self.sum_xy - self.sum_x * self.sum_y
        denominator = self.n * self.sum_x2 - self.sum_x**2
        
        if denominator == 0:
            return np.zeros((self.H, self.W), dtype=np.float32)
            
        slope = numerator / denominator
        slope = np.nan_to_num(slope, copy=False, nan=0.0)
        return np.maximum(slope, 0.0).astype(np.float32)

def generate_live_heatmap(frame, ols_calc):
    """
    Updates OLS and returns a blended image mimicking the Fused Signal Map style.
    (Clean Black background, Glowing Red/Orange hotspots)
    """
    ols_calc.add_frame(frame)
    slope_map = ols_calc.calculate_slope_map()
    
    if slope_map.max() > 0:
        # 1. Enhance Contrast to make peaks pop
        map_enhanced = slope_map ** 1.3
        
        # 2. Threshold: Clip bottom 20% to Black to remove blue background noise
        vmin = map_enhanced.min()
        vmax = map_enhanced.max()
        threshold = vmin + 0.20 * (vmax - vmin)
        map_enhanced[map_enhanced < threshold] = 0
        
        # 3. Apply Colormap (INFERNO looks closest to "Heat")
        map_norm = cv2.normalize(map_enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(map_norm, cv2.COLORMAP_INFERNO)
        
        # 4. Create Mask: Only blend where the heatmap is actually hot
        # This keeps the background looking like the original video (Grayscale)
        mask = map_norm > 15 # Binary mask of hot spots
        
        # Prepare original frame (Grayscale -> RGB)
        frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2RGB)
        
        # 5. Blend
        out = frame_rgb.copy()
        # Blend: 30% Video + 70% Heatmap (makes the leak glow strongly)
        out[mask] = cv2.addWeighted(frame_rgb[mask], 0.3, heatmap[mask], 0.7, 0)
        
        return out
    else:
        frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2RGB)