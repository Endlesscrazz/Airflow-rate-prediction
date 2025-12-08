# web_app/utils_viz.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_patch(patch_32):
    """
    Converts a raw float32 patch (normalized 0-1) to a colormapped RGB image 
    for Streamlit display.
    """
    # 1. Normalize min-max to 0-1 range for visualization
    p_min, p_max = patch_32.min(), patch_32.max()
    if p_max > p_min:
        p_norm = (patch_32 - p_min) / (p_max - p_min)
    else:
        p_norm = patch_32
    
    # 2. Convert to Uint8
    p_uint8 = (p_norm * 255).astype(np.uint8)
    
    # 3. Apply 'Inferno' Colormap (looks like thermal data)
    # OpenCV applies colormap in BGR, we convert to RGB for Streamlit
    p_color = cv2.applyColorMap(p_uint8, cv2.COLORMAP_INFERNO)
    p_rgb = cv2.cvtColor(p_color, cv2.COLOR_BGR2RGB)
    
    return p_rgb

def draw_overlays(image_rgb, leaks, show_masks=True, show_boxes=True, show_labels=True):
    """
    Draws green Oriented Bounding Boxes (OBBs), flow rate labels, and masks.
    """
    canvas = image_rgb.copy()
    
    # Sort leaks by flow rate (descending) so biggest text draws on top
    sorted_leaks = sorted(leaks, key=lambda x: x['flow_rate'], reverse=True)
    
    for leak in sorted_leaks:
        # 1. Draw Mask (Semi-transparent Green)
        if show_masks and leak['mask'] is not None:
            color = (0, 255, 0) # Green
            mask = leak['mask']
            
            # Blend only on the masked pixels
            roi = canvas[mask]
            colored_roi = np.zeros_like(roi)
            colored_roi[:] = color
            
            # Blend: 0.7 Original + 0.3 Green
            canvas[mask] = cv2.addWeighted(roi, 0.7, colored_roi, 0.3, 0)
            
        # 2. Draw Oriented Bounding Box
        if show_boxes:
            box = leak['obb_box']
            cv2.drawContours(canvas, [box], 0, (0, 255, 0), 2)
            
        # 3. Draw Label
        if show_labels:
            cx, cy = leak['centroid']
            # If ID is unknown (Ux), show just flow rate
            id_str = f"{leak['id']}" if isinstance(leak['id'], int) else leak['id']
            label_text = f"ID:{id_str} | {leak['flow_rate']:.1f} L/m"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (w, h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Black background rect for text
            cv2.rectangle(canvas, (cx, cy - h - 5), (cx + w, cy + 5), (0, 0, 0), -1)
            cv2.putText(canvas, label_text, (cx, cy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
    return canvas

def create_heatmap_overlay(gray_image, score_map):
    """
    Overlays the 'Fused Signal Map' (Red/Hot) onto the Grayscale video frame.
    """
    score_norm = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(score_norm, cv2.COLORMAP_HOT)
    
    if len(gray_image.shape) == 2:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        
    blended = cv2.addWeighted(gray_image, 0.6, heatmap, 0.4, 0)
    return blended