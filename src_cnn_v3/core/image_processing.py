# src_cnn_v3/debug_files/image_processing.py
import cv2
import numpy as np

def crop_rotate_resize(image, center, size, angle, target_size=(32, 32)):
    """
    Extracts an OBB from an image and resizes it to target_size.
    
    Args:
        image: Full thermal frame (H, W).
        center: (x, y) tuple.
        size: (width, height) tuple from minAreaRect.
        angle: Angle from minAreaRect.
        target_size: (32, 32).
        
    Returns:
        standardized_patch: (32, 32) numpy array.
    """
    # 1. Get the rotation matrix
    # OpenCV minAreaRect angle is usually in range [-90, 0).
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 2. Rotate the entire image
    # We use cubic interpolation for smoother rotation
    img_h, img_w = image.shape[:2]
    rotated_img = cv2.warpAffine(image, rot_mat, (img_w, img_h), flags=cv2.INTER_CUBIC)
    
    # 3. Crop the upright rectangle
    # "size" from minAreaRect is (width, height) relative to the rotation angle
    w, h = size
    
    # Ensure we are cropping the correct region from the rotated image
    # The center of rotation remains the same in the rotated image
    x = center[0] - w / 2
    y = center[1] - h / 2
    
    # Extract crop with boundary checks
    # We use getRectSubPix which handles float coordinates nicely (sub-pixel accuracy)
    try:
        crop = cv2.getRectSubPix(rotated_img, (int(w), int(h)), center)
    except Exception as e:
        # Fallback for edge cases where getRectSubPix might fail
        x, y, w, h = int(x), int(y), int(w), int(h)
        crop = rotated_img[max(0, y):y+h, max(0, x):x+w]

    if crop is None or crop.size == 0:
        return np.zeros(target_size, dtype=np.float32)

    # 4. Resize to Target (Squash and Stretch)
    # Use INTER_AREA if shrinking, INTER_LINEAR if growing
    interp = cv2.INTER_AREA if (crop.shape[0] > target_size[0]) else cv2.INTER_LINEAR
    
    standardized_patch = cv2.resize(crop, target_size, interpolation=interp)
    
    return standardized_patch