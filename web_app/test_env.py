import torch
import cv2
import streamlit as st
import segment_anything
import ultralytics # This is FastSAM/YOLO
from sklearn.preprocessing import StandardScaler

print("\n=== SYSTEM CHECK ===")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ Device: {'MPS (Apple Metal)' if torch.backends.mps.is_available() else 'CPU'}")
print(f"✅ Streamlit: {st.__version__}")
print(f"✅ Standard SAM: Installed (Location: {segment_anything.__file__})")
print(f"✅ FastSAM (Ultralytics): {ultralytics.__version__}")
print("✅ OpenCV & Scikit-Learn loaded.")
print("====================\n")