# ThermalFlow AI: Automated Air Leakage Auditing

**Deep Learning Pipeline & Real-Time Visualization Dashboard**  
**Status:** Prototype (V3) | **Tech Stack:** PyTorch, Streamlit, OpenCV, SAM (Segment Anything)

ThermalFlow AI is a deep learning system designed to automate building energy audits. By ingesting raw radiometric thermal video, the system identifies air leakage points and—crucially—quantifies the airflow rate (L/min) and estimates financial loss without the need for manual interpretation.

---

## Key Innovation: The "Two-Stream" Architecture

Unlike traditional black-box AI, ThermalFlow utilizes a **Two-Stream Asynchronous Architecture** to bridge the gap between heavy Deep Learning models and real-time User Experience:

**Foreground Stream (CPU):**  
Runs an Online OLS (Ordinary Least Squares) algorithm to render a Live Heatmap instantaneously (30 FPS). This mimics the experience of using a physical thermal camera, allowing auditors to see leaks develop in real-time.

**Background Stream (GPU):**  
Runs the heavy V3 Inference Pipeline (Signal Detection + SAM Segmentation + CNN-LSTM) silently in the background to perform precise quantification.

---

## Evolution of Performance

The project has evolved through rigorous experimentation, moving from basic regression to a shape-aware hybrid system.

| Version | Methodology | Performance (R²) | Limitation |
|------|-----------|------------------|------------|
| V1 | Handcrafted Features (Optical Flow) | < 0.30 | Failed to capture temporal dynamics. |
| V2 | Fixed-Crop CNN-LSTM | ≈ 0.52 | Struggled with non-circular leaks (slits) and noise. |
| **V3 (Current)** | **Hybrid Shape-Aware (SAM + OBB)** | **0.86 – 0.89** | **Solved! Robust to geometry and texture.** |

### V3 Results by Material

- **Gypsum Board (10-Hole):**  
  R² = 0.863 | MAE = 0.078 L/min

- **Brick Cladding:**  
  R² = 0.892 | MAE = 0.117 L/min

- **HardyBoard:**  
  R² = 0.855 | MAE = 0.119 L/min

---

## The V3 Inference Pipeline

The core logic (`src_cnn_v3`) follows a physics-informed computer vision pipeline:

**Fused Signal Detection:**  
Combines Temporal Trends (Theil-Sen Slope) and Spatial Heat (Local Z-Score) to create a high-contrast map, detecting anomalies invisible to the naked eye.

**Smart Localization:**  
Uses Adaptive Thresholding to detect leaks without knowing the count beforehand, merging proximity peaks to handle complex shapes (e.g., vertical slits).

**Segmentation (SAM):**  
Utilizes the Segment Anything Model with dynamic box prompting to generate precise binary masks.

**Hybrid Quantification:**
- **Visual Stream:** A CNN-LSTM network processes the raw thermal video stack (150 frames).
- **Geometric Stream:** An MLP processes handcrafted features (Area, Aspect Ratio, Extent, Pressure).
- **Fusion:** Both streams are combined to predict the scalar flow rate.

---

## Web Application (Prototype)

The repository includes a fully functional Streamlit application (`web_app/`) for demonstration.

### Features

- **Live Site Analysis:** Real-time visualization of developing thermal anomalies using `COLORMAP_HOT`.
- **AI Quantification:** Overlays precise Green Oriented Bounding Boxes (OBB) on detected leaks.
- **Financial Metrics:** Estimates annual financial loss ($/yr) based on total leakage.
- **Explainability:** Toggleable "Fused Signal Map" to verify AI attention regions.

---

## Running the App Locally

### Requirements

- Python 3.10+
- GPU Recommended (NVIDIA CUDA or Mac M1/M2 MPS)

```bash
# 1. Clone Repository
git clone https://github.com/Endlesscrazz/Airflow-rate-prediction.git
cd Airflow-rate-prediction

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Download Model Assets
# Ensure best_model.pth and feature_scaler.pkl are in:
# model_assets/gypsum_10_hole/

# 4. Run Streamlit App
streamlit run web_app/app.py
```

---

## Project Structure

```
.
├── src_cnn_v3/               # Core V3 Learning Library
│   ├── core/                 # Detection, Preprocessing, and Image Logic
│   ├── models_v3.py          # Hybrid CNN-LSTM Architecture
│   ├── dataset_utils_v3.py   # PyTorch Data Loaders
│   └── config_v3.py          # Central Configuration
│
├── web_app/                  # User Interface (Streamlit)
│   ├── app.py                # Main Application Entry Point
│   ├── inference.py          # Orchestrator for SAM + CNN-LSTM
│   ├── online_processor.py   # CPU-Optimized Live Visualizer
│   └── utils_viz.py          # Drawing & Overlay Utilities
│
├── scripts_v3/               # Offline Processing Scripts
│   ├── generate_masks_v3.py  # SAM Integration for Training Data
│   └── train_v3.py           # Training Loop
│
└── model_assets/             # Trained Weights & Scalers
```

---

## Roadmap & Future Work

- **Edge Deployment:** Distilling the heavy Standard SAM (90M parameters) to FastSAM (5M parameters) for deployment on iPads/Tablets without cloud dependency.
- **Persistent Hosting:** Containerizing the application (Docker) for deployment on University High-Performance Computing (CHPC) nodes.
- **Live Camera Integration:** Moving from file-based uploads (`.mat`) to live USB streaming using the Fluke/FLIR SDK.

---

## Contributors

**Shreyas Patil**  
Graduate Research Assistant, University of Utah  

**Advisor:**  
Dr. Yongzhi Qu — Utah Lab of Artificial Intelligence Powered Systems
