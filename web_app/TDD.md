Technical Design Document: ThermalFlow AI (Prototype)
Document Info	Details
Project Name	ThermalFlow AI Inspector
Version	1.0 (Architecture Decision Record)
Date	December 6, 2025
Status	Approved for Implementation
Tech Stack	Python, Streamlit, PyTorch, Google Colab (Backend)
1. Overview & Problem Statement

1.1 Problem Statement
Building energy audits are currently labor-intensive and subjective. Auditors use handheld thermal cameras to identify leaks but lack the tools to quantify the severity of those leaks (in Liters/minute) or estimate financial loss. Existing software provides simple color palettes but does not use deep learning to understand the physics of airflow, often leading to false positives from reflective surfaces or residual heat.

1.2 Solution Overview
ThermalFlow AI is a web-based analytical dashboard. It ingests raw radiometric thermal video (.mat files), stabilizes the footage, isolates thermal anomalies using a "Fused Signal" (Temporal 
×
×
 Spatial) approach, and uses a Hybrid CNN-LSTM neural network to predict precise airflow rates.

2. Goals & Non-Goals

2.1 Goals
Quantification: Predict airflow leakage rates with an MAE < 1.0 L/min.
Explainability: Visualize why a leak was detected by showing the "Fused Signal Map" and the internal "Standardized AI Patches" (32x32 crops).
Usability: Provide a "No-Code" interface for auditors to upload files and see financial impact metrics immediately.
Visual Precision: Overlay precise Oriented Bounding Boxes (OBB) on leaks, tracking them even if the camera moves.
2.2 Non-Goals
Real-Time Streaming: We are not connecting directly to a camera via USB/RTSP for this version. Input is file-based (.mat).
User Authentication: No login system required for the prototype.
Database Persistence: History of uploads will not be saved; the app is session-based.
3. System Architecture
The system follows a Client-Server-Tunnel architecture to leverage high-performance GPUs (Colab) while serving a web frontend.

3.1 High-Level Diagram
code
Mermaid
graph LR
    User[End User Browser] <-->|HTTPS via Ngrok| Streamlit[Streamlit Frontend (Colab)]
    
    subgraph "Colab Backend Runtime"
        Streamlit --> Controller[App Logic Controller]
        Controller --> Pre[Preprocessor]
        Controller --> Detector[Fused Signal Detector]
        Controller --> SAM[FastSAM / SAM Wrapper]
        Controller --> Model[Hybrid CNN-LSTM]
        
        Model --> GPU[(NVIDIA A100/T4 GPU)]
    end
3.2 Component Definitions
Frontend (Streamlit): Renders the UI, handles file uploads, and manages Session State (variables that persist during user interaction).
Inference Engine (Backend): A Python class (AirflowPredictor) that wraps your V3 scripts. It keeps the model loaded in GPU memory to ensure low latency.
Tunnel (Ngrok): Exposes the local Streamlit port (8501) running inside the Google Colab container to the public internet.
4. Data Models & Data Flow

4.1 Input Data Model (.mat)
The application expects a MATLAB file containing:

TempFrames: 3D Array (Height, Width, Time) of float32. Represents raw temperature values.
Metadata (Derived): Voltage, Pressure, Delta T (parsed from filename or user input).
4.2 Internal "Leak Object"
For every detected anomaly, the system passes this dictionary structure:

code
JSON
{
  "id": 1,
  "frame_idx": 75,
  "centroid": [200, 150],
  "bbox_rotated": ((cx, cy), (w, h), angle),
  "mask_binary": [2D Array bool],
  "patch_32x32": [32x32 Array float32],
  "tabular_features": [area, aspect_ratio, delta_t],
  "prediction_lpm": 4.2
}
4.3 Execution Flow
Ingest: User uploads file -> Save to temp dir -> Load via scipy.io.
Global Detection: Run FusedSignalDetector on the full video stack to generate the Score Map.
Localization: Identify top N peaks from the Score Map.
Segmentation: Pass peaks to SAM to get binary masks.
Standardization: Convert masks to OBBs -> Crop -> Rotate -> Resize to 32x32.
Inference: Batch process all 32x32 patches through CNN-LSTM (GPU).
Aggregation: Sum predictions to get "Total Airflow."
Rendering: Draw overlays on the average frame and push to UI.
5. User Interface (UI) Specification
Based on your design prototype, the UI is divided into 4 logical zones.

Zone A: Sidebar (Control & Input)
File Uploader: Drag & Drop for .mat.
Delta T Input: Numeric input (default 5.0 Pa).
Sensitivity Slider: Adjusts threshold for FusedSignalDetector (Z-Score).
Toggle: "View Fused Signal Map" (Switches main view mode).
Zone B: Heads Up Display (Top Metrics)
Card 1: Total Airflow Loss. (e.g., 15.4 L/min). Calculation: Sum of all active leak predictions.
Card 2: Active Leaks. (e.g., 4 Detects). Calculation: Count of valid OBBs.
Card 3: Est. Annual Cost. (e.g., ~$450/yr). Calculation: Flow Rate * Cost Factor (constant).
Zone C: The Workspace (Main Visualization)
Video Player: Shows the thermal feed.
Overlays:
Green Bounding Boxes: Rotated rectangles around leaks.
Floating Labels: "ID:01 | 4.2 L/m" drawn next to the box using OpenCV.
Interaction: Play/Pause slider (Standard Streamlit video controls).
Zone D: Explainability (Right & Bottom Panels)
Right Panel (List): A scrollable list of "Leak Cards." Each card shows a thumbnail of the leak and its specific flow rate.
Bottom Panel (AI Internals): "X-Ray View." Displays the raw 32x32 input tensors that the CNN is seeing. This proves to the user that the model is looking at a specific shape (vertical slit vs. circle).
6. Tech Stack & Rationale
Component	Choice	Rationale
Language	Python 3.10+	Native support for PyTorch/SciPy.
Frontend	Streamlit	Fastest way to build data dashboards. Handles .mat files and plotting natively.
Computer Vision	OpenCV	Robust image manipulation (drawing OBBs, text overlays, normalizing frames).
Deep Learning	PyTorch	Required to load your trained .pth model.
Segmentation	SAM (Segment Anything)	High-precision masking. We will cache the model to avoid reloading latency.
Math Backend	SciPy / NumPy	Essential for loading .mat files and signal processing (Theil-Sen).
Deployment	Colab Pro + Ngrok	Provides the necessary GPU (A100/T4) for free; Ngrok bridges it to the web.
7. Security, Privacy, Performance

7.1 Performance Optimization
Caching: Use @st.cache_resource for the Model, SAM, and Scaler. They load once on startup, never again.
Lazy Loading: Use @st.cache_data for the .mat file processing. If the user changes the "Sensitivity Slider," we re-run detection but do not re-load the file.
Resolution: Process internally at full resolution, but display video in UI at 640px width to save bandwidth.
7.2 Security Considerations
Ephemeral Data: Uploaded files are stored in /tmp and deleted when the runtime disconnects.
Ngrok Security: The tunnel URL is random and temporary. For extra security, we can enable a password on the Streamlit app.
8. Testing & Development Plan

Phase 1: The Skeleton (Day 1)
Setup VS Code Remote SSH to Colab.
Create app.py.
Implement File Upload -> Load .mat -> Display Raw Video.
Phase 2: The Core Logic Integration (Day 2)
Import FusedSignalDetector and HybridCropRegressor.
Implement the "Inference Pipeline" (Detect -> SAM -> Predict).
Verify that app.py can load best_model.pth correctly.
Phase 3: UI Realization (Day 3)
Implement the OpenCV Overlay function: Draw the green rotated boxes and text on the frames before sending them to the frontend.
Build the Side Panel and Bottom Panel layouts using st.columns.
Implement the Metric Cards (Flow/Cost calculations).
Phase 4: Polish (Day 4)
Add the "Fused Signal Map" toggle.
Tune the color palettes to match the "Dark Mode" aesthetic.
Record the demo video.
9. API / Internal Interfaces
Since this is a monolithic app, these are Python functions, not REST endpoints.

code
Python
# Interface for the Inference Engine
def run_full_analysis(video_frames, delta_t, sensitivity):
    """
    Args:
        video_frames: (H, W, T) Float32 array
        delta_t: Float
        sensitivity: Float (0.0 - 1.0)
    Returns:
        analysis_results: Dict containing:
            - 'total_flow': float
            - 'leaks': List of LeakObjects
            - 'processed_video_path': str (path to mp4 with overlays)
            - 'debug_patches': List of 32x32 arrays
    """