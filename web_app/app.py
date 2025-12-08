import streamlit as st
import time
import cv2
import numpy as np
import scipy.io
import tempfile
import concurrent.futures
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from web_app.inference import AirflowPredictor
from web_app.online_processor import OnlineOLSCalculator, generate_live_heatmap
from web_app.utils_viz import draw_overlays
from src_cnn_v3 import config_v3 as cfg

# --- APP CONFIG ---
st.set_page_config(
    page_title="ThermalFlow AI: Smart Audit", 
    page_icon="🔥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    div[data-testid="stMetric"] { background-color: #1E1E1E; padding: 15px; border-radius: 8px; border: 1px solid #333; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    .success-text { color: #4CAF50; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE MANAGEMENT ---
# We need to persist data across Streamlit reruns
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'frames' not in st.session_state:
    st.session_state.frames = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""

# --- CACHED RESOURCE LOADING ---
@st.cache_resource
def get_predictor():
    """
    Loads the Heavy AI Model once.
    """
    # Define paths to your specific model assets
    # Update these if you change folders!
    model_path = os.path.join("model_assets", "gypsum_10_hole", "best_model.pth")
    scaler_path = os.path.join("model_assets", "gypsum_10_hole", "feature_scaler.pkl")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        return None
        
    # Initialize Predictor (uses device from config_v3.py, likely 'mps' on M1)
    return AirflowPredictor(model_path, scaler_path, max_flow_rate=24.26)

predictor = get_predictor()

# --- BACKGROUND WORKER ---
def run_ai_inference(frames_copy, delta_t, sensitivity):
    """
    This runs in a separate thread to avoid blocking the UI video playback.
    """
    try:
        # Run the full V3 pipeline
        results = predictor.process_video(frames_copy, delta_t, sensitivity)
        return results
    except Exception as e:
        print(f"Background Thread Error: {e}")
        return None

# --- MAIN UI LAYOUT ---
st.title("🔥 ThermalFlow AI: Smart Auditor")

# 1. SIDEBAR: Configuration
with st.sidebar:
    st.header("1. Configuration")
    st.info(f"System: {cfg.DEVICE.upper()} Acceleration")
    
    delta_t = st.number_input("Delta T (Pa)", value=12.0, min_value=1.0)
    sensitivity = st.slider("AI Sensitivity", 0.0, 1.0, 0.35, help="Lower = detects faint leaks. Higher = detects only hot spots.")
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Thermal Video (.mat)", type=['mat'])
    
    # Reset state if new file uploaded
    if uploaded_file and uploaded_file.name != st.session_state.file_name:
        st.session_state.frames = None
        st.session_state.processing_done = False
        st.session_state.prediction_results = None
        st.session_state.file_name = uploaded_file.name

# 2. MAIN LOGIC
if uploaded_file and predictor:
    # A. Load File (Lazy Load)
    if st.session_state.frames is None:
        with st.spinner("Loading video into memory..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp:
                tmp.write(uploaded_file.getvalue())
                try:
                    mat_data = scipy.io.loadmat(tmp.name)
                    # Extract frames: Expecting (H, W, T)
                    raw_frames = mat_data.get('TempFrames').astype(np.float32)
                    st.session_state.frames = raw_frames
                except Exception as e:
                    st.error(f"Failed to load .mat file: {e}")
                finally:
                    os.remove(tmp.name)

    frames = st.session_state.frames
    if frames is not None:
        H, W, T = frames.shape
        
        # B. Layout Columns
        col_vis, col_res = st.columns([2, 1])

        with col_vis:
            st.subheader("🎥 Live Site Analysis")
            vis_container = st.empty()
            progress_bar = st.empty()
            status_text = st.empty()
            
            # --- ACTION: VISUALIZE ---
            # Only show start button if we haven't processed or user wants to restart
            if st.button("▶️ Visualize & Detect", type="primary"):
                
                # Reset previous results
                st.session_state.processing_done = False
                st.session_state.prediction_results = None
                
                status_text.markdown("**Status:** Initializing AI...")
                
                # 1. Launch Background Thread
                # We copy frames to prevent race conditions
                frames_copy = frames.copy()
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(run_ai_inference, frames_copy, delta_t, sensitivity)
                
                # 2. Run Foreground Visualization Loop
                ols_calc = OnlineOLSCalculator(H, W)
                status_text.markdown("**Status:** Scanning video stream...")
                
                # Simulate playback (adjust sleep for speed)
                playback_speed = 0.03 # ~30 FPS
                
                for i in range(T):
                    frame = frames[:, :, i]
                    
                    # Generate blended heat map
                    live_view = generate_live_heatmap(frame, ols_calc)
                    
                    # Update UI
                    vis_container.image(live_view, caption=f"Frame {i+1}/{T}", width='stretch', channels="RGB")
                    progress_bar.progress((i+1)/T)
                    
                    time.sleep(playback_speed)
                
                progress_bar.empty()
                
                # 3. Wait for Background Thread (if video finished before AI)
                if not future.done():
                    with st.spinner("Finalizing Quantification Models..."):
                        results = future.result()
                else:
                    results = future.result()
                
                if results:
                    st.session_state.prediction_results = results
                    st.session_state.processing_done = True
                    status_text.markdown(":white_check_mark: **Status:** Scan Complete. Ready to Quantify.")
                    st.toast("AI Analysis Complete!", icon="✅")
                else:
                    st.error("AI Analysis Failed.")

        with col_res:
            st.subheader("📊 Audit Results")
            
            # --- ACTION: QUANTIFY ---
            # Only active once processing is done
            if st.session_state.processing_done:
                if st.button("💰 Quantify Airflow", type="secondary"):
                    res = st.session_state.prediction_results
                    leaks = res['leaks']
                    
                    # 1. Top Metrics
                    total_flow = sum(l['flow_rate'] for l in leaks)
                    c1, c2 = st.columns(2)
                    c1.metric("Total Leakage", f"{total_flow:.2f}", "L/min")
                    c2.metric("Est. Cost", f"${total_flow * 45:.0f}", "/yr")
                    
                    st.divider()
                    
                    # 2. Leak List
                    st.markdown("### Detected Leaks")
                    if not leaks:
                        st.info("No leaks detected above sensitivity threshold.")
                    else:
                        for leak in leaks:
                            with st.expander(f"**ID: {leak['id']}** | {leak['flow_rate']:.2f} L/m"):
                                st.image(leak['patch_vis'], caption="Normalized Tensor Input", width=100)
                                st.caption(f"Area: {leak['debug_features']['Area']}px")
                    
                    # 3. Update Visualizer with Final Result
                    # Overlay precise green boxes on the final frame
                    final_overlay = draw_overlays(res['vis_frame'], leaks)
                    vis_container.image(final_overlay, caption="Final AI Analysis Overlay", width='stretch', channels="RGB")
            
            else:
                st.info("Click 'Visualize & Detect' to begin the site audit.")

else:
    # Landing State
    st.markdown("---")
    st.markdown("""
    ### 👋 Welcome to ThermalFlow AI
    **Instructions:**
    1. Upload a thermal video (`.mat`) in the sidebar.
    2. Adjust the **Delta T** (Pressure difference).
    3. Click **Visualize & Detect** to watch the live scan.
    4. Click **Quantify Airflow** to get precise measurements.
    """)