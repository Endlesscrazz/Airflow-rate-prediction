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
from web_app.online_processor import OnlineOLSCalculator
from web_app.utils_viz import draw_overlays
from src_cnn_v3 import config_v3 as cfg

# --- HELPER: LIVE HEATMAP GENERATION (Red/Hot Fix) ---
def generate_live_heatmap_rgb(frame, ols_calc):
    ols_calc.add_frame(frame)
    slope_map = ols_calc.calculate_slope_map()
    
    frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2RGB)
    
    if slope_map.max() > 0:
        map_enhanced = slope_map ** 1.3
        vmin, vmax = map_enhanced.min(), map_enhanced.max()
        threshold = vmin + 0.15 * (vmax - vmin)
        map_enhanced[map_enhanced < threshold] = 0
        
        map_norm = cv2.normalize(map_enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_bgr = cv2.applyColorMap(map_norm, cv2.COLORMAP_HOT)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        mask = map_norm > 10
        out = frame_rgb.copy()
        out[mask] = cv2.addWeighted(frame_rgb[mask], 0.3, heatmap_rgb[mask], 0.7, 0)
        return out
    else:
        return frame_rgb

# --- APP CONFIG ---
st.set_page_config(page_title="ThermalFlow AI", page_icon="🔥", layout="wide")

st.markdown("""
<style>
    div[data-testid="stMetric"] { background-color: #1E1E1E; padding: 10px; border-radius: 5px; border: 1px solid #333; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- STATE ---
if 'processing_done' not in st.session_state: st.session_state.processing_done = False
if 'quantified' not in st.session_state: st.session_state.quantified = False
if 'prediction_results' not in st.session_state: st.session_state.prediction_results = None
if 'frames' not in st.session_state: st.session_state.frames = None
if 'file_name' not in st.session_state: st.session_state.file_name = ""

# --- LOADER ---
@st.cache_resource
def get_predictor():
    model_path = os.path.join("model_assets", "gypsum_10_hole", "best_model.pth")
    scaler_path = os.path.join("model_assets", "gypsum_10_hole", "feature_scaler.pkl")
    if not os.path.exists(model_path):
        st.error(f"❌ Model missing: {model_path}"); return None
    return AirflowPredictor(model_path, scaler_path, max_flow_rate=24.26)

predictor = get_predictor()

# --- WORKER ---
def run_ai_inference(frames_copy, delta_t, sensitivity):
    try: return predictor.process_video(frames_copy, delta_t, sensitivity)
    except Exception as e: print(e); return None

# --- UI ---
st.title("🔥 ThermalFlow AI: Smart Auditor")

# SIDEBAR
with st.sidebar:
    st.header("1. Configuration")
    delta_t = st.number_input("Delta T (Pa)", value=12.0, min_value=1.0)
    sensitivity = st.slider("AI Sensitivity", 0.0, 1.0, 0.25)
    st.divider()
    # HEATMAP TOGGLE
    show_heatmap = st.toggle("Show Fused Signal Map", value=False)
    st.divider()
    uploaded_file = st.file_uploader("Upload .mat Video", type=['mat'])
    
    if uploaded_file and uploaded_file.name != st.session_state.file_name:
        st.session_state.clear()
        st.session_state.file_name = uploaded_file.name
        st.rerun()

# MAIN LOGIC
if uploaded_file and predictor:
    if st.session_state.frames is None:
        with st.spinner("Loading video..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp:
                tmp.write(uploaded_file.getvalue())
                try:
                    mat = scipy.io.loadmat(tmp.name)
                    st.session_state.frames = mat.get('TempFrames').astype(np.float32)
                except Exception as e: st.error(f"Error: {e}")
                finally: os.remove(tmp.name)

    frames = st.session_state.frames
    if frames is not None:
        H, W, T = frames.shape
        col_vis, col_res = st.columns([2, 1])

        # LEFT COLUMN: VISUALIZATION AREA
        with col_vis:
            st.subheader("🎥 Live Site Analysis")
            
            # 1. Main Video/Image Container
            vis_container = st.empty()
            
            # 2. Heatmap Container (Appears BELOW main image)
            heatmap_container = st.empty()
            
            # --- SCENARIO A: ANALYSIS DONE & QUANTIFIED ---
            if st.session_state.processing_done and st.session_state.prediction_results and st.session_state.quantified:
                # 1. Show Main Overlay
                final_img = draw_overlays(
                    st.session_state.prediction_results['vis_frame'], 
                    st.session_state.prediction_results['leaks'],
                    show_labels=True
                )
                vis_container.image(final_img, caption="Final AI Analysis", width='stretch', channels="RGB")
                
                # 2. Show Fused Map (If Toggled)
                if show_heatmap:
                    res = st.session_state.prediction_results
                    norm = cv2.normalize(res['score_map'], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    hm = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
                    hm_rgb = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
                    heatmap_container.image(hm_rgb, caption="Fused Signal Map (Internal)", width='stretch')
                else:
                    heatmap_container.empty()
                
                # 3. Reset Button
                if st.button("🔄 New Scan"):
                    st.session_state.processing_done = False
                    st.session_state.quantified = False
                    st.session_state.prediction_results = None
                    st.rerun()

            # --- SCENARIO B: INITIAL STATE / RUNNING ---
            else:
                # Clear heatmap container if we are back in scanning mode
                heatmap_container.empty()

                if st.button("▶️ Visualize & Detect", type="primary"):
                    
                    # 1. Start Background Inference
                    frames_copy = frames.copy()
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(run_ai_inference, frames_copy, delta_t, sensitivity)
                    
                    # 2. Run Foreground Animation
                    ols_calc = OnlineOLSCalculator(H, W)
                    progress = st.progress(0)
                    
                    for i in range(T):
                        frame = frames[:, :, i]
                        live_view = generate_live_heatmap_rgb(frame, ols_calc)
                        vis_container.image(live_view, caption=f"Scanning Frame {i+1}/{T}", width='stretch', channels="RGB")
                        progress.progress((i+1)/T)
                        time.sleep(0.05) 
                    
                    progress.empty()
                    
                    # 3. Wait for Background Result
                    if not future.done():
                        with st.spinner("Finalizing Quantification Models..."):
                            results = future.result()
                    else:
                        results = future.result()
                    
                    if results:
                        st.session_state.prediction_results = results
                        st.session_state.processing_done = True
                        st.rerun() 
                    else:
                        st.error("AI Analysis Failed.")
                
                # Placeholder text if not running
                elif not st.session_state.processing_done:
                    vis_container.info("Click 'Visualize & Detect' to begin scan.")
                
                # If processing done but NOT quantified yet (Waiting for user to click Quantify)
                elif st.session_state.processing_done:
                     # Show last frame of live scan (Scan Complete)
                     # Since we don't persist the live frame in this simplified logic to save memory, 
                     # we can show the clean visual frame or a placeholder "Ready".
                     # A good UX choice: Show the 'vis_frame' (Clean RGB)
                     clean_frame = st.session_state.prediction_results['vis_frame']
                     vis_container.image(clean_frame, caption="Scan Complete. Ready to Quantify.", width='stretch', channels="RGB")


        # RIGHT COLUMN: RESULTS AREA
        with col_res:
            st.subheader("📊 Audit Results")
            
            if st.session_state.processing_done:
                if st.session_state.quantified:
                    # --- FINAL METRICS ---
                    res = st.session_state.prediction_results
                    leaks = res['leaks']
                    total = sum(l['flow_rate'] for l in leaks)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Total Leakage", f"{total:.2f}", "L/min")
                    c2.metric("Est. Cost", f"${total * 45:.0f}", "/yr")
                    st.divider()
                    
                    st.markdown("### Detected Leaks")
                    if not leaks: st.info("No leaks detected.")
                    else:
                        for leak in leaks:
                            with st.expander(f"**ID: {leak['id']}** | {leak['flow_rate']:.2f} L/m"):
                                st.image(leak['patch_vis'], width=80)
                else:
                    # --- QUANTIFY BUTTON ---
                    if st.button("💰 Quantify Airflow", type="secondary"):
                        st.session_state.quantified = True
                        st.rerun()
            else:
                st.info("Results will appear here.")

else:
    # Landing State
    st.markdown("---")
    st.markdown("""
    ### Welcome to ThermalFlow AI
    **Instructions:**
    1. Upload a thermal video (`.mat`) in the sidebar.
    2. Adjust the **Delta T** (Pressure difference).
    3. Click **Visualize & Detect** to watch the live scan.
    4. Click **Quantify Airflow** to get precise measurements.
    """)