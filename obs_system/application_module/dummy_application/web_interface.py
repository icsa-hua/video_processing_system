from __future__ import annotations

from obs_system.application_module.dummy_application.pipeline_config import DEFAULT_BENCH_LABELS
from obs_system.utils.logger import get_logger

import os
import tempfile
import warnings
from pathlib import Path
from typing import Text

import requests
import streamlit as st
from PIL import Image


logger = get_logger("obs_system." + __name__)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BACKEND_URL = f"http://{st.get_option('server.address')}:8000"
static_folder = PACKAGE_ROOT / "static"
logo_image = static_folder / "logo.png"

MODEL_OPTIONS = {
    "TensorRT FP16 engine": "assets/compressed_models/mixed_dataset_trained_yolov8s_mixed_batch_trt_fp16_noint8.engine",
    "YOLOv8 ONNX": "assets/compressed_models/mixed_dataset_trained_yolov8s.onnx",
    "YOLOv8 PT": "assets/compressed_models/yolov8s.pt",
}


def _cleanup_uploaded_file() -> None:
    temp_path = st.session_state.get("uploaded_video_path")
    if not temp_path:
        return
    try:
        os.remove(temp_path)
    except OSError as exc:
        warnings.warn(f"Error deleting temporary file: {exc}. File may not exist.")
    finally:
        st.session_state["uploaded_video_path"] = None


st.set_page_config(
    page_title="EDGEAI-VPS",
    page_icon=logo_image,
    layout="wide",
)

if "uploaded_video_path" not in st.session_state:
    st.session_state["uploaded_video_path"] = None

js = f"""
<script>
window.addEventListener("beforeunload", async function() {{
    fetch("{BACKEND_URL}/shutdown", {{ method: "POST" }});
}});
</script>
"""
st.markdown(js, unsafe_allow_html=True)

with st.sidebar:
    logo = Image.open(logo_image)
    icon, title = st.columns([0.4, 0.63])

    with icon:
        st.image(logo, width=100)

    with title:
        repo_link: Text = "https://edge-ai-tech.eu/"
        st.markdown(
            f"""<h4 style='color: #f0eef0;'>Real-Time Intersection Monitoring
            <a href="{repo_link}" target="_blank">🏢</a></h4>""",
            unsafe_allow_html=True,
        )

st.title("Real-Time Intersection Intelligence")
st.caption(
    "Edge-based video analytics for real-time traffic monitoring, object tracking, "
    "and road hazard awareness."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Inference", "Technology Stack", "Development Timeline", "Challenges & Lessons"]
)

with tab1:
    with st.sidebar:
        option = st.radio("Select Video Source", ("Local Video", "Live Stream"))
        st.subheader("Select Object Detection Model")
        model_label = st.selectbox("Select a model", list(MODEL_OPTIONS.keys()), index=0)

        show = st.checkbox("Show Real-Time Inference", value=True)
        mqtt = st.checkbox("Use MQTT to send data to server", value=False)
        save = st.checkbox("Save Video after Inference", value=False)
        verbose = st.checkbox("Show logs in Terminal", value=False)
        roi = st.checkbox("Enable ROI", value=True)
        use_trt = st.checkbox("Use TensorRT", value=model_label == "TensorRT FP16 engine")
        only_fps = st.checkbox("Measure FPS Only", value=True)
        half = st.checkbox("Use Half Precision", value=False)
        fep = st.checkbox("Enable FishEye Projection", value=False)

        start_button = st.button("Start", type="primary")
        stop_button = st.button("Stop/Close")

    video_source = None
    if option == "Local Video":
        uploaded_file = st.file_uploader("Upload Video", accept_multiple_files=False, type=["mp4", "avi"])
        if uploaded_file:
            suffix = Path(uploaded_file.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.read())
                st.session_state["uploaded_video_path"] = temp_file.name

        video_source = st.session_state.get("uploaded_video_path")
        if uploaded_file and not video_source:
            st.error("Please upload a video file.")
    else:
        video_source = st.text_input("Enter Video Stream URL")
        if not video_source:
            st.warning("Please enter a stream URL.")

    if start_button:
        col1, col2, col3 = st.columns([1, 2, 1])

        if not video_source:
            st.error("Please provide a video path or stream URL.")
        else:
            st.write("Calling server for processing...")

            payload = {
                "video_source": video_source,
                "model_name": MODEL_OPTIONS[model_label],
                "type": "tracking",
                "show": show,
                "mqtt": mqtt,
                "save": save,
                "verbose": verbose,
                "roi": roi,
                "half": half,
                "fep": fep,
                "bench": False,
                "bench_labels": DEFAULT_BENCH_LABELS,
                "use_TRT": use_trt,
                "plot_perf": False,
                "only_FPS": only_fps,
            }
            logger.debug("UI payload: %s", payload)

            try:
                response = requests.post(f"{BACKEND_URL}/", json=payload, timeout=15)
                response.raise_for_status()
                st.success("Configuration added successfully.")
                st.write(response.json())

                if not show:
                    st.info("Preview is disabled. Processed video will be saved under runs/detect when saving is enabled.")
                else:
                    stframe = st.empty()
                    with requests.get(f"{BACKEND_URL}/video_feed", stream=True, timeout=(5, 60)) as video_stream:
                        buffer = b""
                        for chunk in video_stream.iter_content(chunk_size=65536):
                            if not chunk:
                                continue
                            buffer += chunk
                            while b"--frame\r\n" in buffer:
                                start_buf = buffer.find(b"--frame\r\n")
                                end_buf = buffer.find(b"--frame\r\n", start_buf + 1)
                                if end_buf == -1:
                                    break

                                frame_raw = buffer[start_buf:end_buf]
                                buffer = buffer[end_buf:]
                                headers_end = frame_raw.find(b"\r\n\r\n")
                                if headers_end == -1:
                                    continue
                                image_bytes = frame_raw[headers_end + 4:].strip()
                                if not image_bytes:
                                    continue
                                with col2:
                                    stframe.image(image_bytes, caption="Live inference stream")
            except requests.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.response.json().get("detail", "")
                except Exception:
                    detail = exc.response.text
                st.error(f"Error: {detail or exc}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Connection Error: {exc}")

    if stop_button:
        try:
            response = requests.post(f"{BACKEND_URL}/shutdown", timeout=10)
            response.raise_for_status()
            st.success("Application stopped successfully.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Failed to stop the backend server: {exc}")
        finally:
            _cleanup_uploaded_file()
        st.rerun()

with tab2:
    st.subheader("Technology Stack")
    st.markdown("""
    ### Core Components
    - **Frontend:** Streamlit
    - **Backend API:** FastAPI
    - **Inference Engine:** YOLOv5 / YOLOv8
    - **Accelerated Deployment:** TensorRT, ONNX
    - **Computer Vision:** OpenCV
    - **Messaging:** MQTT
    - **Runtime Environment:** Windows + WSL, Jetson-ready deployment

    ### System Capabilities
    - Real-time object detection
    - Multi-object tracking
    - ROI-based monitoring
    - Performance-oriented inference
    - Stream and file-based video processing
    - Optional MQTT event forwarding
    """)

with tab3:
    st.subheader("Development Timeline")
    st.markdown("""
    ### Project Evolution
    **Phase 1 – Local inference prototype**
    - Initial object detection pipeline on standard desktop hardware
    - Validation of model outputs and frame rendering

    **Phase 2 – Tracking and optimization**
    - Added object tracking and ROI logic
    - Introduced ONNX and TensorRT deployment paths

    **Phase 3 – Web interface**
    - Replaced dependency on local GUI windows with browser-based monitoring
    - Added Streamlit + FastAPI integration for headless execution

    **Phase 4 – Edge deployment preparation**
    - Adapted workflow for Jetson-style execution
    - Added MQTT-based communication and performance measurement options
    """)

with tab4:
    st.subheader("Challenges & Lessons")
    st.markdown("""
    ### Main Challenges Encountered
    - **Headless display issues:** GUI-based display checks were still being triggered in WSL/headless environments.
    - **Model abstraction issues:** Wrapper and backend configuration paths needed better separation.
    - **TensorRT migration complexity:** Runtime behavior differed from `.pt` and ONNX execution.
    - **Streaming pipeline debugging:** Frame transport and frontend rendering required careful synchronization.
    - **Deployment consistency:** Behavior changed between Windows desktop execution and WSL/Jetson-like environments.

    ### Key Lessons
    - Browser rendering is more robust than local GUI rendering for deployment scenarios.
    - Model wrappers should expose a clean, explicit configuration interface.
    - TensorRT integration benefits from isolated testing before pipeline-wide integration.
    - Logging and small validation checkpoints make debugging much easier.
    """)
