from obs_system.utils.logger import get_logger 

import os 
import cv2 
import sys 
import time 
import requests 
import tempfile 
import warnings 
import subprocess    
import numpy as np  
import streamlit as st

from PIL import Image 
from typing import Text
from pathlib import Path

logger = get_logger("obs_system."+__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # Moves up to `mypackage/`
BACKEND_URL = f"http://{st.get_option('server.address')}:8000"
static_folder =  PACKAGE_ROOT / "static"
logo_image = static_folder / "logo.png" 
HOST = st.get_option('server.address')
PORT = st.get_option('server.port')

st.set_page_config(
    page_title="EDGEAI-VPS", 
    page_icon=logo_image,
    layout="wide"
)

#Title 
# st.title("Video Processing System")
#Information 
st.markdown(":snowflake: **Video Processing System** is an application that allows you to process any video or live stream using pretrained Yolov5 and Yolov8 models."
            "Used for multiple object detection, tracking, and classification in real-time. :snowflake:")

js = """
<script>
window.addEventListener("beforeunload", async function() {
    fetch("""f'{BACKEND_URL}/shutdown'""", { method: 'POST' });
});
</script>
"""
st.markdown(js, unsafe_allow_html=True)

show = False
mqtt = False 
save = False 
verbose = False 

with st.sidebar:
    logo_image = Image.open(logo_image)
    icon, title = st.columns([0.4, 0.63])

    with icon: 
        st.image(logo_image, width=100)

    with title: 
        repo_link: Text = ("https://edge-ai-tech.eu/")
        st.markdown(f"""<h4 style='color: #f0eef0;'>Real-Time Intersection Monitoring<a href="{repo_link}" target="_blank">🏢</a></h2>""", unsafe_allow_html=True)        

    #Input Options 
    option = st.radio("Select Video Source", ("Local Video", "Live Stream"))
    video_path = None
    
    st.subheader("Select Object Detection Model")
    model_choice = st.selectbox(
        "Select a model",
        ["Yolov5n", "Yolov8n", 
        "Yolov5s", "Yolov8s", 
        "Onnx (Yolov8s)"]
    )

    if st.checkbox("Show Real-Time Inference"):
        show = True

    if st.checkbox("Use MQTT to send data to server"):
       mqtt = True
    
    if st.checkbox("Save Video after Inference"):
       save = True
    
    if st.checkbox("Show logs in Terminal"):
       verbose = True

    start_button = st.button('Start', type='primary')
    stop_button = st.button("Stop/Close")
    
if option == "Local Video":
    uploaded_file = st.file_uploader("Upload Video",accept_multiple_files=False,type=["mp4", "avi"])
    if uploaded_file: 
        # video_path = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        video_path = temp_path 
        if not video_path:
            st.error("Please upload a video file.")

elif option == "Live Stream":
    video_path = st.text_input("Enter Video Stream URL")
    if not video_path:
        st.warning("Please enter a stream URL.")

if start_button: 
    col1, col2, col3 = st.columns([1, 2, 1])  
    if not video_path: 
        st.error('Please provide a video path or stream URL.')
    else: 
        st.write("Calling Server for processing...")
        logger.debug(f"video_path:{video_path}, name_model: {model_choice}, show:{show}, mqtt:{mqtt}, save:{save}, verbose:{verbose}")

        try:
            response = requests.post(f"{BACKEND_URL}/",
                json={"video_path":video_path, "name_model": model_choice, "show":show, "mqtt":mqtt, 'save':save, 'verbose': verbose}
            )

            if response.status_code == 200:
                st.success("Configuration added successfully!")
                if not show: 
                    st.write("Processed video will be save locally in ../video_processing_system/runs/detect/")
                stframe = st.empty()
                #Stream Frames 
                with requests.get(f"{BACKEND_URL}/video_feed", stream=True) as video_stream: 
                    
                    buffer = b""
                    for chunk in video_stream.iter_content(chunk_size=1024): 
                        buffer += chunk 
                        while b"--frame\r\n" in buffer:
                            #Find the boundary 
                            start_buf = buffer.find(b"--frame\r\n")
                            end_buf = buffer.find(b"--frame\r\n", start_buf+1)

                            if end_buf == -1:
                                break
                            # Extract the raw image data 
                            frame_raw = buffer[start_buf:end_buf]
                            buffer = buffer[end_buf:]
                            # Extract JPEG bytes after headers 
                            try: 
                                headers_end = frame_raw.find(b"\r\n\r\n") + 4
                                image_bytes = frame_raw[headers_end:]
                                if not image_bytes:
                                    continue
                                image_array = cv2.imdecode(
                                    np.frombuffer(image_bytes,dtype=np.uint8),
                                    cv2.IMREAD_COLOR
                                )
                                if image_array is None:
                                    raise ValueError("Failed to decode image.")
                                with col2:                                
                                    stframe.image(image_array, channels="BGR")
                            except Exception as e:
                                st.error(f"Error decoding frame: {e}")

            else:
                st.error(f"Error: {response.json().get('detail')}")
                sys.exit(1)

            st.write(response.json())

        except requests.exceptions.ConnectionError as coe: 
            st.error(f"Connection Error: {coe}")


if stop_button:

    try:
        os.remove(video_path)
    except OSError as e:
        warnings.warn(f"Error deleting temporary file: {e}. File may not exist.")
    
    response = requests.post(f"{BACKEND_URL}/shutdown")
    
    if response.status_code == 200:
        st.success("Application stopped successfully!")
        time.sleep(2) 
        subprocess.run(["streamlit", "run", 'obs_system/application_module/dummy_application/web_interface.py', f'--server.port={str(PORT)}', f"--server.address={str(HOST)}"])

    else: 
        st.error("Failed to stop the backend server!")

    st.stop() 



















