import streamlit as st
import requests 
import tempfile 
import os 
import cv2 
import numpy as np  

BACKEND_URL = 'http://127.0.0.1:8000'


#Title 
st.title("Video Processing System")

#Information 
st.markdown(":snowflake: **Video Processing System** is an application that allows you to process any video or live stream using pretrained Yolov5 and Yolov8 models."
            "It can be used for multiple object detection, tracking, and classification in real-time. :snowflake:")

#Input Options 
option = st.radio("Select Video Source", ("Local Video", "Live Stream"))
video_path = None

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


st.subheader("Select Object Detection Model")
model_choice = st.selectbox(
    "Select a model",
    ["Yolov5n (Nano)", "Yolov8n (Nano)", 
     "Yolov5s (Small)", "Yolov8s (Small)", 
     "Yolov5m (Medium)", "Yolov8m (Medium)"]
)


_,col1, col2,_ = st.columns([1,2,2,1])
show = False
mqtt = False 
 
with col1:
    if st.checkbox("Show Real-Time Inference"):
        show = True
        
with col2:
    if st.checkbox("Use MQTT to send data to server"):
       mqtt = True

coll1, coll2, col3, col4= st.columns([2,0.75,1,2])

with coll2:
    start_button = st.button('Start', type='primary')
    
with col3:
    stop_button = st.button("Stop/Close")
    

if start_button: 
    if not video_path: 
        st.error('Please provide a video path or stream URL.')
    else: 
        st.write("Calling Server for processing...")
        try:
            response = requests.post(f"{BACKEND_URL}/",
                json={"video_path":video_path, "name_model": model_choice, "show":show, "mqtt":mqtt}
            )
        except requests.exceptions.ConnectionError as coe: 
            st.error(f"Connection Error: {coe}")

        # st.write(response.json())

        
        if response.status_code == 200:
            st.success("Configuration added successfully!")
            if not show: 
                st.write("Processed video will be save locally in ../video)_processing_system/runs/detect/")
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
                            image_array = cv2.imdecode(
                                np.frombuffer(image_bytes,dtype=np.uint8),
                                cv2.IMREAD_COLOR
                            )
                            stframe.image(image_array, channels="BGR")
                        except Exception as e:
                            print(f"Error decoding frame: {e}")
                            st.error(f"Error decoding frame: {e}")

        else:
            st.error(f"Error: {response.json().get('detail')}")

if stop_button:
    try:
        os.remove(video_path)
        print(f"Temporary file {video_path} deleted successfully.")
    except OSError as e:
        print(f"Error deleting temporary file: {e}. File may not exist.")
    response = requests.post(f"{BACKEND_URL}/shutdown")
    if response.status_code == 200:
        st.success("Application stopped successfully!")
        print(response.json())
    else: 
        st.error("Failed to stop the backend server!")

    st.stop() 



















