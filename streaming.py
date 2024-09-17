import cv2
import numpy as np 
import warnings
import os 
import psutil 
import pynvml
import time 
import multiprocessing
import pdb
from obs_system.camera_module.dummy_impl.video_consumer import DummyConsumer


process_memory = psutil.Process(os.getpid())
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

parent_conn, child_conn = multiprocessing.Pipe()    
source = '/home/icsa_jim/Jupyter Notebook/EDGEAI_OBS_ROI/Obstacle_Recognition_Edge_Ai-main/samples/10_DrivingWith.mp4'
#Create the data acquisition from video source 
consumer = DummyConsumer(child_conn, source)
process = multiprocessing.Process(target=consumer.run, args=())
process.start()    
start_time = time.time()
# Open the RTSP stream
cap = cv2.VideoCapture(source)
cv2.namedWindow("Name Window", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame (e.g., YOLO inference)
    # Your YOLO inference code here
    # Display the frame
    cv2.imshow("YOLO Inference", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
