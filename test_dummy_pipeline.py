from obs_system.camera_module.dummy_impl.video_consumer import DummyConsumer
from obs_system.detection_module.dummy_predictor.dummy_yolo import DummyPredictor
from obs_system.logic_module.dummy_logic.overlap_detection import BoundingBoxOverlapDetector
from obs_system.communication_module.mqtt_com.message_transmitter import RealMQTT
from obs_system.conversion_module.dummy_convertor.frame_convertor import FrameConvertor
from obs_system.detection_module.interface.config import Camera360Config
from obs_system.application_module.dummy_application.onnx_workflow.dummy_compression.compression import Compression
# from obs_system.application_module.dummy_application.dummy_app import MainWindow

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import cv2
import multiprocessing
import json
import time 
import numpy as np
import warnings
import sys
import psutil 
import pynvml

def main():

    #GPU usage 
    process_memory = psutil.Process(os.getpid())
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    #Create Multiprocess pipeline
    parent_conn, child_conn = multiprocessing.Pipe()

    #Create Video Feed
    camera_id = 0 #for later when we have to connect to camera.
    video_path = "samples/10_DrivingWith.mp4"
    converted_video_path = "converted_mp4/converted_video_1.mp4"
    
    #Create the data acquisition
    consumer = DummyConsumer(child_conn, camera_id, video_path)

    #Creates the main process.
    process = multiprocessing.Process(target=consumer.run) 
    process.start()
    start_time = time.time()
    #Get the detection model 
    model_name = 'Yolo'
    if model_name == "Yolo" or model_name=="YOLO": 
        model_weights = "/home/icsa_jim/Jupyter Notebook/EDGEAI_OBS_ROI/Obstacle_Recognition_Edge_Ai/yolov5s.pt"    
    elif model_name == "maskrcnn" or model_name == "MaskRCNN":   
        model_weights = "/home/icsa_jim/Jupyter Notebook/EDGEAI_OBS_ROI/Obstacle_Recognition_Edge_Ai-main/obs_system/Mask_RCNN/mask_rcnn_coco.h5"
    elif model_name == "onnx" or model_name == "ONNX": 
        model_weights = "/home/icsa_jim/Jupyter Notebook/EDGEAI_OBS_ROI/Obstacle_Recognition_Edge_Ai-main/yolov5s.onnx"

    model_predictor = DummyPredictor(model_path=model_weights, model_name=model_name)

    #Overlapping detector 
    logic_module = BoundingBoxOverlapDetector()
    
    #MQTT instance
    mqtt_topic = "test/topic"
    mqtt_host = "mqtt.eclipse.org" #broker
    mqtt_port = 1883 
    mqtt = RealMQTT(mqtt_host)
    keepalive = 60
    # mqtt.client.connect(mqtt_port,keepalive)
    # mqtt.client.loop_start()

    # main_window = MainWindow(model_predictor)
    # main_window.show()
    frame_counter = 0
    try:
        while True:
            if parent_conn.poll():
                try:
                    #Get frame from feed (child process)
                    shape, dtype, data = parent_conn.recv()  
                    
                    #Construct the frame from the buffer and get the RGB image.
                    frame = np.frombuffer(data, dtype=dtype)
                    frame = frame.copy()
                    frame = frame.reshape(shape)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

                    #Convertor used to decompose and reconstruct the 360 image. 
                    convertor = FrameConvertor(
                        frame=frame,
                        shape=shape,
                        output_path=converted_video_path, 
                        save_conversion=False,
                        model_name=model_name)

                    #Decompose the 360 image into identical dimension patches. 
                    patches, positions = convertor.decomposition()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                    
                        #Process the patches simulaneoudly.                    
                        if patches is not None: 

                            #Concurrent predictions (multithreading inference on patches) 
                            predictions = model_predictor.concurrent_prediction(patches)

                            #Reconstruct the 360 frame
                            full_image = convertor.reconstruct_image(patches, positions)
                            stride = convertor.get_stride()

                            #Visualization of the predictions from the detection model 
                            predictions = model_predictor.draw_boxes(full_image, patches, positions, predictions, shape, stride)
                            full_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR) 

                            #Show image
                            # cv2.imshow('Detections on large image', full_image)
                            # cv2.waitKey(100)
                            
                            #Detect overlapping boxes
                            results = logic_module.detect(predictions)
                            overlap_frame = logic_module.draw_overlaps(full_image, np.asarray(results))
                            
                            #Show image
                            # cv2.imshow("Overlap frame", overlap_frame)
                            # cv2.waitKey(1)
                            # if frame_counter <= 20: 
                            #     directory = r'/home/icsa_jim/Jupyter Notebook/EDGEAI_OBS_ROI/Obstacle_Recognition_Edge_Ai-main/converted_mp4/det/images/'
                            #     os.chdir(directory)
                            #     filename = f"obs_frame_{frame_counter}.png"
                            #     cv2.imwrite(filename,full_image)
                            
                            # Mqtt client sends the data 
                            # mqtt.client.publish(mqtt_topic, str(results))

                            frame_counter += 1 
                        else:
                            print("Received None frame")

                except Exception as e:
                    print("Error while receiving/processing frame:", e)

    except KeyboardInterrupt:
        #Display GPU usage after initial configuration
        end_time = time.time()  
        mem_info1 = process_memory.memory_info().rss / (1024**2)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gpu_mem = mem_info.total / (1024 ** 2)  # Convert bytes to MB
        used_gpu_mem = mem_info.used / (1024 ** 2)
        free_gpu_mem = mem_info.free / (1024 ** 2)
        print(f"-------------------------------------")
        print(f"|  Total Inference time: {end_time - start_time} seconds ")
        print(f"|  Current RAM usage: {mem_info1:.2f} MB    ")
        print(f"|  Total GPU memory: {total_gpu_mem:.2f} MB     ")
        print(f"|  Used GPU memory: {used_gpu_mem:.2f} MB       ")
        print(f"|  Free GPU memory: {free_gpu_mem:.2f} MB      ")
        print(f"-------------------------------------")
       
        #Close MQTT connection with server
        # mqtt.client.loop_stop()
        # mqtt.client.disconnect()

        #Close main process
        process.terminate()
        process.join()

        #Close nvidia GPU process 
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
    

