from obs_system.camera_module.dummy_impl.video_consumer import DummyConsumer
from obs_system.detection_module.dummy_predictor.dummy_yolo import DummyPredictor
from obs_system.logic_module.dummy_logic.overlap_detection import BoundingBoxOverlapDetector
from obs_system.communication_module.dummy_com.message_transmitter import DummyMQTT
from obs_system.conversion_module.dummy_convertor.frame_convertor import FrameConvertor
from obs_system.detection_module.interface.config import Camera360Config
from obs_system.application_module.dummy_application.dummy_app import MainWindow
# from PyQt5.QtWidgets import QApplication
import cv2
import multiprocessing
import json
import time 
import numpy as np
import os
import sys
import psutil 
import pynvml

#PYTHONPATH=/home/user/my_project python obs_system/camera_module/some_script.py
def main():
    # app = QApplication(sys.argv)
    process_memory = psutil.Process(os.getpid())
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    parent_conn, child_conn = multiprocessing.Pipe()
    video_path = "samples/10_DrivingWith.mp4"
    converted_video_path = "converted_mp4/converted_video_1.mp4"
    camera_id = 0
    consumer = DummyConsumer(child_conn, camera_id, video_path)
    process = multiprocessing.Process(target=consumer.run) #Creates the single process. 
    process.start()
    config = Camera360Config()
    config.display()    
    weights_path = "/home/icsa_jim/Jupyter Notebook/EDGEAI_OBS_ROI/Obstacle_Recognition_Edge_Ai-main/obs_system/Mask_RCNN/mask_rcnn_coco.h5"
    model_name = 'Yolo'
    model_predictor = DummyPredictor(config,"/obs_system/Mask_RCNN/logs",weights_path, model_name)
    logic_module = BoundingBoxOverlapDetector()
    mqtt_topic = "test_topic"
    mqtt_host = "123.123.123.123"
    mqtt_com = DummyMQTT(mqtt_host)
    
    mem_info = process_memory.memory_info().rss / (1024**2)
    print(f"-----------------------------------------------")
    print(f"|  Current memory usage: {mem_info:.2f} MB             |")
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_gpu_mem = mem_info.total / (1024 ** 2)  # Convert bytes to MB
    used_gpu_mem = mem_info.used / (1024 ** 2)
    free_gpu_mem = mem_info.free / (1024 ** 2)
    print(f"|  Total GPU memory: {total_gpu_mem:.2f} MB                |")
    print(f"|  Used GPU memory: {used_gpu_mem:.2f} MB                 |")
    print(f"|  Free GPU memory: {free_gpu_mem:.2f} MB                 |")
    print(f"-----------------------------------------------")

    # main_window = MainWindow(model_predictor)
    # main_window.show()

    try:
        while True:
            if parent_conn.poll():
                try:
                    shape, dtype, data = parent_conn.recv() #Gets the data from the video feed. 
                    frame = np.frombuffer(data, dtype=dtype)

                    frame = frame.copy()
                    frame = frame.reshape(shape)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

                    convertor = FrameConvertor(
                        frame=frame,
                        shape=shape,
                        model=None,
                        output_path=converted_video_path, 
                        model_name=model_name)

                    patches, positions = convertor.decomposition()
                    num_patches = len(patches)
                    
                    if patches is not None: 
                        # predictions = main_window.start_detection(patches)
                        predictions = model_predictor.concurrent_prediction(patches)
                        # full_image = convertor.reconstruct_image(patches, positions)
                        stride = convertor.get_stride()
                        # model_predictor.clear_gpu_memory()
                        predictions = model_predictor.draw_boxes(frame, patches, positions, predictions, shape, stride)
                        cv2.imshow('Detections on large image', frame)
                        cv2.waitKey(0)

                        time.sleep(100)
                        result = logic_module.detect(predictions)
                        overlap_frame = logic_module.draw_overlaps(frame.copy(), result)

                        cv2.imshow("Overlap frame", overlap_frame)
                        cv2.waitKey(0)
                        mqtt_com.publish(mqtt_topic, str(result))
                    else:
                        print("Received None frame")

                    # if frame is not None:                         
                    #     cv2.imshow("frame", frame)
                    #     cv2.waitKey(0)

                        

                    #     #Process Frames Method module
                    #     time.sleep(90)

                    #     #Model prediction
                    #     predictions = model_predictor.predict(frame)
                    #     detect_frame = model_predictor.draw_boxes(frame.copy(), predictions)
                    #     cv2.imshow("Detection frame", detect_frame)
                    #     cv2.waitKey(1) 

                    #     #Logic module
                    #     result = logic_module.detect(predictions)
                    #     overlap_frame = logic_module.draw_overlaps(frame.copy(), result)
                    #     cv2.imshow("Overlap frame", overlap_frame)
                    #     cv2.waitKey(1)
             
                    #     #Communication module
                    #     mqtt_com.publish(mqtt_topic, str(result))

                    # else:
                    #     print("Received None frame")
                except Exception as e:
                    print("Error while receiving/processing frame:", e)

    except KeyboardInterrupt:
        process.terminate()
        process.join()
        pynvml.nvmlShutdown()
        # sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    

