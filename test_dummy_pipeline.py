from obs_system.camera_module.dummy_impl.video_consumer import DummyConsumer
from obs_system.detection_module.dummy_predictor.dummy_yolo import DummyPredictor
from obs_system.detection_module.dummy_predictor.stream_yolov5 import Yolov5Streamer
from obs_system.detection_module.dummy_predictor.stream_yolov8 import Yolov8Streamer

from obs_system.logic_module.dummy_logic.overlap_detection import BoundingBoxOverlapDetector
from obs_system.communication_module.mqtt_com.message_transmitter import RealMQTT
from obs_system.conversion_module.dummy_convertor.frame_convertor import FrameConvertor
# from obs_system.detection_module.interface.config import Camera360Config
from obs_system.application_module.dummy_application.onnx_workflow.dummy_compression.compression import Compression
# from obs_system.application_module.dummy_application.dummy_app import MainWindow
from ultralytics.utils import DEFAULT_CFG
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #suppress warnings. 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import cv2
import multiprocessing
import time 
import numpy as np
import warnings
import psutil 
import pynvml
import sys
import argparse


def statistics(process_memory, handle, initial_time): 
     end_time = time.time()
     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
     total_gpu_mem = mem_info.total / (1024 ** 2)  # Convert bytes to MB
     used_gpu_mem = mem_info.used / (1024 ** 2)
     free_gpu_mem = mem_info.free / (1024 ** 2)
     print(f"-------------------------------------")
     print(f"|  Total Inference time: {end_time - initial_time} seconds ")
     print(f"|  Current RAM usage: {process_memory.memory_info().rss / (1024**2)} MB.")
     print(f"|  Total GPU memory: {total_gpu_mem:.2f} MB")
     print(f"|  Used GPU memory: {used_gpu_mem:.2f} MB")
     print(f"|  Free GPU memory: {free_gpu_mem:.2f} MB")
     print(f"-------------------------------------")


def main():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--model', metavar='M', default='Yolo', help='Model to use (Yolo, MaskRCNN, ONNX)')
    argparser.add_argument('--op', metavar='O', default='streaming', help='Operation to perform (streaming, inference, compression, detection)')

    if len(sys.argv) < 1:
         argparser.print_help()
         return
    
    args = argparser.parse_args()

    parent_path = os.getcwd()
    test_video_path = os.path.join(parent_path, 'samples/10_DrivingWith.mp4')
    converted_video_path = os.path.join(parent_path, 'converted_mp4/converted_video_1.mp4')
    data = cv2.VideoCapture(test_video_path)
    length_of_film = data.get(cv2.CAP_PROP_FRAME_COUNT)

    #GPU usage 
    process_memory = psutil.Process(os.getpid())
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    #Create Multiprocess pipeline
    parent_conn, child_conn = multiprocessing.Pipe()

    #Create Video Feed
    camera_id = 0 #for later when we have to connect to camera.
    
    #Create the data acquisition
    consumer = DummyConsumer(child_conn, camera_id, test_video_path)

    #Creates the main process.
    process = multiprocessing.Process(target=consumer.run) 
    process.start()
    start_time = time.time()

    #Get the detection model 
    config = args.op
    model_name = args.model

    if model_name == "yolov5" or model_name=="YOLO5" or model_name=="Yolo5": 
        model_weights = os.path.join(parent_path,'yolov5s.pt')

        if config == "streaming":
             e = Yolov5Streamer()
             e.predict_cli(source=test_video_path, model=model_weights)
             statistics(process_memory, handle, start_time) 
             #Close main process
             process.terminate()
             process.join()

             #Close nvidia GPU process 
             pynvml.nvmlShutdown()
             exit(0)

    elif model_name == "yolov8" or model_name == "Yolo8" or model_name == "YOLO8":
        model_weights = 'yolov8n.pt'
        if config == "streaming":
             
            e = Yolov8Streamer(DEFAULT_CFG, {}, None)
            e.predict_cli(source=test_video_path, model=model_weights)
            statistics(process_memory, handle, start_time) 
            #Close main process
            process.terminate()
            process.join()
            #Close nvidia GPU process 
            pynvml.nvmlShutdown()
            exit(0)



    elif model_name == "maskrcnn" or model_name == "MaskRCNN":   
        model_weights = os.path.join(parent_path,'obs_system/Mask_RCNN/mask_rcnn_coco.h5')

    elif model_name == "onnx" or model_name == "ONNX": 
        model_weights = os.path.join(parent_path,'obs_system/application_module/dummy_application/onnx_workflow/dummy_compression/yolov5s.onnx')
    
    model_predictor = DummyPredictor(model_path=model_weights, model_name=model_name)

    #Overlapping detector 
    logic_module = BoundingBoxOverlapDetector()
    
    #MQTT instance
    # mqtt_topic = "test/topic"
    # mqtt_host = "mqtt.eclipse.org" #broker
    # mqtt_port = 1883 
    # mqtt = RealMQTT(mqtt_host)
    # keepalive = 60
    # mqtt.client.connect(mqtt_port,keepalive)
    # mqtt.client.loop_start()

    # main_window = MainWindow(model_predictor)
    # main_window.show()
    frame_counter = 0
    try:
        while True:
            if parent_conn.poll():
                try:
                    if frame_counter > length_of_film: 
                                break
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
                            
                            # Show image
                            cv2.imshow("Overlap frame", overlap_frame)
                            cv2.waitKey(1)
                            if frame_counter <= 20: 
                                directory = r'obs_system/converted_mp4/det/images/'
                                os.chdir(directory)
                                filename = f"obs_frame_{frame_counter}.png"
                                cv2.imwrite(filename,full_image)
                            
                            # Mqtt client sends the data 
                            # mqtt.client.publish(mqtt_topic, str(results))

                            frame_counter += 1 

                        else:
                            print("Received None frame")

                except Exception as e:
                    print("Error while receiving/processing frame:", e)
        raise KeyboardInterrupt("Mannually triggered exception")
    except KeyboardInterrupt as e:
        #Display GPU usage after initial configuration
        print(f"Exception caught: {e}")
        print("Terminating process...")
         
        statistics(process_memory, handle, start_time) 
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
    

