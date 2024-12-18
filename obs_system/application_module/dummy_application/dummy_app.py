from obs_system.logic_module.dummy_logic.overlap_detection import BoundingBoxOverlapDetector
from obs_system.detection_module.dummy_predictor.stream_yolov5 import Yolov5Streamer
from obs_system.detection_module.dummy_predictor.stream_yolov8 import Yolov8Streamer
from obs_system.communication_module.mqtt_com.message_transmitter import RealMQTT
from ultralytics.utils import DEFAULT_CFG

import numpy as np 
import warnings
import os 
import psutil 
import pynvml
import tracemalloc 
import time 
import multiprocessing
import pdb

class Application: 
   
    def __init__(self):
        self.object_detector = None
        self.frame_counter = 0 
        self.model = None
        self.parent_path =  os.getcwd()
        self.model_name = None
        self.show = False 
        self.mqtt = False


    def get_streaming_detector(self, model_name):
        
        set_object_detector_func = {
            "yolo":self.yolov5_streaming,
            "yolo5":self.yolov5_streaming,
            "yolov5":self.yolov5_streaming,
            "yolov5s":self.yolov5_streaming,
            "yolov5n":self.yolov5_streaming,
            "yolov5m":self.yolov5_streaming,
            "yolo8":self.yolov8_streaming,
            "yolov8":self.yolov8_streaming,
            "yolov8s":self.yolov8_streaming,
            "yolov8n":self.yolov8_streaming,
            "yolov8m":self.yolov8_streaming
        }

        return set_object_detector_func.get(model_name, lambda *args:None)
    

    def yolov5_streaming(self, opt):
        if self.model_name != "yolov5n" and self.model_name != "yolov5s":
            model_weights = "yolov5n" + ".pt"
        else:
            model_weights = self.model_name + ".pt"
        
        self.streamer = Yolov5Streamer(DEFAULT_CFG, {}, None)
        self.streamer.setup_model(model=model_weights, verbose=False, opt=opt)
        self.model = self.streamer.model


    def yolov8_streaming(self, opt):
        if self.model_name != "yolov8n" and self.model_name != "yolov8s":
            model_weights = "yolov8n" + ".pt"
        else:
            model_weights = self.model_name + ".pt"
        self.streamer = Yolov8Streamer(DEFAULT_CFG, {}, None)
        self.streamer.setup_model(model=model_weights, verbose=False, opt=opt)
        self.model = self.streamer.model


    def setup_process(self, source, args): 
        self.process_memory = psutil.Process(os.getpid())
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.source = os.path.join(self.parent_path, source) if os.path.isfile(source) else source
        DEFAULT_CFG.show = args.show if args.show is not None else False
        DEFAULT_CFG.verbose = args.verbose if args.verbose is not None else False
        self.mqtt = args.mqtt if args.mqtt is not None else False 
        self.start_time = time.time()

        tracemalloc.start()
        return 


    def setup_model(self, model_name, stream, opt="tracking"):
        
        self.stream = stream
        self.model_name = model_name        
        setup_func = self.get_streaming_detector(model_name)
        return setup_func(opt=opt)
        

    def setup_logic_module(self): 
        self.logic_module = BoundingBoxOverlapDetector()


    def run_app(self, output_path, save, model, length_of_film=0): 
        process_video_func = self.get_process_video()
        return process_video_func(model=model)


    def get_process_video(self): 
        return  self.process_stream
          

    def process_stream(self, model):
        self.statistics()
        if isinstance(self.source, str):
            # length_of_film = self.get_length_of_film(self.source)
            #Streaming the video as before             
            self.streamer(source=self.source, model=model, stream=self.stream, mqtt_broker=self.mqtt_interface) # e(source=0, model=model_weights, stream=True)                
            return self.streamer.results

        raise ValueError("Only string is supported as source")
    

    def setup_mqtt(self, topic, broker_address, port):
        if not self.mqtt: 
            self.mqtt_interface = None 
            return
        self.mqtt_topic = topic
        self.mqtt_interface = RealMQTT(broker_address, self.mqtt_topic)
        self.mqtt_interface.connect(port=port, keepalive=60)
        self.mqtt_interface.client.loop_start() #Not loop.forever as main thread will be taken over for the MQTT process. 


    def publish_mqtt(self, message):
        self.mqtt_interface.publish(topic=self.mqtt_topic, message=message)


    def statistics(self):
        end_time = time.time()
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        total_gpu_mem = mem_info.total / (1024 ** 2)  # Convert bytes to MB
        used_gpu_mem = mem_info.used / (1024 ** 2)
        free_gpu_mem = mem_info.free / (1024 ** 2)
        current, peak = tracemalloc.get_traced_memory()
        print(f"-------------------------------------")
        print(f"|  Total Inference time: {end_time - self.start_time} seconds ")
        print(f"|  Current Environment RAM usage (Psutil): {self.process_memory.memory_info().rss / (1024**2)} MB.")
        print(f"|  Current memory usage (Tracemalloc): {current / (1024 ** 2):.2f} MB.")
        print(f"|  Peak memory usage (Tracemalloc): {peak / (1024 ** 2):.2f} MB.")
        print(f"|  Total GPU memory: {total_gpu_mem:.2f} MB")
        print(f"|  Used GPU memory: {used_gpu_mem:.2f} MB")
        print(f"|  Free GPU memory: {free_gpu_mem:.2f} MB")
        print(f"-------------------------------------")


    def close_app(self): 

        #Display GPU usage after execution
        self.statistics()

        # #Close MQTT connection with server
        if self.mqtt_interface is not None:
            self.mqtt_interface.client.loop_stop()
            self.mqtt_interface.client.disconnect()


        pynvml.nvmlShutdown()


