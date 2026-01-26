from obs_system.detection_module.dummy_predictor.stream_yolov5 import Yolov5Streamer 
from obs_system.detection_module.dummy_predictor.stream_yolov8 import Yolov8Streamer 
from obs_system.detection_module.dummy_predictor.stream_y8_onnx import OnnxY8Streamer
from obs_system.detection_module.dummy_predictor.stream_trt import TensorRTRTXStreamer
from obs_system.communication_module.mqtt_com.message_transmitter import RealMQTT
from obs_system.logic_module.dummy_logic.region_setter import RegionSetter
from obs_system.logic_module.dummy_logic.subtractor import Subtractor
from obs_system.logic_module.dummy_logic.fisheye import FishEyeProjection
from obs_system.utils.common import check_nvidia_existence
from obs_system.utils.global_config import EMPTY_IMAGE_PATH
from obs_system.utils.logger import get_logger 
from obs_system.utils.appraisal import perf, frame_list 

import os 
import pdb
import numpy as np
import psutil 
import pynvml
import time 
import tracemalloc 

from pathlib import Path
from typing import Any, Optional
from collections import defaultdict
from ultralytics.utils import DEFAULT_CFG

logger = get_logger("obs_system."+__name__)

class Application: 

    """
    This class is responsible for the application logic. It creates the detection class instances, 
    initiates the broker, and starts the detection process. It also provides metrics that show
    the hardware utilization and the memory usage.
    """

    def __init__(self, save:bool=False, verbose:bool=False):

        self.source:str = ""
        self.save_outputs = save
        self.verbose_outputs = verbose 
        self.use_TRT=None
        self.model = None
        self.parent_path:str =  os.getcwd()
        self.mqtt:bool = False
        self.streamer:Any = None 
        self.mqtt_interface:Any = None 
        self.logic_module = defaultdict() 
        self.gpu_enabled:bool = False 


    def get_streaming_detector(self, model_name:str):
        
        # Associate the model from the model key to the corresponding streaming function. 
        model_name_ending = model_name.split('.')[-1]
        # set_object_detector_func = {
        #     "yolov5s": self.yolov5_streaming,
        #     "yolov5n": self.yolov5_streaming,
        #     "yolov5m": self.yolov5_streaming,
        #     "yolov8s": self.yolov8_streaming,
        #     "yolov8n": self.yolov8_streaming,
        #     "yolov8m": self.yolov8_streaming, 
        #     "onnx":    self.onnx_streaming, 
        #     "compressed": self.onnx_streaming, 
        #     "yolov8.onnx": self.onnx_streaming, 
        #     "trt": self.trt_streaming,
        #     "engine":self.trt_streaming
        # }

        set_object_detector_func = {
            "onnx":self.onnx_streaming if not self.use_TRT else self.trt_streaming,
            "engine": self.trt_streaming, 
            "pt": self.yolov8_streaming
        }

        return set_object_detector_func.get(model_name_ending, lambda *args:None)
     

    # def yolov5_streaming(self, model_name, path_to_load:Optional[str|Path]):
    #
    #     # Check the model version. If the perscribed model is not lower than the medium version then default to the nano version. 
    #     self.streamer = Yolov5Streamer(DEFAULT_CFG, {}, None)
    #     self.streamer.setup_model(model=model_name, path_to_load=path_to_load)
    #     self.model = self.streamer.model
    #     
    #     logger.debug(f"-- Streaming Through {model_name} found in {path_to_load}. --")


    def yolov8_streaming(self, model_name, path_to_load:Optional[str|Path]):

        # Check the model version. If the perscribed model is not lower than the medium version then default to the nano version. 
        self.streamer = Yolov8Streamer(DEFAULT_CFG, {}, None)
        self.streamer.setup_model(model_name=model_name, path_to_load=path_to_load)
        self.model = self.streamer.model

        logger.debug(f"-- Streaming Through {model_name} found in {path_to_load}. --")

    
    def onnx_streaming(self,model_name, path_to_load:Optional[str|Path]):
        self.streamer = OnnxY8Streamer(DEFAULT_CFG, {}, None) 
        self.streamer.setup_model(model_name=model_name, path_to_load=path_to_load) 
        self.model = self.streamer.model 

        logger.debug(f"-- Streaming Through {model_name} found in {path_to_load}. --")


    def trt_streaming(self,model_name, path_to_load:Optional[str|Path]): 
        self.streamer = TensorRTRTXStreamer(DEFAULT_CFG, {}, None)
        self.streamer.setup_model(model_name=model_name, path_to_load=path_to_load)
        self.model = self.streamer.model

        logger.debug(f"-- Streaming Through {model_name} found in {path_to_load}. --")


    def setup_process(self, args): 

        # Check for GPU (NVIDIA) to allow the program to run GPU statistics 
        self.gpu_enabled = check_nvidia_existence()

        if self.gpu_enabled:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        self.process_memory = psutil.Process(os.getpid())
        self.source = os.path.join(self.parent_path, args.video_source) if args.video_source else self.source
        self.use_TRT = args.use_TRT if args.use_TRT is not None else False
        self.mqtt = args.mqtt if args.mqtt is not None else False 
        self.start_time = time.time()

        DEFAULT_CFG.show = args.show if args.show is not None else False
        DEFAULT_CFG.gui = args.gui if args.gui is not None else False
        DEFAULT_CFG.save = self.save_outputs
        DEFAULT_CFG.verbose = self.verbose_outputs
        DEFAULT_CFG.half = args.half
        DEFAULT_CFG.bench = args.bench 
        DEFAULT_CFG.bench_labels = args.bench_labels
        DEFAULT_CFG.roi = args.roi if args.roi is not None else False
        DEFAULT_CFG.plot_performance = args.plot_perf if args.plot_perf is not None else False 
        DEFAULT_CFG.only_FPS = args.only_FPS if args.only_FPS is not None else False
        
        tracemalloc.start()


    def setup_model(self,
                    model_name:str="yolov8s.onnx",
                    path_to_load:Optional[str|Path]="assets/compressed_models",
                    opt:str="tracking"):

        if path_to_load is None: 
            raise TypeError("path_to_load cannot be None")

        if not model_name.endswith('.pt') and  not model_name.endswith('.onnx') and model_name.endswith('.engine'): 
            raise TypeError("The model is imperative to be either .pt, .onnx or .engine format.")

        if path_to_load=="":
            if not os.path.exists(model_name):
                raise FileNotFoundError("model_name was passed as the path, and it does not exist") 
            else: 
                path_to_load = model_name
        else: 
            if not os.path.exists(path_to_load): 
                raise FileNotFoundError(path_to_load)

        setup_func = self.get_streaming_detector(model_name)
        setup_func(model_name,path_to_load)
        
 
    def setup_logic_module(self, args): 

        # Change this based on your video. Get the first frame. 
        self.logic_module["SUBTRACTOR"] = Subtractor()
        self.logic_module["ROI"] = RegionSetter() 
        
        if args.fep: 
            self.logic_module["FEP"] = FishEyeProjection(crop=0.00)
        else: 
            self.logic_module["FEP"] = None


    def run_app(self, producer_flag=None, queue=None): 
        self.statistics()

        process_video_func = self.process_stream
        process_video_func(producer_flag=producer_flag, queue=queue)
        perf.finalize() 
        stats = perf.results() 

        # add FPS & percentiles for total frame time
        ft = np.array(frame_list, dtype=np.float32)
        stats.update({
            "frame_p50_ms": float(np.percentile(ft, 50)) if ft.size else 0.0,
            "frame_p95_ms": float(np.percentile(ft, 95)) if ft.size else 0.0,
            "frame_p99_ms": float(np.percentile(ft, 99)) if ft.size else 0.0,
            "FPS_mean": (1000.0 / float(ft.mean())) if ft.size else 0.0,
        })

        logger.debug(stats)

        return 


    def process_stream(self, producer_flag=None, queue=None):
        logger.debug("-- Starting the video streaming process --")

        # Streaming the video as before
        kwargs = {
            "save":self.save_outputs,
            "verbose":self.verbose_outputs
        }

        self.streamer(
            source=self.source,
            model=self.model,
            logic_module=self.logic_module,
            mqtt_broker=self.mqtt_interface,
            producer_flag=producer_flag, 
            queue=queue, 
            **{key: kwargs[key] for key in ['verbose', 'save']}
        ) 
                            
        return self.streamer.results
        

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
        logger.debug("-- Performance metrics --")

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        
        logger.info(f"------------------------------------------------------------------------------------")
        logger.info(f"|  Total Inference time: {end_time - self.start_time} seconds ")
        logger.info(f"|  Current Environment RAM usage (Psutil): {self.process_memory.memory_info().rss / (1024**2)} MB.")
        logger.info(f"|  Current memory usage (Tracemalloc): {current / (1024 ** 2):.2f} MB.")
        logger.info(f"|  Peak memory usage (Tracemalloc): {peak / (1024 ** 2):.2f} MB.")
       
        if self.gpu_enabled:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        
            # Convert bytes to MB
            total_gpu_mem = int(mem_info.total) / (1024 ** 2)  
            used_gpu_mem = int(mem_info.used) / (1024 ** 2)
            free_gpu_mem = int(mem_info.free) / (1024 ** 2)
        
            logger.info(f"|  Total GPU memory: {total_gpu_mem:.2f} MB")
            logger.info(f"|  Used GPU memory: {used_gpu_mem:.2f} MB")
            logger.info(f"|  Free GPU memory: {free_gpu_mem:.2f} MB")

        logger.info(f"------------------------------------------------------------------------------------")


    def close_app(self): 
        logger.debug("-- Terminating the application --")
        
        #Display GPU usage after execution
        self.statistics()

        # #Close MQTT connection with server
        if self.mqtt_interface is not None:
            self.mqtt_interface.client.loop_stop()
            self.mqtt_interface.client.disconnect()

        if self.gpu_enabled:
            pynvml.nvmlShutdown()
        
        tracemalloc.stop()


