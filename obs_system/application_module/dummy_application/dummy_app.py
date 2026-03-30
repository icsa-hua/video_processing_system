from obs_system.communication_module.mqtt_com.config import BROKER, CALLBACK_API_VERSION, CREATE_SUBSCRIBER, KEEPALIVE, PORT
from obs_system.communication_module.mqtt_com.config import *
from obs_system.communication_module.mqtt_com.message_transmitter import CBORMQTTCropClientCV2, RealMQTT
from obs_system.detection_module.interface.factory import StreamerFactory
from obs_system.detection_module.interface.model_registry import build_default_model_registry
from obs_system.logic_module.dummy_logic.region_setter import RegionSetter
from obs_system.logic_module.dummy_logic.subtractor import Subtractor
from obs_system.logic_module.dummy_logic.fisheye import FishEyeProjection
from obs_system.utils.common import check_nvidia_existence
from obs_system.utils.logger import get_logger
from obs_system.utils.appraisal import perf, frame_list

import os
import numpy as np
import psutil
import pynvml
import time
import platform
import tracemalloc

from jtop import jtop
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict
from ultralytics.utils import DEFAULT_CFG

logger = get_logger("obs_system." + __name__)


class Application:
    """
    This class is responsible for the application logic. It creates the detection class instances,
    initiates the broker, and starts the detection process. It also provides metrics that show
    the hardware utilization and the memory usage.
    """

    def __init__(self, save: bool = False, verbose: bool = False):
        self.source: str = ""
        self.save_outputs = save
        self.verbose_outputs = verbose
        self.use_TRT = None
        self.model = None
        self.parent_path: str = os.getcwd()
        self.mqtt: bool = False
        self.streamer: Any = None
        self.mqtt_publisher: Any = None
        self.mqtt_subscriber: Any = None
        self.logic_module = defaultdict()
        self.gpu_enabled: bool = False
        self.machine_type = self.get_device_type()
        self.model_registry = build_default_model_registry()
        self.streamer_factory = StreamerFactory(
            registry=self.model_registry,
            cfg=DEFAULT_CFG,
            overrides={},
            callbacks=None,
        )


    def get_device_type(self): 
        arch = platform.machine() 

        if arch in ['x86_64', 'AMD64']: 
            return "desktop" 

        if arch == 'aarch64': 
            return 'jetson' 

        return 'unknown_arm' 


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


    def setup_model(
        self,
        model_name: str = "yolov8s.onnx",
        path_to_load: Optional[str | Path] = "assets/compressed_models",
        opt: str = "tracking",
    ):
        if path_to_load is None:
            raise TypeError("path_to_load cannot be None")

        if not model_name.endswith(".pt") and not model_name.endswith(".onnx") and not model_name.endswith(".engine"):
            raise TypeError("The model is imperative to be either .pt, .onnx or .engine format.")

        if path_to_load == "":
            if not os.path.exists(model_name):
                raise FileNotFoundError("model_name was passed as the path, and it does not exist")
            path_to_load = model_name
        elif not os.path.exists(path_to_load):
            raise FileNotFoundError(path_to_load)

        self.streamer, resolved = self.streamer_factory.create(
            model_name=model_name,
            path_to_load=path_to_load,
            use_tensorrt=bool(self.use_TRT),
            opt=opt,
        )
        self.model = self.streamer.model

        logger.debug(
            f"-- Streaming through {model_name} found in {path_to_load}. "
            f"Backend: {resolved.backend}. --"
        )


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

        self.process_stream(producer_flag=producer_flag, queue=queue)
        perf.finalize()
        stats = perf.results()

        # add FPS & percentiles for total frame time
        ft = np.array(frame_list, dtype=np.float32)
        stats.update(
            {
                "frame_p50_ms": float(np.percentile(ft, 50)) if ft.size else 0.0,
                "frame_p95_ms": float(np.percentile(ft, 95)) if ft.size else 0.0,
                "frame_p99_ms": float(np.percentile(ft, 99)) if ft.size else 0.0,
                "FPS_mean": (1000.0 / float(ft.mean())) if ft.size else 0.0,
            }
        )

        logger.debug(stats)

        return


    def process_stream(self, producer_flag=None, queue=None):
        logger.debug("-- Starting the video streaming process --")

        kwargs = {"save": self.save_outputs, "verbose": self.verbose_outputs}
        self.streamer(
            source=self.source,
            model=self.model,
            logic_module=self.logic_module,
            mqtt_broker=self.mqtt_publisher,
            producer_flag=producer_flag,
            queue_list=queue,
            **{key: kwargs[key] for key in ["verbose", "save"]},
        )

        return self.streamer.results


    def setup_mqtt(self, qos: int = QOS, jpeg_quality: int = JPEG_QUALITY):
        if not self.mqtt:
            self.mqtt_publisher = None
            self.mqtt_subscriber = None
            return

        self.mqtt_topic = TOPIC

        if CREATE_SUBSCRIBER:
            self.mqtt_subscriber = CBORMQTTCropClientCV2(
                broker_address=BROKER,
                topic=self.mqtt_topic,
                callback_version=CALLBACK_API_VERSION,
                client_id="obs-receiver",
                qos=qos,
                jpeg_quality=jpeg_quality,
            )
            self.mqtt_subscriber.set_on_batch_callback(self.mqtt_subscriber.on_batch)
            self.mqtt_subscriber.client.on_message = self.mqtt_subscriber.on_message
            self.mqtt_subscriber.client.on_connect = self.mqtt_subscriber.on_connect
            self.mqtt_subscriber.connect(port=PORT, keepalive=KEEPALIVE)
            self.mqtt_subscriber.start_loop(background=True)
            time.sleep(1)
            self.mqtt_subscriber.subscribe(self.mqtt_topic)

        self.mqtt_publisher = CBORMQTTCropClientCV2(
            broker_address=BROKER,
            topic=self.mqtt_topic,
            callback_version=CALLBACK_API_VERSION,
            client_id="obs-sender",
            qos=qos,
            jpeg_quality=jpeg_quality,
        )

        self.mqtt_publisher.connect(port=PORT, keepalive=KEEPALIVE)

    def statistics(self):
        logger.debug("-- Performance metrics --")

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()

        logger.info(f"-" * 84)
        logger.info(f"|  Total Inference time: {end_time - self.start_time:.4f} seconds ")
        logger.info(
            f"|  Current Environment RAM usage (Psutil): {self.process_memory.memory_info().rss / (1024**2):.2f} MB."
        )
        logger.info(f"|  Current memory usage (Tracemalloc): {current / (1024 ** 2):.2f} MB.")
        logger.info(f"|  Peak memory usage (Tracemalloc): {peak / (1024 ** 2):.2f} MB.")

        if self.machine_type == 'desktop' : 
            if self.gpu_enabled and hasattr(self, "handle"):
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

                    total_gpu_mem = int(mem_info.total) / (1024 ** 2)
                    used_gpu_mem = int(mem_info.used) / (1024 ** 2)
                    free_gpu_mem = int(mem_info.free) / (1024 ** 2)

                    logger.info(f"|  Total GPU memory: {total_gpu_mem:.2f} MB")
                    logger.info(f"|  Used GPU memory: {used_gpu_mem:.2f} MB")
                    logger.info(f"|  Free GPU memory: {free_gpu_mem:.2f} MB")

                except pynvml.NVMLError as e:
                    logger.error(f"| Failed to get GPU metrics: {e}")
        elif self.machine_type == 'jetson': 
            with jtop() as jetson: 
                if jetson.ok(): 
                    print(jetson.memory)

        logger.info(f"-" * 84)

    def close_app(self):
        logger.debug("-- Terminating the application --")

        self.statistics()

        if self.mqtt_subscriber is not None:
            self.mqtt_subscriber.client.loop_stop()
            self.mqtt_subscriber.client.disconnect()

        if self.mqtt_publisher is not None:
            self.mqtt_publisher.client.disconnect()

        if self.gpu_enabled:
            pynvml.nvmlShutdown()

        tracemalloc.stop()
