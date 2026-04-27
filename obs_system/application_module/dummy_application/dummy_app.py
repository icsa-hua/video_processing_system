from obs_system.communication_module.mqtt_com.config import BROKER, CALLBACK_API_VERSION, CREATE_SUBSCRIBER, KEEPALIVE, PORT
from obs_system.communication_module.mqtt_com.config import *
from obs_system.communication_module.mqtt_com.message_transmitter import CBORMQTTCropClientCV2, RealMQTT
from obs_system.application_module.dummy_application.pipeline_config import PipelineConfig
from obs_system.detection_module.interface.factory import StreamerFactory
from obs_system.detection_module.interface.model_registry import build_default_model_registry
from obs_system.logic_module.dummy_logic.region_setter import RegionSetter
from obs_system.logic_module.dummy_logic.subtractor import Subtractor
from obs_system.logic_module.dummy_logic.fisheye import FishEyeProjection
from obs_system.utils.common import check_nvidia_existence
from obs_system.utils.logger import get_logger
from obs_system.utils.appraisal import perf, frame_list, StepContext

import os
import gc
import numpy as np
import psutil
import pynvml
import time
import platform
import tracemalloc
import torch

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


    def setup_process(self, config: PipelineConfig):
        # Check for GPU (NVIDIA) to allow the program to run GPU statistics
        self.gpu_enabled = check_nvidia_existence()

        if self.gpu_enabled:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        self.process_memory = psutil.Process(os.getpid())
        if "://" in config.video_source:
            self.source = config.video_source
        else:
            source_path = Path(config.video_source)
            self.source = str(source_path if source_path.is_absolute() else Path(self.parent_path) / source_path)
        self.use_TRT = bool(config.use_TRT)
        self.mqtt = bool(config.mqtt)
        self.start_time = time.time()

        DEFAULT_CFG.show = bool(config.show)
        DEFAULT_CFG.gui = bool(config.gui)
        DEFAULT_CFG.save = self.save_outputs
        DEFAULT_CFG.verbose = self.verbose_outputs
        DEFAULT_CFG.half = bool(config.half)
        DEFAULT_CFG.bench = bool(config.bench)
        DEFAULT_CFG.bench_labels = config.bench_labels
        DEFAULT_CFG.roi = bool(config.roi)
        DEFAULT_CFG.plot_performance = bool(config.plot_perf)
        DEFAULT_CFG.only_FPS = bool(config.only_FPS)
        DEFAULT_CFG.preview_max_width = int(config.preview_max_width)
        DEFAULT_CFG.preview_jpeg_quality = int(config.preview_jpeg_quality)
        DEFAULT_CFG.preview_fps = float(config.preview_fps)
        DEFAULT_CFG.stream_limit_hours = float(config.stream_limit_hours)

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


    def setup_logic_module(self, config: PipelineConfig):
        # Change this based on your video. Get the first frame.
        self.logic_module["SUBTRACTOR"] = Subtractor()
        self.logic_module["ROI"] = RegionSetter()

        if config.fep:
            self.logic_module["FEP"] = FishEyeProjection(crop=0.00)
        else:
            self.logic_module["FEP"] = None


    def run_app(self, producer_flag=None, preview_queue=None):
        self.statistics()

        self.process_stream(producer_flag=producer_flag, preview_queue=preview_queue)
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


    def process_stream(self, producer_flag=None, preview_queue=None):
        logger.debug("-- Starting the video streaming process --")

        kwargs = {"save": self.save_outputs, "verbose": self.verbose_outputs}
        self.streamer(
            source=self.source,
            model=self.model,
            logic_module=self.logic_module,
            mqtt_broker=self.mqtt_publisher,
            producer_flag=producer_flag,
            preview_queue=preview_queue,
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
            client_id=CLIENT_NAME,
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


    def cleanup_runtime_resources(self, *, producer_flag=None, preview_queue=None) -> None:
        if self.mqtt_subscriber is not None:
            try:
                self.mqtt_subscriber.client.loop_stop()
                self.mqtt_subscriber.client.disconnect()
            except Exception:
                logger.exception("Failed to close MQTT subscriber cleanly")
            finally:
                self.mqtt_subscriber = None

        if self.mqtt_publisher is not None:
            try:
                self.mqtt_publisher.client.disconnect()
            except Exception:
                logger.exception("Failed to close MQTT publisher cleanly")
            finally:
                self.mqtt_publisher = None

        if self.streamer is not None and hasattr(self.streamer, "release_session_resources"):
            self.streamer.release_session_resources(
                preview_queue=preview_queue,
                producer_flag=producer_flag,
            )

        self.model = None
        self.streamer = None
        self.logic_module.clear()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                logger.exception("Failed to release CUDA cache cleanly")

        if self.gpu_enabled:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                logger.debug("NVML was already shut down")

        if tracemalloc.is_tracing():
            tracemalloc.stop()

        gc.collect()


    def run_application(
        self,
        config: PipelineConfig,
        *,
        producer_flag=None,
        preview_queue=None,
    ) -> None:
        model_specification = config.resolve_model()
        self.save_outputs = bool(config.save)
        self.verbose_outputs = bool(config.verbose)
        process_initialized = False

        try:
            with StepContext(name="Setup Process", catch=(KeyError, ModuleNotFoundError)):
                self.setup_process(config)
                process_initialized = True

            with StepContext(name="Setup Model", catch=(OSError, ValueError)):
                self.setup_model(
                    model_name=f"{model_specification.name}.{model_specification.kind}",
                    path_to_load=model_specification.path,
                    opt=config.type,
                )

            with StepContext(name="Setup Logic", catch=(KeyError, IndexError)):
                self.setup_logic_module(config)

            with StepContext(name="Setup MQTT", catch=(ConnectionError, TimeoutError)):
                if self.mqtt:
                    self.setup_mqtt()
                else:
                    logger.debug("[MQTT] interface is disabled")

            with StepContext(name="Run_App", catch=(RuntimeError,)):
                self.run_app(producer_flag=producer_flag, preview_queue=preview_queue)
        finally:
            if process_initialized:
                try:
                    self.close_app()
                except Exception:
                    logger.exception("Failed to collect final application statistics")
            self.cleanup_runtime_resources(
                producer_flag=producer_flag,
                preview_queue=preview_queue,
            )
