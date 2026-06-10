from __future__ import annotations

import os 
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obs_system.utils.common import ModelSpecification, check_model_name
from obs_system.application_module.dummy_application.camera_config import keys

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp" 

DEFAULT_MODEL = "assets/compressed_models/yolov8s.engine"
# DEFAULT_VIDEO_SOURCE = "rtsp://admin:edgeAI!kamera1@tbfw.edi.lv:50854/ISAPI/Streaming/Channels/101"
# DEFAULT_VIDEO_SOURCE = f"rtsp://{keys.USER}:{keys.PASS}@vtfw.edi.lv:{keys.PORT1}/axis-media/media.amp?resolution=1280x960"
DEFAULT_VIDEO_SOURCE = keys.RECTILINEAR_RTSP
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8503
DEFAULT_BENCH_LABELS = "samples/labels"
DEFAULT_STREAM_LIMIT_HOURS = 1.0


@dataclass(frozen=True)
class PipelineConfig:
    model_name: str = DEFAULT_MODEL
    video_source: str = DEFAULT_VIDEO_SOURCE
    type: str = "tracking"
    gui: bool = False
    mqtt: bool = False
    show: bool = False
    verbose: bool = False
    port_address: int = DEFAULT_PORT
    host_address: str = DEFAULT_HOST
    save: bool = False
    roi: bool = False
    half: bool = False
    fep: bool = False
    bench: bool = False
    bench_labels: str = DEFAULT_BENCH_LABELS
    use_TRT: bool = False
    plot_perf: bool = False
    only_FPS: bool = False
    preview_max_width: int = 960
    preview_jpeg_quality: int = 70
    preview_fps: float = 8.0
    stream_limit_hours: float = DEFAULT_STREAM_LIMIT_HOURS
    lane_recalibration_interval_frames: int = 20
    jetson_profile: bool = False
    jetson_hazard_scale: float = 1.0
    jetson_cpu_threads: int = 0
    force_tiles: bool = False
    panorama: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "PipelineConfig":
        values: dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(args, field_name):
                value = getattr(args, field_name)
                values[field_name] = value if value is not None else cls.__dataclass_fields__[field_name].default
        return cls(**values)

    def with_updates(self, **kwargs: Any) -> "PipelineConfig":
        data = {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}
        data.update(kwargs)
        return PipelineConfig(**data)

    def validate(self) -> "PipelineConfig":
        if self.bench and not Path(self.bench_labels).exists():
            raise ValueError("Submit a correct path for the GT labels, that matches the video")

        suffix = Path(self.model_name).suffix.lower()
        if self.use_TRT and suffix not in {".onnx", ".engine"}:
            raise TypeError(
                "Can't use TRT if the model is not in ONNX or TRT format. "
                "Check ultralytics guide for more information: https://docs.ultralytics.com/modes/export/"
            )

        if not self.use_TRT and suffix == ".engine":
            raise TypeError(
                "Can't use TRT model without passing use_TRT. "
                "You can pass this argument with python3 scripts/obs_pipeline.py --use_TRT"
            )

        if self.host_address != "localhost" and self.host_address != "0.0.0.0":
            raise ValueError("Set host address to 'localhost' or '0.0.0.0'")

        if float(self.stream_limit_hours) < 0:
            raise ValueError("Set stream_limit_hours to a value greater than or equal to 0")

        if int(self.lane_recalibration_interval_frames) < 0:
            raise ValueError("Set lane_recalibration_interval_frames to a value greater than or equal to 0")

        if not (0.0 < float(self.jetson_hazard_scale) <= 1.0):
            raise ValueError("Set jetson_hazard_scale to a value in the interval (0, 1]")

        if int(self.jetson_cpu_threads) < 0:
            raise ValueError("Set jetson_cpu_threads to a value greater than or equal to 0")

        return self

    def resolve_model(self) -> ModelSpecification:
        return check_model_name(
            model=self.model_name,
            model_dirs=["assets/compressed_models"],
            must_exist=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--model_name",
        metavar="M",
        default=DEFAULT_MODEL,
        help="Model to use (Yolov5, Yolov8 (Default), MaskRCNN, ONNX (yolov5, yolov8))",
    )
    argparser.add_argument(
        "--video_source",
        metavar="SO",
        default=DEFAULT_VIDEO_SOURCE,
        help="Source to use - Local video path (.mp4) or stream index (key needs to be provided)",
    )
    argparser.add_argument("--type", metavar="T", default="tracking", help="Use tracking with bytetracker or simple detection")
    argparser.add_argument("--gui", metavar="G", action=argparse.BooleanOptionalAction, help="Use GUI to select video source and model")
    argparser.add_argument("--mqtt", metavar="M", action=argparse.BooleanOptionalAction, help="Use MQTT to send data to server")
    argparser.add_argument("--show", metavar="SH", action=argparse.BooleanOptionalAction, help="Show real-time result inference")
    argparser.add_argument("--verbose", metavar="V", action=argparse.BooleanOptionalAction, help="Show results of inference in stdout")
    argparser.add_argument("--port_address", metavar="P", type=int, default=DEFAULT_PORT, help="Port for Streamlit service interface")
    argparser.add_argument("--host_address", metavar="H", default=DEFAULT_HOST, help="Host server for both streamlit and fastapi")
    argparser.add_argument("--save", metavar="SA", action=argparse.BooleanOptionalAction, help="Save inference results to file")
    argparser.add_argument("--roi", metavar="R", action=argparse.BooleanOptionalAction, help="Use Region of Interest to detect obstacles")
    argparser.add_argument("--half", metavar="HF", action=argparse.BooleanOptionalAction, help="Use Half the available resources by reducing the data size")
    argparser.add_argument("--fep", metavar="F", action=argparse.BooleanOptionalAction, help="Use of FishEye Projection based on camera")
    argparser.add_argument("--bench", metavar="BM", action=argparse.BooleanOptionalAction, help="Benchmark the Performance of the model and hardware.")
    argparser.add_argument("--bench-labels", metavar="BL", default=DEFAULT_BENCH_LABELS, help="Submit the label path for GT")
    argparser.add_argument("--use_TRT", metavar="TRT", action=argparse.BooleanOptionalAction, help="Use TensorRT engine for model inference")
    argparser.add_argument("--plot_perf", metavar="TRT", action=argparse.BooleanOptionalAction, help="Plot performance diagrams")
    argparser.add_argument("--only_FPS", metavar="TRT", action=argparse.BooleanOptionalAction, help="Measure average FPS regardless of plotting.")
    argparser.add_argument( "--stream_limit_hours", metavar="SL", type=float, default=DEFAULT_STREAM_LIMIT_HOURS, help="Maximum runtime in hours for live streams only. Set to 0 to disable the limit.")
    argparser.add_argument("--lane_recalibration_interval_frames", metavar="LR", type=int, default=0, help="Rerun lane calibration after this many frames for both local videos and live streams. Set to 0 to disable.")
    argparser.add_argument("--jetson_profile", metavar="JP", action=argparse.BooleanOptionalAction, help="Enable the Jetson-optimized execution branch.")
    argparser.add_argument("--jetson_hazard_scale", metavar="JHS", type=float, default=1.0, help="Scale factor for Jetson hazard-mask processing. Use values below 1.0 to downscale the hazard masks.")
    argparser.add_argument("--jetson_cpu_threads", metavar="JCT", type=int, default=0, help="CPU thread cap for the Jetson-optimized execution branch. Use 0 to keep the runtime default.")
    argparser.add_argument("--force_tiles", metavar="FT", action=argparse.BooleanOptionalAction, help="Force tiled inference regardless of image dimensions.")
    argparser.add_argument("--panorama", metavar="PAN", action=argparse.BooleanOptionalAction, help="Input is an equirectangular panorama (already unwrapped). Generates overlapping perspective views for YOLO and back-projects detections to panorama space.")
    return argparser
