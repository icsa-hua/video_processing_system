from __future__ import annotations

from obs_system.compressed.interface.compressed_yolo import CompressedYOLO
from obs_system.detection_module.interface.streaming_compressed import OptimizedStreamer
from obs_system.logic_module.dummy_logic.tracker_sv import TrackerHandler
from obs_system.utils.logger import get_logger

import numpy as np
import torch

from memory_profiler import profile as mem_profile
from pathlib import Path
from typing import Any
from ultralytics.data.augment import LetterBox
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


logger = get_logger("obs_system." + __name__)


class OnnxY8Streamer(OptimizedStreamer):
    def __init__(self, cfg: Any = DEFAULT_CFG, overrides=None, _callbacks=None) -> None:
        super().__init__(cfg, overrides or {}, _callbacks)
        self.force_streaming_no_tiles = True
        self.model_tag = "onnx_yolov8"

    def __call__(self, source=None, model=None, logic_module=None, mqtt_broker=None, producer_flag=None, preview_queue=None, *args, **kwargs):
        return super().__call__(source, model, logic_module, mqtt_broker, producer_flag, preview_queue, *args, **kwargs)

    def pre_transform(self, im):
        self.stride = 16 if self.args.half else 32
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(self.imgsz, auto=same_shapes, stride=self.stride)
        return [letterbox(image=x) for x in im]

    def preprocess(self, im: Any):
        not_tensor = not isinstance(im, torch.Tensor)

        if not_tensor:
            if isinstance(im, np.ndarray):
                im = list(im)
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.float()

        if not_tensor:
            im = im.div(255.0)

        return im

    def setup_model(self, model_name: str, path_to_load: str | Path, opt: str = "tracking") -> None:
        model_path = Path(path_to_load)
        if model_path.is_dir():
            model_path = model_path / model_name

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        self.device = select_device(self.args.device, verbose=self.args.verbose)
        self.model = CompressedYOLO(str(model_path))
        self.height = self.model.input_height
        self.width = self.model.input_width
        self.tracker_model = TrackerHandler(tracker_choice="byte_tracker") if opt == "tracking" else None
        self.stride = 16 if self.args.half else 32
        self.model_tag = f"onnx_{model_path.stem}"

        if self.args.verbose:
            logger.info("[checked] Model %s successfully set up", model_path.name)

    @smart_inference_mode()
    def stream_inference(self, source, model, producer_flag, preview_queue, *args, **kwargs):
        return super().stream_inference(source, model, producer_flag, preview_queue, *args, **kwargs)

    @mem_profile
    def _stream_inference_impl(self, **kwargs):
        return super()._stream_inference_impl(**kwargs)

    def _stream_inference_impl_tiles(self, **kwargs):
        return self._stream_inference_impl(**kwargs)
