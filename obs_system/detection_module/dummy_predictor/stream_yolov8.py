from __future__ import annotations

from obs_system.detection_module.interface.streaming_compressed import OptimizedStreamer
from obs_system.logic_module.dummy_logic.tracker_sv import TrackerHandler
from obs_system.utils.global_config import CONF_THR, NMS_IOU
from obs_system.utils.logger import get_logger

import numpy as np
import torch

from memory_profiler import profile as mem_profile
from pathlib import Path
from typing import Any
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


logger = get_logger("obs_system." + __name__)


class _Yolov8PtAdapter:
    fp16 = False

    def __init__(self, *, model_path: str | Path, device: torch.device) -> None:
        self._wrapper_model = YOLO(str(model_path)).to(device)
        self._infer_model = self._wrapper_model.model
        self._device = device

    def __call__(self, images: torch.Tensor, orig_imgs=None, debug: bool = False):
        results = self._wrapper_model.predict(
            source=images,
            stream=False,
            verbose=debug,
            half=self.fp16,
            conf=CONF_THR,
            iou=NMS_IOU,
        )

        all_boxes = []
        all_scores = []
        all_classes = []

        for result in results:
            if result.boxes is None or result.boxes.xyxy.numel() == 0:
                all_boxes.append(torch.empty((0, 4), device=self._device, dtype=torch.float32))
                all_scores.append(torch.empty((0,), device=self._device, dtype=torch.float32))
                all_classes.append(torch.empty((0,), device=self._device, dtype=torch.int64))
                continue

            all_boxes.append(result.boxes.xyxy.detach().to(self._device))
            all_scores.append(result.boxes.conf.detach().to(self._device))
            all_classes.append(result.boxes.cls.detach().to(self._device).long())

        return (all_boxes, all_scores, all_classes), None

    def warmup(self, *, micro: int, warmup_sessions: int) -> None:
        if self._device.type != "cuda":
            return

        dummy = torch.zeros((1, 3, 640, 640), device=self._device, dtype=torch.float32)
        for _ in range(max(1, warmup_sessions)):
            _ = self._wrapper_model.predict(source=dummy, stream=False, verbose=False)


class Yolov8Streamer(OptimizedStreamer):
    def __init__(self, cfg: Any = DEFAULT_CFG, overrides=None, _callbacks=None) -> None:
        super().__init__(cfg, overrides or {}, _callbacks)
        self.force_streaming_no_tiles = True
        self.model_tag = "pt_yolov8"

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
        im = im.half() if getattr(self.model, "fp16", False) else im.float()

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
        self.model = _Yolov8PtAdapter(model_path=model_path, device=self.device)
        self.tracker_model = TrackerHandler(tracker_choice="byte_tracker") if opt == "tracking" else None
        self.stride = 16 if self.args.half else 32
        self.model_tag = f"pt_{model_path.stem}"

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
