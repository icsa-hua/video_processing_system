from __future__ import annotations
from abc import abstractmethod

from obs_system.compressed.interface.compressed_yolo import CompressedYOLO
from obs_system.compressed.interface.tensor_yolo import TensorRTYOLO
from obs_system.detection_module.interface.streaming_compressed import OptimizedStreamer
from obs_system.logic_module.dummy_logic.tracker_sv import TrackerHandler
from obs_system.utils.global_config import CONF_THR, NMS_IOU
from obs_system.utils.logger import get_logger

import numpy as np
import torch

from pathlib import Path
from typing import Any, Generator, Optional
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


logger = get_logger("obs_system." + __name__)


class _BaseModelAdapter:
    fp16 = False


    def warmup(self, *, micro: int, warmup_sessions: int) -> None:
        return


class _TensorRTAdapter(_BaseModelAdapter):
    fp16 = True

    def __init__(self, *, model_name: str, engine_path: str | Path) -> None:
        self._model = TensorRTYOLO(
            model_name=model_name,
            engine_path=engine_path,
            conf_thres=CONF_THR,
            iou_thres=NMS_IOU,
            fp16=True,
        )


    def __call__(self, images: torch.Tensor, orig_imgs=None, debug: bool = False):
        return self._model(images, orig_imgs=orig_imgs, debug=debug)


    def warmup(self, *, micro: int, warmup_sessions: int) -> None:
        self._model.warmup(micro=micro, warmup_sessions=warmup_sessions)


class _OnnxAdapter(_BaseModelAdapter):
    fp16 = False

    def __init__(self, *, model_path: str | Path, device: torch.device) -> None:
        self._model = CompressedYOLO(
            str(model_path),
            conf_thres=CONF_THR,
            iou_thres=NMS_IOU,
        )
        self._device = device


    def __call__(self, images: torch.Tensor, orig_imgs=None, debug: bool = False):
        boxes, scores, classes = self._model(images, debug=debug)
        return (boxes, scores, classes), None


    def warmup(self, *, micro: int, warmup_sessions: int) -> None:
        if not torch.cuda.is_available() or self._device.type != "cuda":
            return
        self._model.warmup(micro=micro, warmup_sessions=warmup_sessions, device=str(self._device))


class _PtAdapter(_BaseModelAdapter):
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
            half=_PtAdapter.fp16, # Necesssary for error unsupported operand Nonetype and bool
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


class UnifiedModelStreamer(OptimizedStreamer):
    def __init__(self, cfg: Any = DEFAULT_CFG, overrides=None, _callbacks=None) -> None:
        super().__init__(cfg, overrides or {}, _callbacks)
        self.model_backend = ""
        self.model_tag = "model"


    def pre_transform(self, im):
        self.stride = 16 if self.args.half else 32
        same_shapes = len({x.shape for x in im}) == 1
        if self.model_backend == "trt":
            letterbox = LetterBox(self.imgsz, auto=False, stride=self.stride)
        else:
            letterbox = LetterBox(self.imgsz, auto=same_shapes, stride=self.stride)
        return [letterbox(image=x) for x in im]

    # --- Chenked that this is the same for all models --- #
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


    def setup_model(
        self,
        *,
        model_name: str,
        path_to_load: str | Path,
        opt: str = "tracking",
        backend: str,
    ) -> None:
        model_path = Path(path_to_load)
        if model_path.is_dir():
            model_path = model_path / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        self.device = select_device(self.args.device, verbose=self.args.verbose)
        self.model_backend = backend
        self.model_tag = f"{backend}_{Path(model_name).stem}"

        if backend == "trt":
            self.model = _TensorRTAdapter(model_name=model_name, engine_path=model_path)
        elif backend == "onnx":
            self.model = _OnnxAdapter(model_path=model_path, device=self.device)
        elif backend == "pt":
            self.model = _PtAdapter(model_path=model_path, device=self.device)
        else:
            raise ValueError(f"Unsupported backend '{backend}'")

        self.tracker_model = TrackerHandler(tracker_choice="byte_tracker") if opt == "tracking" else None
        self.stride = 32 if not self.args.half else 16

        if self.args.verbose:
            logger.info(f"[checked] Model {model_name} successfully set up with backend '{backend}'")
        else:
            logger.debug(f"[checked] Model {model_name} successfully set up with backend '{backend}'")


    @smart_inference_mode()
    def stream_inference(self, source, model, producer_flag, preview_queue, *args, **kwargs):
        return super().stream_inference(source, model, producer_flag, preview_queue, *args, **kwargs)


    @smart_inference_mode()
    def _stream_inference_impl_tiles(self, **kwargs) -> Generator[Optional[Any], None, None]:
        return super()._stream_inference_impl_tiles(**kwargs)


    def _stream_inference_impl(self, **kwargs):
        return super()._stream_inference_impl(**kwargs)
