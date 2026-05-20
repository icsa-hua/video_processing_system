from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch
from ultralytics.engine.results import Results


@dataclass
class FrameDetections:
    frame_id: Any
    batch_index: int
    orig_img: np.ndarray
    boxes: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    classes: Optional[torch.Tensor] = None
    track_ids: Optional[torch.Tensor] = None
    sv_detections: Any = None
    _results_cache: Optional[Results] = field(default=None, init=False, repr=False)

    @classmethod
    def empty(cls, frame_id: Any, batch_index: int, orig_img: np.ndarray) -> "FrameDetections":
        return cls(frame_id=frame_id, batch_index=batch_index, orig_img=orig_img)

    @property
    def is_empty(self) -> bool:
        return (
            self.boxes is None
            or self.scores is None
            or self.classes is None
            or self.boxes.numel() == 0
            or self.scores.numel() == 0
            or self.classes.numel() == 0
        )

    def to_results(self, class_names: List[str], speed: Optional[dict] = None) -> Results:
        if self._results_cache is not None:
            if speed is not None:
                self._results_cache.speed = speed
            return self._results_cache

        if self.is_empty:
            boxes = torch.zeros((0, 6), dtype=torch.float32)
        else:
            cols = [self.boxes, self.scores[:, None]]
            if self.track_ids is not None:
                cols.insert(1, self.track_ids[:, None].to(torch.float32))
            cols.append(self.classes[:, None].to(torch.float32))
            boxes = torch.cat(cols, dim=1)

        results = Results(
            orig_img=self.orig_img,
            path=f"image_{self.frame_id}.jpg",
            names=class_names,
            boxes=boxes,
            speed=speed or {},
        )
        if self.sv_detections is not None:
            results.sv_detections = self.sv_detections
        self._results_cache = results
        return results


@dataclass
class DetectionBatch:
    frames: List[FrameDetections]

    def __iter__(self):
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)
