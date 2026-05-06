from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import cv2

from obs_system.utils.logger import get_logger


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

logger = get_logger("obs_system." + __name__)


@dataclass(frozen=True)
class StreamExaminerConfig:
    video_source: str
    preview_max_width: int = 960
    preview_jpeg_quality: int = 70
    preview_fps: float = 8.0

    def validate(self) -> "StreamExaminerConfig":
        if not self.video_source or not self.video_source.strip():
            raise ValueError("Please provide a live stream URL.")

        if int(self.preview_max_width) < 0:
            raise ValueError("preview_max_width must be greater than or equal to 0")

        quality = int(self.preview_jpeg_quality)
        if quality < 1 or quality > 100:
            raise ValueError("preview_jpeg_quality must be between 1 and 100")

        if float(self.preview_fps) < 0:
            raise ValueError("preview_fps must be greater than or equal to 0")

        return self


class StreamExaminer:
    def __init__(self, config: StreamExaminerConfig):
        self.config = config.validate()
        self.capture: cv2.VideoCapture | None = None
        self.last_emit_ts = 0.0

    def _encode_frame(self, frame: Any) -> bytes | None:
        preview = frame
        max_width = int(self.config.preview_max_width)

        if max_width > 0 and frame.shape[1] > max_width:
            scale = max_width / float(frame.shape[1])
            preview = cv2.resize(
                frame,
                (max_width, max(1, int(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.preview_jpeg_quality)],
        )
        if not ok:
            return None

        return encoded.tobytes()

    def run(self, preview_queue: Any, ready_flag: Any, stop_flag: Any) -> None:
        source = self.config.video_source.strip()
        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open live stream: {source}")

        preview_fps = float(self.config.preview_fps)

        try:
            while not bool(stop_flag.value):
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                if preview_fps > 0:
                    now = time.perf_counter()
                    if (now - self.last_emit_ts) < (1.0 / preview_fps):
                        continue
                    self.last_emit_ts = now

                encoded = self._encode_frame(frame)
                if encoded is None:
                    continue

                ready_flag.value = True
                try:
                    preview_queue.put_nowait(encoded)
                except Exception:
                    try:
                        preview_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        preview_queue.put_nowait(encoded)
                    except Exception:
                        pass
        finally:
            ready_flag.value = False
            if self.capture is not None:
                self.capture.release()
