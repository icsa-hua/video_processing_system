from __future__ import annotations

import logging
import queue as queue_module
import time
from multiprocessing import Process, Queue, Value
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from obs_system.application_module.dummy_application.dummy_app import Application
from obs_system.application_module.dummy_application.pipeline_config import PipelineConfig
from obs_system.utils.logger import get_logger


logger = get_logger(f"obs_system.{__name__}")
logging.getLogger("uvicorn.error").propagate = False

server = FastAPI()
worker_lock = Lock()
worker_process: Process | None = None
video_processing = False
frame_queue: Queue = Queue(maxsize=2)
producer_ready = Value("b", False)


class VideoProcessingRequest(BaseModel):
    model_name: str
    video_source: str
    type: str = "tracking"
    mqtt: bool = False
    show: bool = False
    verbose: bool = False
    save: bool = False
    roi: bool = False
    half: bool = False
    fep: bool = False
    bench: bool = False
    bench_labels: str = "samples/labels"
    use_TRT: bool = False
    plot_perf: bool = False
    only_FPS: bool = False
    preview_max_width: int = 960
    preview_jpeg_quality: int = 70
    preview_fps: float = 8.0


def _is_worker_alive() -> bool:
    return worker_process is not None and worker_process.is_alive()


def _offer_queue_item(target_queue: Queue, item: bytes | None) -> None:
    try:
        target_queue.put_nowait(item)
    except queue_module.Full:
        try:
            target_queue.get_nowait()
        except queue_module.Empty:
            pass
        try:
            target_queue.put_nowait(item)
        except queue_module.Full:
            pass


def _reset_preview_state() -> None:
    global frame_queue, producer_ready
    frame_queue = Queue(maxsize=2)
    producer_ready.value = False


def _stop_worker() -> None:
    global worker_process, video_processing

    video_processing = False
    producer_ready.value = False
    _offer_queue_item(frame_queue, None)

    if worker_process is not None:
        if worker_process.is_alive():
            worker_process.terminate()
            worker_process.join(timeout=5)
        worker_process.close()
        worker_process = None


def _worker_main(config: PipelineConfig, preview_queue: Queue, ready_flag: Value) -> None:
    try:
        app = Application() 
        app.run_application(config, producer_flag=ready_flag, preview_queue=preview_queue)
    except Exception:
        logger.exception("Video processing worker failed")
        raise
    finally:
        ready_flag.value = False
        _offer_queue_item(preview_queue, None)


def _build_config(request: VideoProcessingRequest) -> PipelineConfig:
    config = PipelineConfig(**request.model_dump(), gui=True)
    return config.validate()


def _frame_stream(queue_ref: Queue, ready_flag: Value):
    while not ready_flag.value:
        if not _is_worker_alive():
            return
        time.sleep(0.03)

    while True:
        try:
            frame_bytes = queue_ref.get(timeout=0.5)
        except queue_module.Empty:
            if not _is_worker_alive():
                return
            continue

        if frame_bytes is None:
            return

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )


@server.post("/")
def start_video_processing(request: VideoProcessingRequest):
    global worker_process, video_processing

    try:
        config = _build_config(request)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with worker_lock:
        if _is_worker_alive():
            return {"status": "Already running"}

        _reset_preview_state()
        worker_process = Process(target=_worker_main, args=(config, frame_queue, producer_ready))
        worker_process.start()
        video_processing = True

    return {
        "status": "Processing started",
        "model": config.model_name,
        "video_source": config.video_source,
    }


@server.post("/shutdown")
def stop_video_processing():
    with worker_lock:
        if not _is_worker_alive():
            return {"status": "Not running"}
        _stop_worker()

    return {"status": "Worker stopped"}


@server.get("/status")
def get_status():
    return {
        "running": _is_worker_alive(),
        "preview_ready": bool(producer_ready.value),
    }


@server.get("/video_feed")
def get_frame():
    if not _is_worker_alive() and not producer_ready.value:
        raise HTTPException(status_code=409, detail="No active video processing worker")

    return StreamingResponse(
        _frame_stream(frame_queue, producer_ready),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
