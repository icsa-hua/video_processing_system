from __future__ import annotations

import logging
import queue as queue_module
import time
from multiprocessing import Process, Queue, Value
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from obs_system.application_module.dummy_application.dummy_app import Application
from obs_system.application_module.dummy_application.pipeline_config import PipelineConfig
from obs_system.application_module.dummy_application.stream_examiner import (
    StreamExaminer,
    StreamExaminerConfig,
)
from obs_system.utils.logger import get_logger


logger = get_logger(f"obs_system.{__name__}")
logging.getLogger("uvicorn.error").propagate = False

server = FastAPI()
worker_lock = Lock()
worker_process: Process | None = None
video_processing = False
frame_queue: Queue = Queue(maxsize=2)
producer_ready = Value("b", False)
examine_lock = Lock()
examine_process: Process | None = None
examine_frame_queue: Queue = Queue(maxsize=2)
examine_ready = Value("b", False)
examine_stop_requested = Value("b", False)


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
    stream_limit_hours: float = 1.0


class StreamExaminationRequest(BaseModel):
    video_source: str
    preview_max_width: int = 960
    preview_jpeg_quality: int = 70
    preview_fps: float = 8.0


def _is_process_alive(process: Process | None) -> bool:
    return process is not None and process.is_alive()


def _is_worker_alive() -> bool:
    return _is_process_alive(worker_process)


def _is_examine_worker_alive() -> bool:
    return _is_process_alive(examine_process)


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


def _reset_examine_preview_state() -> None:
    global examine_frame_queue, examine_ready
    examine_frame_queue = Queue(maxsize=2)
    examine_ready.value = False
    examine_stop_requested.value = False


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


def _stop_examine_worker() -> None:
    global examine_process

    examine_stop_requested.value = True
    examine_ready.value = False
    _offer_queue_item(examine_frame_queue, None)

    if examine_process is not None:
        examine_process.join(timeout=3)
        if examine_process.is_alive():
            examine_process.terminate()
            examine_process.join(timeout=5)
        examine_process.close()
        examine_process = None

    examine_stop_requested.value = False


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


def _examine_worker_main(
    config: StreamExaminerConfig,
    preview_queue: Queue,
    ready_flag: Value,
    stop_flag: Value,
) -> None:
    try:
        print(f"[EXAMINE] Worker started for source: {config.video_source}", flush=True)
        examiner = StreamExaminer(config)
        examiner.run(preview_queue=preview_queue, ready_flag=ready_flag, stop_flag=stop_flag)
    except Exception:
        print(
            "[EXAMINE] Worker stopped with an error. "
            "If the stream-specific messages above mention open/frame timeouts, the source is the issue. "
            "Otherwise inspect the service traceback below.",
            flush=True,
        )
        logger.exception("Live stream examination worker failed")
        raise
    finally:
        ready_flag.value = False
        _offer_queue_item(preview_queue, None)


def _build_config(request: VideoProcessingRequest) -> PipelineConfig:
    config = PipelineConfig(**request.model_dump(), gui=True)
    return config.validate()


def _build_examination_config(request: StreamExaminationRequest) -> StreamExaminerConfig:
    return StreamExaminerConfig(**request.model_dump()).validate()


def _frame_stream(queue_ref: Queue, ready_flag: Value, process_is_alive):
    while not ready_flag.value:
        if not process_is_alive():
            return
        time.sleep(0.03)

    while True:
        try:
            frame_bytes = queue_ref.get(timeout=0.5)
        except queue_module.Empty:
            if not process_is_alive():
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


@server.post("/examine_stream")
def start_stream_examination(request: StreamExaminationRequest):
    global examine_process

    try:
        config = _build_examination_config(request)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with examine_lock:
        if _is_examine_worker_alive():
            return {"status": "Already running"}

        _reset_examine_preview_state()
        examine_process = Process(
            target=_examine_worker_main,
            args=(config, examine_frame_queue, examine_ready, examine_stop_requested),
        )
        examine_process.start()

    return {
        "status": "Stream examination started",
        "video_source": config.video_source,
    }


@server.post("/shutdown")
def stop_video_processing():
    stopped_any_worker = False

    with worker_lock:
        if _is_worker_alive():
            _stop_worker()
            stopped_any_worker = True

    with examine_lock:
        if _is_examine_worker_alive():
            _stop_examine_worker()
            stopped_any_worker = True

    if not stopped_any_worker:
        return {"status": "Not running"}

    return {"status": "Workers stopped"}


@server.post("/examine_stream/stop")
def stop_stream_examination():
    with examine_lock:
        if not _is_examine_worker_alive():
            return {"status": "Not running"}
        _stop_examine_worker()

    return {"status": "Examination worker stopped"}


@server.get("/status")
def get_status():
    return {
        "running": _is_worker_alive(),
        "preview_ready": bool(producer_ready.value),
    }


@server.get("/examine_stream/status")
def get_examine_status():
    return {
        "running": _is_examine_worker_alive(),
        "preview_ready": bool(examine_ready.value),
    }


@server.get("/video_feed")
def get_frame():
    if not _is_worker_alive() and not producer_ready.value:
        raise HTTPException(status_code=409, detail="No active video processing worker")

    return StreamingResponse(
        _frame_stream(frame_queue, producer_ready, _is_worker_alive),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@server.get("/examine_stream/feed")
def get_examine_frame():
    if not _is_examine_worker_alive() and not examine_ready.value:
        raise HTTPException(status_code=409, detail="No active stream examination worker")

    return StreamingResponse(
        _frame_stream(examine_frame_queue, examine_ready, _is_examine_worker_alive),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@server.get("/examine_stream/view")
def get_examine_view():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <style>
                html, body {
                    margin: 0;
                    height: 100%;
                    background: #0f1116;
                }
                body {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                img {
                    width: 100%;
                    max-width: 960px;
                    border-radius: 0.75rem;
                }
            </style>
        </head>
        <body>
            <img src="/examine_stream/feed" alt="Examine stream feed" />
        </body>
        </html>
        """
    )
