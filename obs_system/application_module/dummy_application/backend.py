from __future__ import annotations

import datetime
import logging
import os
import queue as queue_module
import time
from multiprocessing import Process, Queue, Value
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from obs_system.application_module.dummy_application.pipeline_config import PipelineConfig
from obs_system.application_module.dummy_application.stream_examiner import (
    StreamExaminer,
    StreamExaminerConfig,
)
from obs_system.utils.logger import get_logger


logger = get_logger(f"obs_system.{__name__}")
# logging.getLogger("uvicorn.error").propagate = False

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
record_lock = Lock()
record_process: Process | None = None
record_running = Value("b", False)
record_done = Value("b", False)
record_error = Value("b", False)
record_stop_flag = Value("b", False)
_current_record_output_path: str = ""


class VideoProcessingRequest(BaseModel):
    model_name: str
    video_source: str
    type: str = "tracking"
    mqtt: bool = False
    show: bool = False
    verbose: bool = False
    save: bool = False
    roi: bool = False
    roi_profile: str = ""
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
    lane_recalibration_interval_frames: int = 0
    force_tiles: bool = False
    panorama: bool = False


class StreamExaminationRequest(BaseModel):
    video_source: str
    preview_max_width: int = 960
    preview_jpeg_quality: int = 70
    preview_fps: float = 8.0


class StreamRecordRequest(BaseModel):
    video_source: str
    duration_seconds: int = 30
    output_dir: str = "recordings"


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


def _reset_record_state() -> None:
    record_running.value = False
    record_done.value = False
    record_error.value = False
    record_stop_flag.value = False


def _finalize_process(process: Process | None, *, graceful_timeout: float, force_timeout: float, process_name: str) -> Process | None:
    if process is None:
        return None

    process.join(timeout=graceful_timeout)
    if process.is_alive():
        logger.info("Stopping %s forcefully after graceful shutdown timeout", process_name)
        process.terminate()
        process.join(timeout=force_timeout)

    if process.is_alive():
        kill_fn = getattr(process, "kill", None)
        if callable(kill_fn):
            logger.warning("%s is still running after terminate(); sending kill()", process_name)
            kill_fn()
            process.join(timeout=2.0)

    if process.is_alive():
        logger.error("%s is still running after kill(); skipping close for now", process_name)
        return process

    process.close()
    return None


def _stop_worker() -> None:
    global worker_process, video_processing

    video_processing = False
    producer_ready.value = False
    _offer_queue_item(frame_queue, None)

    worker_process = _finalize_process(
        worker_process,
        graceful_timeout=0.5,
        force_timeout=5.0,
        process_name="video processing worker",
    )


def _stop_examine_worker() -> None:
    global examine_process

    examine_stop_requested.value = True
    examine_ready.value = False
    _offer_queue_item(examine_frame_queue, None)

    examine_process = _finalize_process(
        examine_process,
        graceful_timeout=3.0,
        force_timeout=5.0,
        process_name="stream examination worker",
    )

    examine_stop_requested.value = False


def _stop_record_worker() -> None:
    global record_process

    record_stop_flag.value = True

    record_process = _finalize_process(
        record_process,
        graceful_timeout=5.0,
        force_timeout=5.0,
        process_name="stream recording worker",
    )


def _record_worker_main(
    video_source: str,
    output_path: str,
    duration_seconds: int,
    running_flag: Value,
    done_flag: Value,
    error_flag: Value,
    stop_flag: Value,
) -> None:
    import os
    import time

    import cv2

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cap = None
    writer = None

    try:
        print(f"[RECORD] Opening stream: {video_source}", flush=True)
        cap = cv2.VideoCapture()
        opened = cap.open(video_source, cv2.CAP_FFMPEG)
        if not opened or not cap.isOpened():
            print("[RECORD] Failed to open stream.", flush=True)
            error_flag.value = True
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        running_flag.value = True
        print(f"[RECORD] Recording started → {output_path} ({duration_seconds}s)", flush=True)

        start_time = time.perf_counter()

        while not bool(stop_flag.value):
            if time.perf_counter() - start_time >= duration_seconds:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            writer.write(frame)

        print(f"[RECORD] Recording finished → {output_path}", flush=True)
        done_flag.value = True

    except Exception as exc:
        print(f"[RECORD] Error: {exc}", flush=True)
        error_flag.value = True
    finally:
        running_flag.value = False
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()


def _worker_main(config: PipelineConfig, preview_queue: Queue, ready_flag: Value) -> None:
    try:
        from obs_system.application_module.dummy_application.dummy_app import Application
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


@server.post("/examine_stream/record")
def start_stream_recording(request: StreamRecordRequest):
    global record_process, _current_record_output_path

    if not request.video_source or not request.video_source.strip():
        raise HTTPException(status_code=400, detail="Please provide a stream URL.")

    if request.duration_seconds < 1 or request.duration_seconds > 86400:
        raise HTTPException(status_code=400, detail="duration_seconds must be between 1 and 86400.")

    with record_lock:
        if _is_process_alive(record_process):
            return {"status": "Already recording", "output_path": _current_record_output_path}

        _reset_record_state()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(request.output_dir) / f"record_{timestamp}.mp4")
        _current_record_output_path = output_path

        record_process = Process(
            target=_record_worker_main,
            args=(
                request.video_source.strip(),
                output_path,
                request.duration_seconds,
                record_running,
                record_done,
                record_error,
                record_stop_flag,
            ),
        )
        record_process.start()

    return {
        "status": "Recording started",
        "output_path": output_path,
        "duration_seconds": request.duration_seconds,
    }


@server.get("/examine_stream/record/status")
def get_record_status():
    return {
        "running": bool(record_running.value) or (_is_process_alive(record_process) and not bool(record_done.value) and not bool(record_error.value)),
        "done": bool(record_done.value),
        "error": bool(record_error.value),
        "output_path": _current_record_output_path,
    }


@server.post("/examine_stream/record/stop")
def stop_stream_recording():
    with record_lock:
        if not _is_process_alive(record_process):
            return {"status": "Not recording"}
        _stop_record_worker()

    return {"status": "Recording stopped", "output_path": _current_record_output_path}


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
