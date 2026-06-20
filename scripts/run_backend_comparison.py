from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import cv2
import numpy as np

from contextlib import ExitStack
from pathlib import Path
from types import MethodType
from typing import Any

from obs_system.application_module.dummy_application.dummy_app import Application
from obs_system.application_module.dummy_application.pipeline_config import DEFAULT_BENCH_LABELS, PipelineConfig
from obs_system.detection_module.interface.factory import StreamerFactory
from obs_system.utils.global_config import CONF_THR, NMS_IOU
from obs_system.utils.benchmarking.backend_benchmark import (
    JetsonSampler,
    build_hardware_summary,
    count_data_rows,
    dump_json,
    fmt_opt,
    latency_summary,
    probe_model_backend,
    read_cbor_lines,
    read_csv_rows,
    summarize_stage_latency,
)
from obs_system.utils.appraisal import frame_list, perf
from obs_system.utils.logger import get_logger
from ultralytics.utils import DEFAULT_CFG


logger = get_logger("obs_system." + __name__)

DEFAULT_PT_MODEL = "assets/compressed_models/yolov8s.pt"
DEFAULT_ONNX_MODEL = "assets/compressed_models/yolov8s.onnx"
DEFAULT_ENGINE_MODEL = "assets/compressed_models/yolov8s.engine"
MQTT_ARCHIVE_PATH = Path("assets/mqtt/saved_publishes.cbor")
HAZARD_CSV_PATH = Path("assets/hazard_events/hazard_events.csv")
DEFAULT_FRAME_CAP = 1500
DEFAULT_VIDEO_SECONDS_CAP = 60.0

# Per-frame stage keys present in perf_frames.csv for both regular and tile paths
STAGE_LATENCY_KEYS = [
    "frame_read_ms",
    "roi_ms",
    "mog2_ms",
    "defish_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "nms_ms",
    "tracking_ms",
    "hazard_logic_ms",
    "preview_encode_ms",
    "mqtt_ms",
    "event_saving_ms",
    "total_ms",
]

# Additional per-frame columns written only by the tiled inference path.
# metric_distribution returns None when absent, so these are silently omitted for non-tile runs.
TILE_STAGE_KEYS = [
    "tiles_total",
    "tiles_submitted",
    "tile_skip_rate",
]

# Display names for the stage latency breakdown table
STAGE_DISPLAY_NAMES: dict[str, str] = {
    "frame_read_ms": "Frame read",
    "roi_ms": "ROI crop",
    "mog2_ms": "MOG2 subtraction",
    "defish_ms": "Defish / FEP",
    "preprocess_ms": "Preprocess (YOLO)",
    "inference_ms": "Inference (YOLO)",
    "postprocess_ms": "Postprocess (YOLO)",
    "nms_ms": "NMS",
    "tracking_ms": "ByteTracker",
    "hazard_logic_ms": "Hazard logic",
    "preview_encode_ms": "Preview encode",
    "mqtt_ms": "MQTT publish",
    "event_saving_ms": "Event saving",
    "total_ms": "TOTAL",
    "tiles_total": "Tiles/frame (total)",
    "tiles_submitted": "Tiles/frame (submitted)",
    "tile_skip_rate": "Tile skip rate",
}


def _infer_video_type(source: str) -> str:
    src = (source or "").lower()
    for keyword in ("fisheye", "panorama", "highway", "detrac", "edi", "rectilinear", "ptz"):
        if keyword in src:
            return keyword
    return "unknown"


class _FrameBudgetDataset:
    def __init__(self, dataset: Any, max_frames: int) -> None:
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_max_frames", max(1, int(max_frames)))
        object.__setattr__(self, "_iterator", None)
        object.__setattr__(self, "_remaining", max(1, int(max_frames)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_dataset", "_max_frames", "_iterator", "_remaining"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._dataset, name, value)

    def __iter__(self) -> "_FrameBudgetDataset":
        object.__setattr__(self, "_iterator", iter(self._dataset))
        object.__setattr__(self, "_remaining", self._max_frames)
        return self

    def __next__(self) -> tuple[list[Any], list[Any], list[Any]]:
        if self._remaining <= 0:
            raise StopIteration

        if self._iterator is None:
            object.__setattr__(self, "_iterator", iter(self._dataset))

        batch_payload = next(self._iterator)
        paths, im0s, labels = batch_payload
        paths = list(paths) if isinstance(paths, (list, tuple)) else [paths]
        im0s = list(im0s) if isinstance(im0s, (list, tuple)) else [im0s]
        labels = list(labels) if isinstance(labels, (list, tuple)) else [labels]

        batch_count = len(im0s)
        if batch_count <= self._remaining:
            object.__setattr__(self, "_remaining", self._remaining - batch_count)
            return batch_payload

        truncated = (
            paths[:self._remaining],
            im0s[:self._remaining],
            labels[:self._remaining],
        )
        object.__setattr__(self, "_remaining", 0)
        return truncated


def _build_streamer(model_name: str, model_path: Path, use_tensorrt: bool):
    factory = StreamerFactory(cfg=DEFAULT_CFG, overrides={}, callbacks=None)
    streamer, _ = factory.create(
        model_name=model_name,
        path_to_load=model_path,
        use_tensorrt=use_tensorrt,
        opt="tracking",
    )
    return streamer


def _configure_streamer_args(streamer: Any, config: PipelineConfig, run_dir: Path) -> None:
    streamer.args.show = False
    streamer.args.verbose = bool(config.verbose)
    streamer.args.save = bool(config.save)
    streamer.args.roi = bool(config.roi)
    streamer.args.half = bool(config.half)
    streamer.args.bench = bool(config.bench)
    streamer.args.bench_labels = config.bench_labels
    streamer.args.plot_performance = True
    streamer.args.only_FPS = True
    streamer.args.stream_limit_hours = float(config.stream_limit_hours)
    streamer.args.preview_max_width = int(config.preview_max_width)
    streamer.args.preview_jpeg_quality = int(config.preview_jpeg_quality)
    streamer.args.preview_fps = float(config.preview_fps)
    streamer.args.perf_log = str(run_dir / "perf_log.csv")
    streamer.args.perf_log_frames = str(run_dir / "perf_frames.csv")
    streamer.args.perf_timeline = str(run_dir / "perf_timeline.jsonl")
    streamer.args.perf_log_flush_every = 64
    streamer.args.perf_frame_log_flush_every = 128
    streamer.args.perf_timeline_flush_every = 128
    streamer.args.jetson_profile = bool(config.jetson_profile)
    streamer.args.jetson_hazard_scale = float(config.jetson_hazard_scale)
    streamer.args.jetson_cpu_threads = int(config.jetson_cpu_threads)
    streamer.args.force_tiles = bool(config.force_tiles)
    streamer.args.panorama = bool(config.panorama)
    streamer.args.lane_recalibration_interval_frames = int(config.lane_recalibration_interval_frames)

def _estimate_video_frame_cap(video_source: str) -> int:
    frame_cap = int(DEFAULT_FRAME_CAP)
    source = (video_source or "").strip()
    if not source:
        return frame_cap

    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return frame_cap
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    if fps > 0.0:
        time_budget_frames = max(1, int(DEFAULT_VIDEO_SECONDS_CAP * fps))
        return max(1, min(frame_cap, time_budget_frames))

    return frame_cap


def _apply_video_cap(exit_stack: ExitStack, streamer: Any, video_source: str) -> int:
    frame_cap = _estimate_video_frame_cap(video_source)
    original_setup_source = streamer.setup_source

    def setup_source_with_budget(self, source: str) -> None:
        original_setup_source(source)
        self.dataset = _FrameBudgetDataset(self.dataset, max_frames=frame_cap)
        self.run_metrics["frame_cap"] = int(frame_cap)
        self.run_metrics["video_seconds_cap"] = float(DEFAULT_VIDEO_SECONDS_CAP)

    original = getattr(streamer, "setup_source")
    setattr(streamer, "setup_source", MethodType(setup_source_with_budget, streamer))
    exit_stack.callback(lambda: setattr(streamer, "setup_source", original))
    return frame_cap


def _summarize_output_load(
    hazard_rows_before: int,
    mqtt_rows_before: int,
    streamer_metrics: dict[str, Any],
) -> dict[str, Any]:
    hazard_rows = read_csv_rows(HAZARD_CSV_PATH)
    mqtt_messages = read_cbor_lines(MQTT_ARCHIVE_PATH)

    new_hazards = hazard_rows[hazard_rows_before:]
    new_mqtt = mqtt_messages[mqtt_rows_before:]

    hazard_events = len({row.get("event_image", "") for row in new_hazards if row.get("event_image")})
    hazard_crops = sum(1 for row in new_hazards if row.get("crop_image"))

    crop_sizes = []
    crop_counts = []
    for message in new_mqtt:
        items = message.get("items", [])
        crop_counts.append(len(items))
        for item in items:
            crop_sizes.append(len(item.get("img", b"")))

    return {
        "saved_event_count": hazard_events,
        "saved_hazard_rows": len(new_hazards),
        "saved_hazard_crop_count": hazard_crops,
        "mqtt_crop_batch_count": len(new_mqtt),
        "mqtt_no_detection_batch_count": int(streamer_metrics.get("mqtt_no_detection_batches", 0)),
        "avg_crops_per_mqtt_batch": float(np.mean(crop_counts)) if crop_counts else 0.0,
        "avg_crop_jpeg_bytes": float(np.mean(crop_sizes)) if crop_sizes else 0.0,
    }


def _run_single_benchmark(model_path: str, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    model_suffix = Path(model_path).suffix.lower()
    run_name = Path(model_path).stem
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        model_name=model_path,
        video_source=args.video_source,
        type="tracking",
        gui=False,
        mqtt=bool(args.mqtt),
        show=False,
        verbose=bool(args.verbose),
        save=bool(args.save_outputs),
        roi=bool(args.roi),
        roi_profile=args.roi_profile,
        half=True,
        fep=bool(args.fep),
        bench=bool(args.labels_dir),
        bench_labels=args.labels_dir or DEFAULT_BENCH_LABELS,
        use_TRT=(model_suffix == ".engine"),
        plot_perf=True,
        only_FPS=False,
        preview_max_width=960,
        preview_jpeg_quality=70,
        preview_fps=8.0,
        stream_limit_hours=float(args.stream_limit_hours),
        lane_recalibration_interval_frames=int(args.lane_recalibration_interval_frames),
        jetson_profile=bool(args.jetson_profile),
        jetson_hazard_scale=float(args.jetson_hazard_scale),
        jetson_cpu_threads=int(args.jetson_cpu_threads),
        force_tiles=bool(args.force_tiles),
        panorama=bool(args.panorama),
    ).validate()

    perf.reset()
    frame_list.clear()

    app = Application(save=config.save, verbose=config.verbose)
    app.setup_process(config)
    app.setup_logic_module(config)
    app.setup_mqtt() if config.mqtt else None

    model_spec = config.resolve_model()
    model_name = f"{model_spec.name}.{model_spec.kind}"
    try:
        streamer = _build_streamer(
            model_name=model_name,
            model_path=model_spec.path,
            use_tensorrt=bool(config.use_TRT),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{model_path}': {exc}") from exc

    _configure_streamer_args(streamer, config, run_dir)

    app.streamer = streamer
    app.model = streamer.model

    hazard_rows_before = count_data_rows(HAZARD_CSV_PATH)
    mqtt_rows_before = len(read_cbor_lines(MQTT_ARCHIVE_PATH))

    jetson_sampler = JetsonSampler(interval_s=args.jetson_interval)
    elapsed_s = 0.0

    try:
        with ExitStack() as run_exit_stack:
            _apply_video_cap(run_exit_stack, streamer, config.video_source)
            jetson_sampler.start()
            t0 = time.perf_counter()
            streamer(
                source=app.source,
                model=app.model,
                logic_module=app.logic_module,
                mqtt_broker=app.mqtt_publisher,
                producer_flag=None,
                preview_queue=None,
            )
            elapsed_s = time.perf_counter() - t0
    finally:
        jetson_sampler.stop()
        try:
            app.close_app()
        except Exception:
            logger.debug("Application close_app failed during benchmark cleanup", exc_info=True)
        app.cleanup_runtime_resources()

    perf.finalize()
    setup_metrics = perf.results()

    frame_rows = read_csv_rows(run_dir / "perf_frames.csv")
    batch_rows = read_csv_rows(run_dir / "perf_log.csv")
    latency = latency_summary(frame_rows)
    streamer_metrics = dict(streamer.run_metrics)
    hardware = build_hardware_summary(
        batch_rows=batch_rows,
        jetson_samples=jetson_sampler.samples,
        jetson_context=jetson_sampler.static_context,
    )

    detection_metrics = streamer.mp.results() if config.bench and streamer.mp is not None else {}
    output_load = _summarize_output_load(
        hazard_rows_before=hazard_rows_before,
        mqtt_rows_before=mqtt_rows_before,
        streamer_metrics=streamer_metrics,
    )

    frames_emitted = int(streamer_metrics.get("frames_emitted", 0))
    fps = (frames_emitted / elapsed_s) if elapsed_s > 0 else 0.0

    video_type = (args.video_type or "").strip() or _infer_video_type(args.video_source)
    inference_mode = "tiles" if config.force_tiles else ("panorama" if config.panorama else ("fisheye" if config.fep else "standard"))

    stage_latency = summarize_stage_latency(frame_rows, STAGE_LATENCY_KEYS)
    tile_metrics = summarize_stage_latency(frame_rows, TILE_STAGE_KEYS)

    summary = {
        "video_type": video_type,
        "inference_mode": inference_mode,
        "model_path": model_path,
        "model_kind": model_suffix.lstrip("."),
        "run_dir": str(run_dir),
        "confidence_threshold": CONF_THR,
        "nms_iou": NMS_IOU,
        "frame_cap": int(streamer_metrics.get("frame_cap", DEFAULT_FRAME_CAP)),
        "video_seconds_cap": float(streamer_metrics.get("video_seconds_cap", DEFAULT_VIDEO_SECONDS_CAP)),
        "runtime_seconds": elapsed_s,
        "fps": fps,
        **latency,
        "stream_open_seconds": streamer_metrics.get("stream_open_seconds"),
        "first_frame_seconds": streamer_metrics.get("first_frame_seconds"),
        "frames_observed": streamer_metrics.get("frames_observed", 0),
        "frames_emitted": frames_emitted,
        "dropped_frames": streamer_metrics.get("dropped_frames", 0),
        "crop_count": streamer_metrics.get("crop_count", 0),
        "crop_avg_bytes_raw": (
            streamer_metrics["crop_total_bytes"] / streamer_metrics["crop_count"]
            if streamer_metrics.get("crop_count")
            else 0.0
        ),
        "crop_avg_pixels": (
            streamer_metrics["crop_total_pixels"] / streamer_metrics["crop_count"]
            if streamer_metrics.get("crop_count")
            else 0.0
        ),
        "setup_metrics": setup_metrics,
        "stage_latency": stage_latency,
        "tile_metrics": tile_metrics,
        "hardware": hardware,
        "detection_metrics": detection_metrics,
        "output_load": output_load,
    }

    dump_json(run_dir / "summary.json", summary)

    return summary


def _write_markdown_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Backend Comparison",
        "",
        f"- Evaluation cap: up to `{DEFAULT_FRAME_CAP}` frames or `{int(DEFAULT_VIDEO_SECONDS_CAP)}` seconds of source video, whichever is smaller.",
        "",
        "## Performance Overview",
        "",
        "| Video Type | Mode | Model | FPS | E2E avg ms | E2E p50 ms | E2E p95 ms | Infer ms | Preproc ms | Post+Track ms | CPU % | GPU % | RAM MB | VRAM MB | Temp C | Power W | Power Mode | Dropped | mAP | F1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]

    for s in summaries:
        hw = s.get("hardware", {})
        det = s.get("detection_metrics", {})
        sl = s.get("stage_latency", {})

        infer_ms = fmt_opt(sl.get("inference_ms", {}).get("mean_ms") if isinstance(sl.get("inference_ms"), dict) else None)
        preproc_ms = fmt_opt(sl.get("preprocess_ms", {}).get("mean_ms") if isinstance(sl.get("preprocess_ms"), dict) else None)
        post_ms_val = None
        if isinstance(sl.get("postprocess_ms"), dict) and isinstance(sl.get("tracking_ms"), dict):
            post_ms_val = sl["postprocess_ms"]["mean_ms"] + sl["tracking_ms"]["mean_ms"]
        post_ms = fmt_opt(post_ms_val)

        ram_mb = fmt_opt(hw.get("jetson_ram_metric_mean") or hw.get("gpu_mem_used_mb_mean"))
        vram_mb = fmt_opt(hw.get("gpu_mem_used_mb_mean"))

        lines.append(
            "| {vtype} | {mode} | {model} | {fps:.2f} | {avg:.2f} | {p50:.2f} | {p95:.2f} | {infer} | {preproc} | {posttrack} | {cpu} | {gpu} | {ram} | {vram} | {temp} | {power} | {pmode} | {dropped} | {map_} | {f1} |".format(
                vtype=s.get("video_type", "-"),
                mode=s.get("inference_mode", "-"),
                model=Path(s["model_path"]).name,
                fps=s.get("fps", 0.0),
                avg=s.get("avg_latency_ms", 0.0),
                p50=s.get("p50_latency_ms", 0.0),
                p95=s.get("p95_latency_ms", 0.0),
                infer=infer_ms,
                preproc=preproc_ms,
                posttrack=post_ms,
                cpu=fmt_opt(hw.get("cpu_util_mean")),
                gpu=fmt_opt(hw.get("gpu_util_mean")),
                ram=ram_mb,
                vram=vram_mb,
                temp=fmt_opt(hw.get("temperature_c_mean")),
                power=fmt_opt(hw.get("power_w_mean")),
                pmode=fmt_opt(hw.get("power_mode")),
                dropped=s.get("dropped_frames", 0),
                map_=fmt_opt(det.get("mAP")),
                f1=fmt_opt(det.get("F1")),
            )
        )

    # Stage latency breakdown — one row per pipeline stage, one column per model run
    model_names = [Path(s["model_path"]).name for s in summaries]
    lines += [
        "",
        "## Stage Latency Breakdown (mean ms per frame)",
        "",
        "| Stage | " + " | ".join(model_names) + " |",
        "| --- |" + " ---: |" * len(summaries),
    ]
    for key in STAGE_LATENCY_KEYS:
        display = STAGE_DISPLAY_NAMES.get(key, key)
        cells = []
        for s in summaries:
            dist = s.get("stage_latency", {}).get(key)
            cells.append(f"{dist['mean_ms']:.2f}" if isinstance(dist, dict) else "-")
        lines.append(f"| {display} | " + " | ".join(cells) + " |")

    # Tile metrics section — only rendered when at least one run has tile data
    has_tile_data = any(bool(s.get("tile_metrics")) for s in summaries)
    if has_tile_data:
        lines += [
            "",
            "## Tile Metrics",
            "",
            "| Metric | " + " | ".join(model_names) + " |",
            "| --- |" + " ---: |" * len(summaries),
        ]
        for key in TILE_STAGE_KEYS:
            display = STAGE_DISPLAY_NAMES.get(key, key)
            cells = []
            for s in summaries:
                dist = s.get("tile_metrics", {}).get(key)
                cells.append(f"{dist['mean_ms']:.3f}" if isinstance(dist, dict) else "-")
            lines.append(f"| {display} | " + " | ".join(cells) + " |")

    # Detection quality + output load footer
    lines += [
        "",
        "## Detection Quality & Output Load",
        "",
        "| Model | Precision | Recall | mAP | Saved Events | MQTT Batches | Avg Crop JPEG B |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        det = s.get("detection_metrics", {})
        ol = s.get("output_load", {})
        lines.append(
            "| {model} | {prec} | {rec} | {map_} | {events} | {mqtt} | {crop:.2f} |".format(
                model=Path(s["model_path"]).name,
                prec=fmt_opt(det.get("Precision")),
                rec=fmt_opt(det.get("Recall")),
                map_=fmt_opt(det.get("mAP")),
                events=ol.get("saved_event_count", 0),
                mqtt=ol.get("mqtt_crop_batch_count", 0),
                crop=ol.get("avg_crop_jpeg_bytes", 0.0),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _benchmark_result_path(output_dir: Path, model_path: str) -> Path:
    return output_dir / Path(model_path).stem / "benchmark_result.json"


def _child_argv(model_path: str, result_path: Path) -> list[str]:
    parent_args = [
        arg for arg in sys.argv[1:]
        if arg not in {"--run-single-benchmark", "--benchmark-result-path"}
    ]
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *parent_args,
        "--run-single-benchmark", model_path,
        "--benchmark-result-path", str(result_path),
    ]


def _run_single_benchmark_child(args: argparse.Namespace, output_dir: Path) -> int:
    model_path = str(args.run_single_benchmark)
    result_path = Path(args.benchmark_result_path)
    try:
        summary = _run_single_benchmark(model_path=model_path, args=args, output_dir=output_dir)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        logger.exception("Benchmark failed for model '%s'", model_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({"model_path": model_path, "status": "failed", "error": str(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 1


def _run_benchmark_subprocess(model_path: str, args: argparse.Namespace, output_dir: Path) -> dict[str, Any] | None:
    result_path = _benchmark_result_path(output_dir, model_path)
    if result_path.exists():
        result_path.unlink()

    cmd = _child_argv(model_path, result_path)
    logger.info("Running benchmark in subprocess: %s", Path(model_path).name)
    completed = subprocess.run(cmd, cwd=Path.cwd())

    if result_path.exists():
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[ERROR] {model_path}  —  Invalid result JSON: {exc}")
            return None
    else:
        print(f"[ERROR] {model_path}  —  subprocess exited with code {completed.returncode} before writing a result file.")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run PT/ONNX/ENGINE backend comparison on the same video source. "
            "On WSL / non-Jetson hosts the .engine backend is automatically skipped "
            "when TensorRT is not installed; missing model files are also skipped. "
            "Recorded videos are capped to the first standardized evaluation window."
        )
    )
    parser.add_argument("--video-source", required=True, help="Input video path or stream URL.")
    parser.add_argument("--video-type", default="", help="Label for this video type in the report (e.g. rectilinear, fisheye, highway). Auto-detected from the path when omitted.")
    parser.add_argument("--labels-dir", default="", help="Optional YOLO label directory for Precision/Recall/F1/mAP.")
    parser.add_argument("--output-dir", default="assets/backend_comparison", help="Directory where run summaries/logs are written.")
    parser.add_argument("--pt-model", default=DEFAULT_PT_MODEL, help="Path to the PT model.")
    parser.add_argument("--onnx-model", default=DEFAULT_ONNX_MODEL, help="Path to the ONNX model.")
    parser.add_argument("--engine-model", default=DEFAULT_ENGINE_MODEL, help="Path to the TensorRT engine.")
    parser.add_argument("--roi", action=argparse.BooleanOptionalAction, default=True, help="Enable ROI cropping.")
    parser.add_argument("--roi-profile", default="", help="Optional ROI profile key from obs_system/utils/roi_profiles.json.")
    parser.add_argument("--fep", action=argparse.BooleanOptionalAction, default=False, help="Enable fisheye projection.")
    parser.add_argument("--force-tiles", action=argparse.BooleanOptionalAction, default=False, help="Force tiled inference regardless of image dimensions.")
    parser.add_argument("--panorama", action=argparse.BooleanOptionalAction, default=False, help="Input is an equirectangular panorama.")
    parser.add_argument("--lane-recalibration-interval-frames", type=int, default=0, help="Rerun lane calibration after this many frames. Use 0 to disable.")
    parser.add_argument("--mqtt", action=argparse.BooleanOptionalAction, default=False, help="Enable MQTT publishing during the benchmark.")
    parser.add_argument("--save-outputs", action=argparse.BooleanOptionalAction, default=False, help="Enable saving rendered outputs during the benchmark.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose streamer logging.")
    parser.add_argument("--stream-limit-hours", type=float, default=0.0, help="Live-stream runtime cap in hours. Use 0 to disable.")
    parser.add_argument("--jetson-interval", type=float, default=1.0, help="Sampling interval for Jetson telemetry in seconds.")
    parser.add_argument("--jetson-profile", action=argparse.BooleanOptionalAction, default=False, help="Enable the Jetson-optimized execution branch.")
    parser.add_argument("--jetson-hazard-scale", type=float, default=1.0, help="Scale factor for Jetson hazard-mask processing. Use values below 1.0 to trade a small amount of precision for speed.")
    parser.add_argument("--jetson-cpu-threads", type=int, default=0, help="CPU thread cap for the Jetson-optimized execution branch. Use 0 to keep runtime defaults.")
    parser.add_argument("--run-single-benchmark", default="", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-result-path", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_single_benchmark:
        if not args.benchmark_result_path:
            raise SystemExit("--benchmark-result-path is required with --run-single-benchmark")
        raise SystemExit(_run_single_benchmark_child(args, output_dir))

    candidate_paths = [args.pt_model, args.onnx_model, args.engine_model]

    # Filter to backends runnable in this environment.
    # On WSL / non-Jetson hosts this drops the .engine backend when TensorRT
    # is not installed, and any path whose file doesn't exist.
    runnable: list[str] = []
    for mp in candidate_paths:
        ok, reason = probe_model_backend(mp)
        if ok:
            runnable.append(mp)
        else:
            print(f"[SKIP] {mp}  —  {reason}")

    if not runnable:
        print("No runnable model backends found. Exiting.")
        return

    summaries: list[dict[str, Any]] = []
    for path in runnable:
        result = _run_benchmark_subprocess(model_path=path, args=args, output_dir=output_dir)
        if result is not None and "fps" in result:
            summaries.append(result)

    if not summaries:
        print("All benchmarks failed. Check model files and backend availability.")
        return

    summary_json = output_dir / "comparison_summary.json"
    dump_json(summary_json, summaries)

    _write_markdown_summary(output_dir / "comparison_summary.md", summaries)


if __name__ == "__main__":
    main()
