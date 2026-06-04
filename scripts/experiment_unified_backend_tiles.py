from __future__ import annotations

import argparse
import csv
import platform
import time

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator

import numpy as np

from obs_system.application_module.dummy_application.dummy_app import Application
from obs_system.application_module.dummy_application.pipeline_config import (
    DEFAULT_BENCH_LABELS,
    PipelineConfig,
)
from obs_system.detection_module.interface.factory import StreamerFactory
from obs_system.utils.appraisal import frame_list, perf
from obs_system.utils.benchmarking.backend_benchmark import (
    JetsonSampler,
    build_hardware_summary,
    dump_json,
    mean_metric,
    probe_model_backend,
    read_csv_rows,
)
from obs_system.utils.logger import get_logger
from ultralytics.utils import DEFAULT_CFG


logger = get_logger("obs_system." + __name__)

DEFAULT_VIDEO_SOURCE = "samples/MVI_39401.mp4"
DEFAULT_PT_MODEL = "assets/compressed_models/mixed_dataset_trained_yolov8s.pt"
DEFAULT_ONNX_MODEL = "assets/compressed_models/yolov8s.onnx"
DEFAULT_ENGINE_MODEL = "assets/compressed_models/edi_jetson_model.engine"
DEFAULT_OUTPUT_DIR = "experiment_results/backend_tiles_benchmark"

STAGE_BATCH_COLUMNS = {
    "frame_read_ms": "frame_read_ms_per_frame",
    "roi_ms": "roi_ms_per_frame",
    "mog2_ms": "mog2_ms_per_frame",
    "defish_ms": "defish_ms_per_frame",
    "preprocess_ms": "preprocess_ms_per_frame",
    "inference_ms": "inference_ms_per_frame",
    "postprocess_ms": "postprocess_ms_per_frame",
    "nms_ms": "nms_ms_per_frame",
    "tracking_ms": "tracking_ms_per_frame",
    "hazard_logic_ms": "hazard_logic_ms_per_frame",
    "preview_encode_ms": "preview_encode_ms_per_frame",
    "mqtt_ms": "mqtt_ms_per_frame",
    "event_saving_ms": "event_saving_ms_per_frame",
    "total_pipeline_ms": "total_ms_per_frame",
}


@dataclass(frozen=True)
class BackendSpec:
    key: str
    model_path: str
    use_tensorrt: bool


@dataclass(frozen=True)
class RunSpec:
    key: str
    backend: str
    model_path: str
    use_tensorrt: bool
    tile_mode: str
    profile: str
    mqtt: bool
    save_outputs: bool
    purpose: str


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
    streamer.args.only_FPS = False
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


@contextmanager
def _temporary_attr(obj: Any, attr: str, value: Any) -> Iterator[None]:
    sentinel = object()
    original = getattr(obj, attr, sentinel)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        if original is sentinel:
            delattr(obj, attr)
        else:
            setattr(obj, attr, original)


@contextmanager
def _temporary_method(obj: Any, method_name: str, fn: Callable[..., Any]) -> Iterator[None]:
    original = getattr(obj, method_name)
    setattr(obj, method_name, MethodType(fn, obj))
    try:
        yield
    finally:
        setattr(obj, method_name, original)


def _machine_type() -> str:
    arch = platform.machine()
    if arch in {"x86_64", "AMD64"}:
        return "desktop"
    if arch == "aarch64":
        return "jetson"
    return "unknown_arm"


def _bool_from_row(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _count_detections(preds: Any) -> int:
    detections = getattr(preds, "sv_detections", None)
    if detections is not None:
        try:
            return int(len(detections))
        except Exception:
            pass

    boxes = getattr(preds, "boxes", None)
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None:
        return 0
    try:
        return int(xyxy.shape[0])
    except Exception:
        return 0


def _install_detection_counter(streamer: Any) -> ExitStack:
    exit_stack = ExitStack()
    streamer.run_metrics.setdefault("detection_total_count", 0)
    streamer.run_metrics.setdefault("detection_frame_count", 0)

    original_postprocess = streamer.postprocess

    def wrapped_postprocess(self, preds: Any, orig_image: Any) -> Any:
        out = original_postprocess(preds, orig_image)
        self.run_metrics["detection_total_count"] += _count_detections(out)
        self.run_metrics["detection_frame_count"] += 1
        return out

    exit_stack.enter_context(_temporary_method(streamer, "postprocess", wrapped_postprocess))
    return exit_stack


def _candidate_backends(args: argparse.Namespace) -> list[BackendSpec]:
    candidates = [
        BackendSpec(key="pt", model_path=args.pt_model, use_tensorrt=False),
        BackendSpec(key="onnx", model_path=args.onnx_model, use_tensorrt=False),
        BackendSpec(key="engine", model_path=args.engine_model, use_tensorrt=True),
    ]

    runnable: list[BackendSpec] = []
    for spec in candidates:
        ok, reason = probe_model_backend(spec.model_path)
        if ok:
            runnable.append(spec)
        else:
            logger.warning("Skipping backend %s: %s", spec.key, reason)
    return runnable


def _select_desktop_full_pipeline_backends(
    backends: list[BackendSpec],
    limit: int,
) -> list[BackendSpec]:
    if limit <= 0:
        return []

    by_key = {spec.key: spec for spec in backends}
    preferred_order = ["onnx", "engine", "pt"]
    selected: list[BackendSpec] = []
    for key in preferred_order:
        spec = by_key.get(key)
        if spec is not None:
            selected.append(spec)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for spec in backends:
            if spec not in selected:
                selected.append(spec)
            if len(selected) >= limit:
                break

    return selected


def _build_run_specs(args: argparse.Namespace, backends: list[BackendSpec]) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for spec in backends:
        runs.append(
            RunSpec(
                key=f"{spec.key}_full_frame_lean",
                backend=spec.key,
                model_path=spec.model_path,
                use_tensorrt=spec.use_tensorrt,
                tile_mode="full_frame",
                profile="lean_benchmark",
                mqtt=False,
                save_outputs=False,
                purpose="Unified streamer benchmark without tiles; MQTT/save disabled for backend comparison.",
            )
        )
        runs.append(
            RunSpec(
                key=f"{spec.key}_tiles_lean",
                backend=spec.key,
                model_path=spec.model_path,
                use_tensorrt=spec.use_tensorrt,
                tile_mode="tiles",
                profile="lean_benchmark",
                mqtt=False,
                save_outputs=False,
                purpose="Unified streamer benchmark with forced tiling; MQTT/save disabled for backend comparison.",
            )
        )

    if _machine_type() == "desktop":
        desktop_specs = _select_desktop_full_pipeline_backends(backends, int(args.desktop_full_pipeline_cases))
        for spec in desktop_specs:
            runs.append(
                RunSpec(
                    key=f"{spec.key}_full_frame_desktop_full",
                    backend=spec.key,
                    model_path=spec.model_path,
                    use_tensorrt=spec.use_tensorrt,
                    tile_mode="full_frame",
                    profile="desktop_full_pipeline",
                    mqtt=bool(args.full_pipeline_mqtt),
                    save_outputs=bool(args.full_pipeline_save_outputs),
                    purpose="Representative desktop full-pipeline run; not expanded to the whole backend x tiling matrix.",
                )
            )

    return runs


def _build_config(run_spec: RunSpec, args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        model_name=run_spec.model_path,
        video_source=args.video_source,
        type="tracking",
        gui=False,
        mqtt=bool(run_spec.mqtt),
        show=False,
        verbose=bool(args.verbose),
        save=bool(run_spec.save_outputs),
        roi=bool(args.roi),
        half=True,
        fep=bool(args.fep),
        bench=bool(args.labels_dir),
        bench_labels=args.labels_dir or DEFAULT_BENCH_LABELS,
        use_TRT=bool(run_spec.use_tensorrt),
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
        force_tiles=(run_spec.tile_mode == "tiles"),
        panorama=False,
    ).validate()


def _stage_means(batch_rows: list[dict[str, str]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for public_name, csv_key in STAGE_BATCH_COLUMNS.items():
        out[public_name] = mean_metric(batch_rows, csv_key)
    return out


def _motion_gate_counts(
    frame_rows: list[dict[str, str]],
    batch_rows: list[dict[str, str]],
) -> tuple[int, int]:
    if frame_rows:
        processed = sum(1 for row in frame_rows if _bool_from_row(row.get("motion_passed")))
        skipped = sum(1 for row in frame_rows if not _bool_from_row(row.get("motion_passed")))
        return processed, skipped

    processed = 0
    total = 0
    for row in batch_rows:
        try:
            processed += int(float(row.get("frames_inferred", 0) or 0))
        except ValueError:
            pass
        try:
            total += int(float(row.get("frames_in_batch", 0) or 0))
        except ValueError:
            pass
    return processed, max(total - processed, 0)


def _summarize_run(
    run_spec: RunSpec,
    args: argparse.Namespace,
    run_dir: Path,
    config: PipelineConfig,
    elapsed_s: float,
    streamer: Any,
    jetson_sampler: JetsonSampler,
) -> dict[str, Any]:
    batch_rows = read_csv_rows(run_dir / "perf_log.csv")
    frame_rows = read_csv_rows(run_dir / "perf_frames.csv")
    stage_means = _stage_means(batch_rows)
    hardware = build_hardware_summary(
        batch_rows=batch_rows,
        jetson_samples=jetson_sampler.samples,
        jetson_context=jetson_sampler.static_context,
    )

    frames_observed = int(streamer.run_metrics.get("frames_observed", 0))
    frames_emitted = int(streamer.run_metrics.get("frames_emitted", 0))
    detection_total_count = int(streamer.run_metrics.get("detection_total_count", 0))
    detection_frame_count = int(streamer.run_metrics.get("detection_frame_count", 0))
    fps = (frames_emitted / elapsed_s) if elapsed_s > 0 else 0.0
    motion_processed, motion_skipped = _motion_gate_counts(frame_rows, batch_rows)

    total_latency_values = []
    for row in batch_rows:
        raw = row.get("total_ms_per_frame")
        if raw in (None, "", "nan"):
            continue
        try:
            total_latency_values.append(float(raw))
        except ValueError:
            continue

    return {
        "run_key": run_spec.key,
        "backend": run_spec.backend,
        "tile_mode": run_spec.tile_mode,
        "profile": run_spec.profile,
        "purpose": run_spec.purpose,
        "status": "completed",
        "video_source": args.video_source,
        "model_path": run_spec.model_path,
        "run_dir": str(run_dir),
        "machine_type": _machine_type(),
        "mqtt_enabled": bool(config.mqtt),
        "save_outputs": bool(config.save),
        "roi_enabled": bool(config.roi),
        "fep_enabled": bool(config.fep),
        "use_tensorrt": bool(config.use_TRT),
        "fps": float(fps),
        "runtime_seconds": float(elapsed_s),
        "total_latency_ms_per_frame": stage_means.get("total_pipeline_ms"),
        "p50_frame_latency_ms": float(np.percentile(total_latency_values, 50)) if total_latency_values else None,
        "p95_frame_latency_ms": float(np.percentile(total_latency_values, 95)) if total_latency_values else None,
        "frame_read_ms_per_frame": stage_means.get("frame_read_ms"),
        "roi_ms_per_frame": stage_means.get("roi_ms"),
        "mog2_ms_per_frame": stage_means.get("mog2_ms"),
        "defish_ms_per_frame": stage_means.get("defish_ms"),
        "preprocess_ms_per_frame": stage_means.get("preprocess_ms"),
        "inference_ms_per_frame": stage_means.get("inference_ms"),
        "postprocess_ms_per_frame": stage_means.get("postprocess_ms"),
        "nms_ms_per_frame": stage_means.get("nms_ms"),
        "tracking_ms_per_frame": stage_means.get("tracking_ms"),
        "hazard_logic_ms_per_frame": stage_means.get("hazard_logic_ms"),
        "mqtt_ms_per_frame": stage_means.get("mqtt_ms"),
        "event_saving_ms_per_frame": stage_means.get("event_saving_ms"),
        "preview_encode_ms_per_frame": stage_means.get("preview_encode_ms"),
        "frames_observed": frames_observed,
        "frames_emitted": frames_emitted,
        "dropped_frames": int(streamer.run_metrics.get("dropped_frames", 0)),
        "motion_gated_frames_processed": int(motion_processed),
        "motion_gated_frames_skipped": int(motion_skipped),
        "detections_total": detection_total_count,
        "detections_per_observed_frame": (
            float(detection_total_count) / float(frames_observed) if frames_observed > 0 else 0.0
        ),
        "detections_per_emitted_frame": (
            float(detection_total_count) / float(frames_emitted) if frames_emitted > 0 else 0.0
        ),
        "detection_frame_count": detection_frame_count,
        "gpu_util_mean": hardware.get("gpu_util_mean"),
        "cpu_util_mean": hardware.get("cpu_util_mean"),
        "gpu_mem_used_mb_mean": hardware.get("gpu_mem_used_mb_mean"),
        "gpu_mem_total_mb_mean": hardware.get("gpu_mem_total_mb_mean"),
        "temperature_c_mean": hardware.get("temperature_c_mean"),
        "power_w_mean": hardware.get("power_w_mean"),
        "emc_frequency_mhz_mean": hardware.get("emc_frequency_mhz_mean"),
        "power_mode": hardware.get("power_mode"),
        "stream_open_seconds": streamer.run_metrics.get("stream_open_seconds"),
        "first_frame_seconds": streamer.run_metrics.get("first_frame_seconds"),
    }


def _run_single_case(
    run_spec: RunSpec,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    run_dir = output_dir / run_spec.key
    run_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(run_spec, args)
    perf.reset()
    frame_list.clear()

    app = Application(save=config.save, verbose=config.verbose)
    streamer = None
    jetson_sampler = JetsonSampler(interval_s=float(args.jetson_interval))
    elapsed_s = 0.0

    try:
        app.setup_process(config)
        app.setup_logic_module(config)
        if config.mqtt:
            app.setup_mqtt()

        model_spec = config.resolve_model()
        model_name = f"{model_spec.name}.{model_spec.kind}"
        streamer = _build_streamer(
            model_name=model_name,
            model_path=model_spec.path,
            use_tensorrt=bool(config.use_TRT),
        )
        _configure_streamer_args(streamer, config, run_dir)

        app.streamer = streamer
        app.model = streamer.model

        with ExitStack() as exit_stack:
            exit_stack.enter_context(_install_detection_counter(streamer))
            if run_spec.tile_mode == "full_frame":
                exit_stack.enter_context(_temporary_attr(streamer, "force_streaming_no_tiles", True))
                exit_stack.enter_context(_temporary_attr(streamer.args, "force_tiles", False))
            else:
                exit_stack.enter_context(_temporary_attr(streamer, "force_streaming_no_tiles", False))
                exit_stack.enter_context(_temporary_attr(streamer.args, "force_tiles", True))

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
            logger.debug("Application close_app failed during backend-tiles cleanup", exc_info=True)
        app.cleanup_runtime_resources()

    perf.finalize()

    if streamer is None:
        raise RuntimeError("Streamer setup failed before execution started.")

    return _summarize_run(
        run_spec=run_spec,
        args=args,
        run_dir=run_dir,
        config=config,
        elapsed_s=elapsed_s,
        streamer=streamer,
        jetson_sampler=jetson_sampler,
    )


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_key",
        "backend",
        "tile_mode",
        "profile",
        "purpose",
        "status",
        "video_source",
        "model_path",
        "run_dir",
        "machine_type",
        "mqtt_enabled",
        "save_outputs",
        "roi_enabled",
        "fep_enabled",
        "use_tensorrt",
        "fps",
        "runtime_seconds",
        "total_latency_ms_per_frame",
        "p50_frame_latency_ms",
        "p95_frame_latency_ms",
        "frame_read_ms_per_frame",
        "roi_ms_per_frame",
        "mog2_ms_per_frame",
        "defish_ms_per_frame",
        "preprocess_ms_per_frame",
        "inference_ms_per_frame",
        "postprocess_ms_per_frame",
        "nms_ms_per_frame",
        "tracking_ms_per_frame",
        "hazard_logic_ms_per_frame",
        "mqtt_ms_per_frame",
        "event_saving_ms_per_frame",
        "preview_encode_ms_per_frame",
        "frames_observed",
        "frames_emitted",
        "dropped_frames",
        "motion_gated_frames_processed",
        "motion_gated_frames_skipped",
        "detections_total",
        "detections_per_observed_frame",
        "detections_per_emitted_frame",
        "detection_frame_count",
        "gpu_util_mean",
        "cpu_util_mean",
        "gpu_mem_used_mb_mean",
        "gpu_mem_total_mb_mean",
        "temperature_c_mean",
        "power_w_mean",
        "emc_frequency_mhz_mean",
        "power_mode",
        "stream_open_seconds",
        "first_frame_seconds",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Unified Streamer Backend + Tiling Benchmark",
        "",
        "| Run | Backend | Tiles | Profile | FPS | Total ms/frame | Detections/frame | Motion processed | Motion skipped |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        if row.get("status") != "completed":
            lines.append(
                f"| {row.get('run_key')} | {row.get('backend')} | {row.get('tile_mode')} | {row.get('profile')} | FAILED | FAILED | FAILED | FAILED | FAILED |"
            )
            continue

        lines.append(
            "| {run_key} | {backend} | {tile_mode} | {profile} | {fps:.2f} | {total_ms:.2f} | {det_per_frame:.2f} | {motion_processed} | {motion_skipped} |".format(
                run_key=row.get("run_key"),
                backend=row.get("backend"),
                tile_mode=row.get("tile_mode"),
                profile=row.get("profile"),
                fps=float(row.get("fps", 0.0)),
                total_ms=float(row.get("total_latency_ms_per_frame", 0.0) or 0.0),
                det_per_frame=float(row.get("detections_per_observed_frame", 0.0) or 0.0),
                motion_processed=int(row.get("motion_gated_frames_processed", 0) or 0),
                motion_skipped=int(row.get("motion_gated_frames_skipped", 0) or 0),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the unified streamer with and without tiles across PT, ONNX, "
            "and TensorRT backends, while keeping a small desktop-only full-pipeline subset."
        )
    )
    parser.add_argument("--video-source", default=DEFAULT_VIDEO_SOURCE, help="Input video path or stream URL.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where summaries and run logs are written.")
    parser.add_argument("--labels-dir", default="", help="Optional YOLO label directory for accuracy metrics.")
    parser.add_argument("--pt-model", default=DEFAULT_PT_MODEL, help="Path to the PT model.")
    parser.add_argument("--onnx-model", default=DEFAULT_ONNX_MODEL, help="Path to the ONNX model.")
    parser.add_argument("--engine-model", default=DEFAULT_ENGINE_MODEL, help="Path to the TensorRT engine.")
    parser.add_argument("--roi", action=argparse.BooleanOptionalAction, default=True, help="Enable ROI cropping.")
    parser.add_argument("--fep", action=argparse.BooleanOptionalAction, default=False, help="Enable fisheye reprojection.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose streamer logging.")
    parser.add_argument("--stream-limit-hours", type=float, default=0.0, help="Live-stream runtime cap in hours. Use 0 to disable.")
    parser.add_argument("--lane-recalibration-interval-frames", type=int, default=0, help="For live streams, rerun lane calibration after this many frames. Use 0 to disable.")
    parser.add_argument("--jetson-interval", type=float, default=1.0, help="Sampling interval for Jetson telemetry in seconds.")
    parser.add_argument("--jetson-profile", action=argparse.BooleanOptionalAction, default=True, help="Enable the Jetson-optimized execution branch.")
    parser.add_argument("--jetson-hazard-scale", type=float, default=1.0, help="Scale factor for Jetson hazard-mask processing.")
    parser.add_argument("--jetson-cpu-threads", type=int, default=0, help="CPU thread cap for the Jetson execution branch. Use 0 for runtime defaults.")
    parser.add_argument("--desktop-full-pipeline-cases", type=int, default=2, help="How many representative desktop full-pipeline runs to add outside the lean backend x tiling matrix.")
    parser.add_argument("--full-pipeline-mqtt", action=argparse.BooleanOptionalAction, default=True, help="Enable MQTT for the representative desktop full-pipeline subset.")
    parser.add_argument("--full-pipeline-save-outputs", action=argparse.BooleanOptionalAction, default=True, help="Enable saving for the representative desktop full-pipeline subset.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if "://" not in args.video_source and not Path(args.video_source).exists():
        raise SystemExit(f"Video source does not exist: {args.video_source}")

    backends = _candidate_backends(args)
    if not backends:
        raise SystemExit("No runnable backends found. Check model paths and TensorRT availability.")

    run_specs = _build_run_specs(args, backends)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for run_spec in run_specs:
        try:
            logger.info(
                "Running backend benchmark: backend=%s tiles=%s profile=%s",
                run_spec.backend,
                run_spec.tile_mode,
                run_spec.profile,
            )
            summaries.append(_run_single_case(run_spec=run_spec, args=args, output_dir=output_dir))
        except Exception as exc:
            logger.exception("Benchmark run failed: %s", run_spec.key)
            summaries.append(
                {
                    "run_key": run_spec.key,
                    "backend": run_spec.backend,
                    "tile_mode": run_spec.tile_mode,
                    "profile": run_spec.profile,
                    "purpose": run_spec.purpose,
                    "status": "failed",
                    "video_source": args.video_source,
                    "model_path": run_spec.model_path,
                    "run_dir": str(output_dir / run_spec.key),
                    "error": str(exc),
                }
            )

    summary_csv = output_dir / "backend_tiles_summary.csv"
    summary_json = output_dir / "backend_tiles_summary.json"
    summary_md = output_dir / "backend_tiles_summary.md"
    _write_summary_csv(summary_csv, summaries)
    dump_json(summary_json, summaries)
    _write_summary_markdown(summary_md, summaries)

    print(summary_csv)


if __name__ == "__main__":
    main()
