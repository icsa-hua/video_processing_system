from __future__ import annotations

import argparse
import csv
import re
import time

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

import cv2
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
    count_data_rows,
    dump_json,
    latency_summary,
    probe_model_backend,
    read_cbor_lines,
    read_csv_rows,
    summarize_stage_latency,
)
from obs_system.utils.global_config import CONF_THR, NMS_IOU
from obs_system.utils.logger import get_logger
from ultralytics.utils import DEFAULT_CFG


logger = get_logger("obs_system." + __name__)

DEFAULT_VIDEO_SOURCE = "samples/MVI_39401.mp4"
DEFAULT_MODEL = "assets/compressed_models/edi_jetson_model.engine"
DEFAULT_OUTPUT_DIR = "experiment_results"
MQTT_ARCHIVE_PATH = Path("assets/mqtt/saved_publishes.cbor")
HAZARD_CSV_PATH = Path("assets/hazard_events/hazard_events.csv")
ABLATION_MAX_FRAMES = 1500
ABLATION_MAX_VIDEO_SECONDS = 60.0
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


@dataclass(frozen=True)
class AblationSpec:
    key: str
    label: str
    purpose: str
    config_updates: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


ABLATION_SPECS: list[AblationSpec] = [
    AblationSpec(
        key="full_pipeline",
        label="Full pipeline",
        purpose="Baseline",
    ),
    AblationSpec(
        key="no_mqtt",
        label="No MQTT",
        purpose="Measure communication overhead",
        config_updates={"mqtt": False},
    ),
    AblationSpec(
        key="no_saving",
        label="No frame/event saving",
        purpose="Measure output I/O overhead",
        config_updates={"save": False},
    ),
    AblationSpec(
        key="no_mqtt_no_saving",
        label="No MQTT + no saving",
        purpose="Measure combined communication and output I/O overhead",
        config_updates={"mqtt": False, "save": False},
    ),
    AblationSpec(
        key="no_mog2_gating",
        label="No MOG2 gating",
        purpose="Show whether motion gating helps or hurts",
    ),
    AblationSpec(
        key="no_lane_segmentation",
        label="No DeepLab / lane segmentation",
        purpose="Measure lane-module cost",
        notes="This repo has no DeepLab module; this disables the classical lane/crosswalk scene-mask path instead.",
    ),
    AblationSpec(
        key="full_frame_only_no_tiling",
        label="No tiling, full-frame only",
        purpose="Compare tiled-vs-full-frame branch cost",
    ),
    AblationSpec(
        key="tiling_enabled",
        label="Tiling enabled",
        purpose="Compare tiled branch speed and optional accuracy tradeoff",
        config_updates={"force_tiles": True},
    ),
    AblationSpec(
        key="no_panorama_reprojection",
        label="No panorama reprojection",
        purpose="Measure panorama reprojection overhead",
        config_updates={"panorama": False},
    ),
]


def _video_source_slug(video_source: str) -> str:
    source = (video_source or "").strip()
    if not source:
        return "video_source"

    parsed = urlparse(source)
    if parsed.scheme and parsed.netloc:
        host = parsed.hostname or parsed.netloc.split("@")[-1] or "stream"
        path_stem = Path(parsed.path).stem or "stream"
        raw_slug = f"{host}_{path_stem}"
    else:
        raw_slug = Path(source).stem or Path(source).name or "video_source"

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_slug).strip("._-")
    return slug or "video_source"


def _estimate_video_frame_budget(video_source: str) -> int:
    frame_budget = int(ABLATION_MAX_FRAMES)
    source = (video_source or "").strip()
    if not source or "://" in source:
        return frame_budget

    source_path = Path(source)
    resolved = source_path if source_path.is_absolute() else Path.cwd() / source_path

    cap = cv2.VideoCapture(str(resolved))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    if fps > 0.0 and np.isfinite(fps):
        time_budget_frames = max(1, int(ABLATION_MAX_VIDEO_SECONDS * fps))
        return max(1, min(frame_budget, time_budget_frames))

    return frame_budget


def _normalize_batch_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _truncate_batch_payload(batch_payload: Any, keep_count: int) -> tuple[list[Any], list[Any], list[Any]]:
    paths, im0s, labels = batch_payload
    return (
        _normalize_batch_items(paths)[:keep_count],
        _normalize_batch_items(im0s)[:keep_count],
        _normalize_batch_items(labels)[:keep_count],
    )


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
        batch_count = len(_normalize_batch_items(batch_payload[1]))
        if batch_count <= self._remaining:
            object.__setattr__(self, "_remaining", self._remaining - batch_count)
            return batch_payload

        truncated = _truncate_batch_payload(batch_payload, self._remaining)
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
    streamer.args.panorama = bool(config.panorama)


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

    crop_sizes: list[int] = []
    crop_counts: list[int] = []
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


def _skip_reason(spec: AblationSpec, base_config: PipelineConfig) -> str | None:
    if spec.key in {"no_mqtt", "no_mqtt_no_saving"} and not base_config.mqtt:
        return "baseline MQTT is disabled"
    if spec.key in {"no_saving", "no_mqtt_no_saving"} and not base_config.save:
        return "baseline saving is disabled"
    if spec.key == "no_panorama_reprojection" and not base_config.panorama:
        return "baseline panorama mode is disabled"
    if spec.key in {"full_frame_only_no_tiling", "tiling_enabled"} and base_config.panorama:
        return "panorama mode overrides the tiling/full-frame branch"
    return None


def _apply_no_saving_patches(exit_stack: ExitStack, app: Application, streamer: Any) -> None:
    subtractor = app.logic_module.get("SUBTRACTOR")
    if subtractor is not None and hasattr(subtractor, "detect"):
        original_detect = subtractor.detect

        def detect_without_debug_saves(self, batch, save_img: bool = False):
            return original_detect(batch, save_img=False)

        exit_stack.enter_context(_temporary_method(subtractor, "detect", detect_without_debug_saves))

    def record_hazard_without_disk(self, preds, image, hazards, frame_id=None, *, record_stage: bool = True):
        if image is None or not hazards:
            return
        frame_id = self._extract_frame_id(preds)
        self._publish_hazard_alert(
            preds=preds,
            hazards=hazards,
            frame_name="",
            frame_id=frame_id,
            record_stage=False,
        )

    exit_stack.enter_context(_temporary_method(streamer, "_record_hazard_evidence", record_hazard_without_disk))


def _apply_no_mog2_gating_patch(exit_stack: ExitStack, app: Application) -> None:
    subtractor = app.logic_module.get("SUBTRACTOR")
    if subtractor is None or not hasattr(subtractor, "detect"):
        return

    original_detect = subtractor.detect

    def detect_without_gate(self, batch, save_img: bool = False):
        motion_flags, lanes_final = original_detect(batch, save_img=save_img)
        return [True] * len(motion_flags), lanes_final

    exit_stack.enter_context(_temporary_method(subtractor, "detect", detect_without_gate))


def _apply_no_lane_segmentation_patches(exit_stack: ExitStack, app: Application) -> None:
    subtractor = app.logic_module.get("SUBTRACTOR")
    if subtractor is None:
        return

    if hasattr(subtractor, "detect"):
        original_detect = subtractor.detect

        def detect_without_lane_outputs(self, batch, save_img: bool = False):
            motion_flags, _ = original_detect(batch, save_img=False)
            self.lanes_mask = None
            self.crosswalk_mask = None
            self._drivable_confidence_map = None
            return motion_flags, None

        exit_stack.enter_context(_temporary_method(subtractor, "detect", detect_without_lane_outputs))

    if hasattr(subtractor, "get_scene_masks"):
        original_get_scene_masks = subtractor.get_scene_masks

        def get_zero_scene_masks(self, expand_px: int = 0):
            scene = original_get_scene_masks(expand_px=expand_px)
            lane_mask = scene.get("lane_mask")
            if lane_mask is not None and getattr(lane_mask, "size", 0) > 0:
                shape = lane_mask.shape[:2]
            elif getattr(self, "_last_frame_shape", None) is not None:
                shape = tuple(self._last_frame_shape)
            else:
                return {"lane_mask": None, "crosswalk_mask": None, "drivable_confidence_map": None}

            zeros = np.zeros(shape, dtype=np.uint8)
            return {
                "lane_mask": zeros,
                "crosswalk_mask": zeros.copy(),
                "drivable_confidence_map": None,
            }

        exit_stack.enter_context(_temporary_method(subtractor, "get_scene_masks", get_zero_scene_masks))

    if hasattr(subtractor, "update_drivable_confidence"):
        def noop_update_drivable_confidence(self, *args, **kwargs):
            return None

        exit_stack.enter_context(
            _temporary_method(subtractor, "update_drivable_confidence", noop_update_drivable_confidence)
        )


def _apply_variant_runtime_overrides(
    spec: AblationSpec,
    app: Application,
    streamer: Any,
    config: PipelineConfig,
) -> ExitStack:
    exit_stack = ExitStack()

    if spec.key in {"no_saving", "no_mqtt_no_saving"}:
        _apply_no_saving_patches(exit_stack, app, streamer)

    if spec.key == "no_mog2_gating":
        _apply_no_mog2_gating_patch(exit_stack, app)

    if spec.key == "no_lane_segmentation":
        _apply_no_lane_segmentation_patches(exit_stack, app)

    if spec.key == "full_frame_only_no_tiling":
        exit_stack.enter_context(_temporary_attr(streamer, "force_streaming_no_tiles", True))
        exit_stack.enter_context(_temporary_attr(streamer.args, "force_tiles", False))

    if spec.key == "tiling_enabled":
        exit_stack.enter_context(_temporary_attr(streamer, "force_streaming_no_tiles", False))
        exit_stack.enter_context(_temporary_attr(streamer.args, "force_tiles", True))

    if spec.key == "no_panorama_reprojection":
        exit_stack.enter_context(_temporary_attr(streamer.args, "panorama", False))

    return exit_stack


def _apply_ablation_video_cap(exit_stack: ExitStack, streamer: Any, config: PipelineConfig) -> int:
    frame_budget = _estimate_video_frame_budget(config.video_source)
    original_setup_source = streamer.setup_source

    def setup_source_with_budget(self, source: str) -> None:
        original_setup_source(source)
        self.dataset = _FrameBudgetDataset(self.dataset, max_frames=frame_budget)
        self.run_metrics["ablation_frame_cap"] = int(frame_budget)
        self.run_metrics["ablation_video_seconds_cap"] = float(ABLATION_MAX_VIDEO_SECONDS)

    exit_stack.enter_context(_temporary_method(streamer, "setup_source", setup_source_with_budget))
    return frame_budget


def _run_variant(
    spec: AblationSpec,
    base_config: PipelineConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    source_slug = _video_source_slug(base_config.video_source)
    skip_reason = _skip_reason(spec, base_config)
    if skip_reason is not None:
        return {
            "key": spec.key,
            "label": spec.label,
            "purpose": spec.purpose,
            "status": "skipped",
            "notes": skip_reason if not spec.notes else f"{skip_reason}. {spec.notes}",
        }

    config = base_config.with_updates(**spec.config_updates).validate()
    run_dir = output_dir / source_slug / spec.key
    run_dir.mkdir(parents=True, exist_ok=True)

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

        hazard_rows_before = count_data_rows(HAZARD_CSV_PATH)
        mqtt_rows_before = len(read_cbor_lines(MQTT_ARCHIVE_PATH))

        with ExitStack() as run_exit_stack:
            _apply_ablation_video_cap(run_exit_stack, streamer, config)
            run_exit_stack.enter_context(_apply_variant_runtime_overrides(spec, app, streamer, config))
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
            logger.debug("Application close_app failed during ablation cleanup", exc_info=True)
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
    if streamer is None:
        raise RuntimeError("Streamer setup failed before execution started.")

    detection_metrics = streamer.mp.results() if config.bench and streamer.mp is not None else {}
    output_load = _summarize_output_load(
        hazard_rows_before=hazard_rows_before,
        mqtt_rows_before=mqtt_rows_before,
        streamer_metrics=streamer_metrics,
    )

    frames_emitted = int(streamer_metrics.get("frames_emitted", 0))
    fps = (frames_emitted / elapsed_s) if elapsed_s > 0 else 0.0
    notes = spec.notes
    if spec.key == "tiling_enabled":
        notes = "Forced tiled branch for comparison."
    elif spec.key == "full_frame_only_no_tiling":
        notes = "Forces the standard full-frame branch even if auto-tiling would trigger."
    elif spec.key == "no_mog2_gating":
        notes = "Keeps the subtractor active but forwards every frame to inference."

    return {
        "key": spec.key,
        "label": spec.label,
        "purpose": spec.purpose,
        "status": "completed",
        "notes": notes,
        "run_dir": str(run_dir),
        "config": {
            "video_source": config.video_source,
            "model_name": config.model_name,
            "mqtt": bool(config.mqtt),
            "save": bool(config.save),
            "roi": bool(config.roi),
            "fep": bool(config.fep),
            "use_TRT": bool(config.use_TRT),
            "jetson_profile": bool(config.jetson_profile),
            "force_tiles": bool(getattr(streamer.args, "force_tiles", False)),
            "panorama": bool(getattr(streamer.args, "panorama", False)),
            "ablation_frame_cap": int(streamer_metrics.get("ablation_frame_cap", ABLATION_MAX_FRAMES)),
            "ablation_video_seconds_cap": float(streamer_metrics.get("ablation_video_seconds_cap", ABLATION_MAX_VIDEO_SECONDS)),
        },
        "metrics": {
            "fps": float(fps),
            "total_ms": float(latency.get("avg_latency_ms", 0.0)),
            "p50_latency_ms": float(latency.get("p50_latency_ms", 0.0)),
            "p95_latency_ms": float(latency.get("p95_latency_ms", 0.0)),
            "runtime_seconds": float(elapsed_s),
            "stream_open_seconds": streamer_metrics.get("stream_open_seconds"),
            "first_frame_seconds": streamer_metrics.get("first_frame_seconds"),
            "frames_observed": int(streamer_metrics.get("frames_observed", 0)),
            "frames_emitted": frames_emitted,
            "dropped_frames": int(streamer_metrics.get("dropped_frames", 0)),
            "confidence_threshold": float(CONF_THR),
            "nms_iou": float(NMS_IOU),
        },
        "setup_metrics": setup_metrics,
        "stage_latency": summarize_stage_latency(frame_rows, STAGE_LATENCY_KEYS),
        "hardware": hardware,
        "detection_metrics": detection_metrics,
        "output_load": output_load,
    }


def _compute_deltas(results: list[dict[str, Any]]) -> None:
    baseline = next((row for row in results if row.get("key") == "full_pipeline" and row.get("status") == "completed"), None)
    if baseline is None:
        return

    baseline_total_ms = float(baseline["metrics"].get("total_ms", 0.0))
    for row in results:
        if row.get("status") != "completed":
            row["difference_from_full_pipeline_pct"] = None
            row["difference_from_full_pipeline_ms"] = None
            continue

        total_ms = float(row["metrics"].get("total_ms", 0.0))
        row["difference_from_full_pipeline_ms"] = baseline_total_ms - total_ms
        if row.get("key") == "full_pipeline":
            row["difference_from_full_pipeline_pct"] = 0.0
        elif baseline_total_ms > 0.0:
            row["difference_from_full_pipeline_pct"] = ((baseline_total_ms - total_ms) / baseline_total_ms) * 100.0
        else:
            row["difference_from_full_pipeline_pct"] = None


def _fmt_metric(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _fmt_delta(row: dict[str, Any]) -> str:
    if row.get("key") == "full_pipeline" and row.get("status") == "completed":
        return "baseline"
    value = row.get("difference_from_full_pipeline_pct")
    if value is None:
        return "n/a"
    sign = "+" if float(value) >= 0.0 else ""
    return f"{sign}{float(value):.1f}%"


def _write_markdown_summary(path: Path, base_config: PipelineConfig, results: list[dict[str, Any]]) -> None:
    lines = [
        "# TensorRT Ablation Study",
        "",
        f"- Video source: `{base_config.video_source}`",
        f"- Model: `{base_config.model_name}`",
        f"- Evaluation cap: up to `{ABLATION_MAX_FRAMES}` frames or `{int(ABLATION_MAX_VIDEO_SECONDS)}` seconds of source video, whichever is smaller.",
        "- Positive percentages mean the variant is faster than the full pipeline based on average `total_ms`.",
        "- `No DeepLab / lane segmentation` disables this repo's classical lane/crosswalk scene-mask path because there is no DeepLab module in the current codebase.",
        "",
        "| Configuration | FPS | Total ms | Difference from full pipeline | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for row in results:
        status = row.get("status")
        if status == "completed":
            fps = _fmt_metric(row["metrics"].get("fps"))
            total_ms = _fmt_metric(row["metrics"].get("total_ms"))
            notes = row.get("notes") or row.get("purpose", "")
        elif status == "skipped":
            fps = "SKIPPED"
            total_ms = "SKIPPED"
            notes = row.get("notes", "")
        else:
            fps = "FAILED"
            total_ms = "FAILED"
            notes = row.get("error", "")

        lines.append(
            f"| {row.get('label', row.get('key', 'variant'))} | {fps} | {total_ms} | {_fmt_delta(row)} | {notes} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv_summary(path: Path, base_config: PipelineConfig, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "key",
        "label",
        "purpose",
        "status",
        "video_source",
        "model_name",
        "fps",
        "total_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "runtime_seconds",
        "frames_observed",
        "frames_emitted",
        "dropped_frames",
        "difference_from_full_pipeline_pct",
        "difference_from_full_pipeline_ms",
        "mqtt_enabled",
        "save_enabled",
        "roi_enabled",
        "fep_enabled",
        "use_trt",
        "jetson_profile",
        "force_tiles",
        "panorama",
        "ablation_frame_cap",
        "ablation_video_seconds_cap",
        "saved_event_count",
        "mqtt_crop_batch_count",
        "avg_crop_jpeg_bytes",
        "notes",
        "run_dir",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            config = row.get("config", {})
            metrics = row.get("metrics", {})
            output_load = row.get("output_load", {})
            writer.writerow(
                {
                    "key": row.get("key"),
                    "label": row.get("label"),
                    "purpose": row.get("purpose"),
                    "status": row.get("status"),
                    "video_source": config.get("video_source", base_config.video_source),
                    "model_name": config.get("model_name", base_config.model_name),
                    "fps": metrics.get("fps"),
                    "total_ms": metrics.get("total_ms"),
                    "p50_latency_ms": metrics.get("p50_latency_ms"),
                    "p95_latency_ms": metrics.get("p95_latency_ms"),
                    "runtime_seconds": metrics.get("runtime_seconds"),
                    "frames_observed": metrics.get("frames_observed"),
                    "frames_emitted": metrics.get("frames_emitted"),
                    "dropped_frames": metrics.get("dropped_frames"),
                    "difference_from_full_pipeline_pct": row.get("difference_from_full_pipeline_pct"),
                    "difference_from_full_pipeline_ms": row.get("difference_from_full_pipeline_ms"),
                    "mqtt_enabled": config.get("mqtt"),
                    "save_enabled": config.get("save"),
                    "roi_enabled": config.get("roi"),
                    "fep_enabled": config.get("fep"),
                    "use_trt": config.get("use_TRT"),
                    "jetson_profile": config.get("jetson_profile"),
                    "force_tiles": config.get("force_tiles"),
                    "panorama": config.get("panorama"),
                    "ablation_frame_cap": config.get("ablation_frame_cap"),
                    "ablation_video_seconds_cap": config.get("ablation_video_seconds_cap"),
                    "saved_event_count": output_load.get("saved_event_count"),
                    "mqtt_crop_batch_count": output_load.get("mqtt_crop_batch_count"),
                    "avg_crop_jpeg_bytes": output_load.get("avg_crop_jpeg_bytes"),
                    "notes": row.get("notes", ""),
                    "run_dir": row.get("run_dir", ""),
                }
            )


def _build_base_config(args: argparse.Namespace) -> PipelineConfig:
    model_suffix = Path(args.model).suffix.lower()
    use_tensorrt = args.use_trt
    if use_tensorrt is None:
        use_tensorrt = model_suffix == ".engine"

    config = PipelineConfig(
        model_name=args.model,
        video_source=args.video_source,
        type="tracking",
        gui=False,
        mqtt=bool(args.mqtt),
        show=False,
        verbose=bool(args.verbose),
        save=bool(args.save_outputs),
        roi=bool(args.roi),
        half=True,
        fep=bool(args.fep),
        bench=bool(args.labels_dir),
        bench_labels=args.labels_dir or DEFAULT_BENCH_LABELS,
        use_TRT=bool(use_tensorrt),
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
        force_tiles=False,
        panorama=bool(args.panorama),
    )
    return config.validate()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a TensorRT ablation study on one representative video by toggling "
            "MQTT, saving, MOG2 gating, lane masks, tiling, and panorama reprojection."
        )
    )
    parser.add_argument("--video-source", default=DEFAULT_VIDEO_SOURCE, help="Input video path or stream URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="TensorRT engine or ONNX model path.")
    parser.add_argument("--use-trt", action=argparse.BooleanOptionalAction, default=None, help="Force TensorRT mode for ONNX models. Defaults to on for `.engine` models.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where ablation logs and summaries are written.")
    parser.add_argument("--labels-dir", default="", help="Optional YOLO label directory for accuracy metrics.")
    parser.add_argument("--roi", action=argparse.BooleanOptionalAction, default=True, help="Enable ROI cropping.")
    parser.add_argument("--fep", action=argparse.BooleanOptionalAction, default=False, help="Enable fisheye reprojection.")
    parser.add_argument("--mqtt", action=argparse.BooleanOptionalAction, default=True, help="Enable MQTT in the full-pipeline baseline.")
    parser.add_argument("--save-outputs", action=argparse.BooleanOptionalAction, default=True, help="Enable frame/event saving in the full-pipeline baseline.")
    parser.add_argument("--panorama", action=argparse.BooleanOptionalAction, default=False, help="Enable panorama reprojection in the full-pipeline baseline.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose streamer logging.")
    parser.add_argument("--stream-limit-hours", type=float, default=0.0, help="Live-stream runtime cap in hours. Use 0 to disable.")
    parser.add_argument("--lane-recalibration-interval-frames", type=int, default=0, help="For live streams, rerun lane calibration after this many frames. Use 0 to disable.")
    parser.add_argument("--jetson-interval", type=float, default=1.0, help="Sampling interval for Jetson telemetry in seconds.")
    parser.add_argument("--jetson-profile", action=argparse.BooleanOptionalAction, default=True, help="Enable the Jetson-optimized execution branch.")
    parser.add_argument("--jetson-hazard-scale", type=float, default=1.0, help="Scale factor for Jetson hazard-mask processing.")
    parser.add_argument("--jetson-cpu-threads", type=int, default=0, help="CPU thread cap for the Jetson execution branch. Use 0 for runtime defaults.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model_ok, model_reason = probe_model_backend(args.model)
    if not model_ok:
        raise SystemExit(f"Cannot run model '{args.model}': {model_reason}")

    if "://" not in args.video_source and not Path(args.video_source).exists():
        raise SystemExit(f"Video source does not exist: {args.video_source}")

    base_config = _build_base_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_slug = _video_source_slug(base_config.video_source)

    results: list[dict[str, Any]] = []
    for spec in ABLATION_SPECS:
        try:
            logger.info("Running ablation variant: %s", spec.label)
            results.append(_run_variant(spec=spec, base_config=base_config, args=args, output_dir=output_dir))
        except Exception as exc:
            logger.exception("Ablation variant failed: %s", spec.label)
            results.append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "purpose": spec.purpose,
                    "status": "failed",
                    "notes": spec.notes,
                    "error": str(exc),
                }
            )

    _compute_deltas(results)

    summary_json = output_dir / f"ablation_summary_{source_slug}.json"
    summary_md = output_dir / f"ablation_summary_{source_slug}.md"
    summary_csv = output_dir / f"ablation_summary_{source_slug}.csv"
    dump_json(summary_json, results)
    _write_markdown_summary(summary_md, base_config, results)
    _write_csv_summary(summary_csv, base_config, results)

    print(summary_md)


if __name__ == "__main__":
    main()
