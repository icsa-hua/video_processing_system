from __future__ import annotations

import argparse
import time

import numpy as np

from pathlib import Path
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
        jetson_profile=bool(args.jetson_profile),
        jetson_hazard_scale=float(args.jetson_hazard_scale),
        jetson_cpu_threads=int(args.jetson_cpu_threads),
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

    summary = {
        "model_path": model_path,
        "model_kind": model_suffix.lstrip("."),
        "run_dir": str(run_dir),
        "confidence_threshold": CONF_THR,
        "nms_iou": NMS_IOU,
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
        "stage_latency": summarize_stage_latency(frame_rows, STAGE_LATENCY_KEYS),
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
        "| Model | FPS | Avg Latency ms | P50 ms | P95 ms | CPU % | GPU % | GPU Mem MB | Temp C | Power W | Power Mode | EMC MHz | Dropped | Stream Open s | First Frame s | mAP | Precision | Recall | F1 | Saved Events | MQTT Batches | Avg Crop JPEG B |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for summary in summaries:
        detection = summary.get("detection_metrics", {})
        hardware = summary.get("hardware", {})
        output_load = summary.get("output_load", {})
        lines.append(
            "| {model} | {fps:.2f} | {avg:.2f} | {p50:.2f} | {p95:.2f} | {cpu} | {gpu} | {mem} | {temp} | {power_w} | {power_mode} | {emc} | {dropped} | {open_s} | {first_s} | {map_} | {prec} | {rec} | {f1} | {events} | {mqtt} | {crop_b:.2f} |".format(
                model=Path(summary["model_path"]).name,
                fps=summary.get("fps", 0.0),
                avg=summary.get("avg_latency_ms", 0.0),
                p50=summary.get("p50_latency_ms", 0.0),
                p95=summary.get("p95_latency_ms", 0.0),
                cpu=fmt_opt(hardware.get("cpu_util_mean")),
                gpu=fmt_opt(hardware.get("gpu_util_mean")),
                mem=fmt_opt(hardware.get("gpu_mem_used_mb_mean")),
                temp=fmt_opt(hardware.get("temperature_c_mean")),
                power_w=fmt_opt(hardware.get("power_w_mean")),
                power_mode=fmt_opt(hardware.get("power_mode")),
                emc=fmt_opt(hardware.get("emc_frequency_mhz_mean")),
                dropped=summary.get("dropped_frames", 0),
                open_s=fmt_opt(summary.get("stream_open_seconds")),
                first_s=fmt_opt(summary.get("first_frame_seconds")),
                map_=fmt_opt(detection.get("mAP")),
                prec=fmt_opt(detection.get("Precision")),
                rec=fmt_opt(detection.get("Recall")),
                f1=fmt_opt(detection.get("F1")),
                events=output_load.get("saved_event_count", 0),
                mqtt=output_load.get("mqtt_crop_batch_count", 0),
                crop_b=output_load.get("avg_crop_jpeg_bytes", 0.0),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run PT/ONNX/ENGINE backend comparison on the same video source. "
            "On WSL / non-Jetson hosts the .engine backend is automatically skipped "
            "when TensorRT is not installed; missing model files are also skipped."
        )
    )
    parser.add_argument("--video-source", required=True, help="Input video path or stream URL.")
    parser.add_argument("--labels-dir", default="", help="Optional YOLO label directory for Precision/Recall/F1/mAP.")
    parser.add_argument("--output-dir", default="assets/backend_comparison", help="Directory where run summaries/logs are written.")
    parser.add_argument("--pt-model", default=DEFAULT_PT_MODEL, help="Path to the PT model.")
    parser.add_argument("--onnx-model", default=DEFAULT_ONNX_MODEL, help="Path to the ONNX model.")
    parser.add_argument("--engine-model", default=DEFAULT_ENGINE_MODEL, help="Path to the TensorRT engine.")
    parser.add_argument("--roi", action=argparse.BooleanOptionalAction, default=True, help="Enable ROI cropping.")
    parser.add_argument("--fep", action=argparse.BooleanOptionalAction, default=False, help="Enable fisheye projection.")
    parser.add_argument("--mqtt", action=argparse.BooleanOptionalAction, default=False, help="Enable MQTT publishing during the benchmark.")
    parser.add_argument("--save-outputs", action=argparse.BooleanOptionalAction, default=False, help="Enable saving rendered outputs during the benchmark.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose streamer logging.")
    parser.add_argument("--stream-limit-hours", type=float, default=0.0, help="Live-stream runtime cap in hours. Use 0 to disable.")
    parser.add_argument("--jetson-interval", type=float, default=1.0, help="Sampling interval for Jetson telemetry in seconds.")
    parser.add_argument("--jetson-profile", action=argparse.BooleanOptionalAction, default=False, help="Enable the Jetson-optimized execution branch.")
    parser.add_argument("--jetson-hazard-scale", type=float, default=1.0, help="Scale factor for Jetson hazard-mask processing. Use values below 1.0 to trade a small amount of precision for speed.")
    parser.add_argument("--jetson-cpu-threads", type=int, default=0, help="CPU thread cap for the Jetson-optimized execution branch. Use 0 to keep runtime defaults.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        try:
            summaries.append(_run_single_benchmark(model_path=path, args=args, output_dir=output_dir))
        except Exception as exc:
            print(f"[ERROR] {path}  —  {exc}")

    if not summaries:
        print("All benchmarks failed. Check model files and backend availability.")
        return

    summary_json = output_dir / "comparison_summary.json"
    dump_json(summary_json, summaries)

    _write_markdown_summary(output_dir / "comparison_summary.md", summaries)


if __name__ == "__main__":
    main()
