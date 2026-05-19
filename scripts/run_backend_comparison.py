from __future__ import annotations

import argparse
import csv
import json
import threading
import time

import cbor2
import numpy as np

from pathlib import Path
from typing import Any

from obs_system.application_module.dummy_application.dummy_app import Application
from obs_system.application_module.dummy_application.pipeline_config import DEFAULT_BENCH_LABELS, PipelineConfig
from obs_system.detection_module.dummy_predictor.stream_trt import TensorRTRTXStreamer
from obs_system.detection_module.dummy_predictor.stream_y8_onnx import OnnxY8Streamer
from obs_system.detection_module.dummy_predictor.stream_yolov8 import Yolov8Streamer
from obs_system.utils.appraisal import frame_list, perf
from obs_system.utils.logger import get_logger
from ultralytics.utils import DEFAULT_CFG


logger = get_logger("obs_system." + __name__)

DEFAULT_PT_MODEL = "assets/compressed_models/yolov8s.pt"
DEFAULT_ONNX_MODEL = "assets/compressed_models/mixed_dataset_trained_yolov8s.onnx"
DEFAULT_ENGINE_MODEL = "assets/compressed_models/mixed_dataset_trained_yolov8s_mixed_batch_trt_fp16_noint8.engine"
MQTT_ARCHIVE_PATH = Path("assets/mqtt/saved_publishes.cbor")
HAZARD_CSV_PATH = Path("assets/hazard_events/hazard_events.csv")


class JetsonSampler:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = max(0.2, float(interval_s))
        self.samples: list[dict[str, float]] = []
        self.available = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        try:
            from jtop import jtop
        except Exception:
            return

        try:
            with jtop() as jetson:
                self.available = True
                while jetson.ok() and not self._stop.is_set():
                    stats = dict(getattr(jetson, "stats", {}) or {})
                    sample = _extract_jetson_sample(stats)
                    if sample:
                        self.samples.append(sample)
                    time.sleep(self.interval_s)
        except Exception:
            logger.debug("Jetson sampling unavailable", exc_info=True)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None

    cleaned = []
    for ch in text:
        if ch.isdigit() or ch in ".-":
            cleaned.append(ch)
        elif cleaned:
            break
    if not cleaned:
        return None
    try:
        return float("".join(cleaned))
    except ValueError:
        return None


def _pick_value(stats: dict[str, Any], *needles: str) -> float | None:
    lowered = [needle.lower() for needle in needles]
    matches: list[float] = []
    for key, value in stats.items():
        key_lower = str(key).lower()
        if any(needle in key_lower for needle in lowered):
            parsed = _to_float(value)
            if parsed is not None:
                matches.append(parsed)
    if matches:
        return float(np.mean(matches))
    return None


def _extract_jetson_sample(stats: dict[str, Any]) -> dict[str, float]:
    sample: dict[str, float] = {}

    cpu_util = _pick_value(stats, "cpu")
    gpu_util = _pick_value(stats, "gpu")
    ram_used = _pick_value(stats, "ram")
    temp_cpu = _pick_value(stats, "temp cpu", "cpu temp")
    temp_gpu = _pick_value(stats, "temp gpu", "gpu temp")
    power_w = _pick_value(stats, "power tot", "power", "vdd")

    if cpu_util is not None:
        sample["jetson_cpu_util"] = cpu_util
    if gpu_util is not None:
        sample["jetson_gpu_util"] = gpu_util
    if ram_used is not None:
        sample["jetson_ram_metric"] = ram_used
    if temp_cpu is not None:
        sample["jetson_cpu_temp_c"] = temp_cpu
    if temp_gpu is not None:
        sample["jetson_gpu_temp_c"] = temp_gpu
    if power_w is not None:
        sample["jetson_power_metric"] = power_w

    return sample


def _count_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _read_cbor_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    messages: list[dict[str, Any]] = []
    for raw in path.read_bytes().splitlines():
        if not raw.strip():
            continue
        try:
            payload = cbor2.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            messages.append(payload)
    return messages


def _latency_summary(frame_rows: list[dict[str, str]]) -> dict[str, float]:
    latencies = [float(row["total_ms"]) for row in frame_rows if row.get("total_ms")]
    if not latencies:
        return {
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    xs = np.asarray(latencies, dtype=np.float32)
    return {
        "avg_latency_ms": float(xs.mean()),
        "p50_latency_ms": float(np.percentile(xs, 50)),
        "p95_latency_ms": float(np.percentile(xs, 95)),
    }


def _mean_metric(rows: list[dict[str, str]], key: str) -> float | None:
    vals = []
    for row in rows:
        value = row.get(key)
        if value in (None, "", "nan"):
            continue
        try:
            vals.append(float(value))
        except ValueError:
            continue
    if not vals:
        return None
    return float(np.mean(vals))


def _summarize_jetson(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {}

    keys = sorted({key for sample in samples for key in sample})
    out: dict[str, float] = {}
    for key in keys:
        vals = [sample[key] for sample in samples if key in sample]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
    return out


def _build_streamer(model_path: str):
    suffix = Path(model_path).suffix.lower()
    if suffix == ".pt":
        return Yolov8Streamer(DEFAULT_CFG, {}, None)
    if suffix == ".onnx":
        return OnnxY8Streamer(DEFAULT_CFG, {}, None)
    if suffix == ".engine":
        return TensorRTRTXStreamer(DEFAULT_CFG, {}, None)
    raise ValueError(f"Unsupported model path: {model_path}")


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


def _summarize_output_load(
    hazard_rows_before: int,
    mqtt_rows_before: int,
    streamer_metrics: dict[str, Any],
) -> dict[str, Any]:
    hazard_rows = _read_csv_rows(HAZARD_CSV_PATH)
    mqtt_messages = _read_cbor_lines(MQTT_ARCHIVE_PATH)

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
        half=False,
        fep=bool(args.fep),
        bench=bool(args.labels_dir),
        bench_labels=args.labels_dir or DEFAULT_BENCH_LABELS,
        use_TRT=(model_suffix == ".engine"),
        plot_perf=True,
        only_FPS=True,
        preview_max_width=960,
        preview_jpeg_quality=70,
        preview_fps=8.0,
        stream_limit_hours=float(args.stream_limit_hours),
    ).validate()

    perf.reset()
    frame_list.clear()

    app = Application(save=config.save, verbose=config.verbose)
    app.setup_process(config)
    app.setup_logic_module(config)
    app.setup_mqtt() if config.mqtt else None

    streamer = _build_streamer(model_path)
    _configure_streamer_args(streamer, config, run_dir)

    model_spec = config.resolve_model()
    model_name = f"{model_spec.name}.{model_spec.kind}"
    streamer.setup_model(model_name=model_name, path_to_load=model_spec.path, opt=config.type)

    app.streamer = streamer
    app.model = streamer.model

    hazard_rows_before = _count_data_rows(HAZARD_CSV_PATH)
    mqtt_rows_before = len(_read_cbor_lines(MQTT_ARCHIVE_PATH))

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

    frame_rows = _read_csv_rows(run_dir / "perf_frames.csv")
    batch_rows = _read_csv_rows(run_dir / "perf_log.csv")
    latency = _latency_summary(frame_rows)
    streamer_metrics = dict(streamer.run_metrics)
    hardware = {
        "cpu_util_mean": _mean_metric(batch_rows, "cpu_util"),
        "gpu_util_mean": _mean_metric(batch_rows, "gpu_util"),
        "gpu_mem_used_mb_mean": _mean_metric(batch_rows, "gpu_mem_used_mb"),
    }
    hardware.update(_summarize_jetson(jetson_sampler.samples))

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
        "hardware": hardware,
        "detection_metrics": detection_metrics,
        "output_load": output_load,
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def _write_markdown_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Backend Comparison",
        "",
        "| Model | FPS | Avg Latency ms | P50 ms | P95 ms | CPU % | GPU % | GPU Mem MB | Dropped | Stream Open s | First Frame s | mAP | Precision | Recall | F1 | Saved Events | MQTT Batches | Avg Crop JPEG B |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for summary in summaries:
        detection = summary.get("detection_metrics", {})
        hardware = summary.get("hardware", {})
        output_load = summary.get("output_load", {})
        lines.append(
            "| {model} | {fps:.2f} | {avg:.2f} | {p50:.2f} | {p95:.2f} | {cpu} | {gpu} | {mem} | {dropped} | {open_s} | {first_s} | {map_} | {prec} | {rec} | {f1} | {events} | {mqtt} | {crop_b:.2f} |".format(
                model=Path(summary["model_path"]).name,
                fps=summary.get("fps", 0.0),
                avg=summary.get("avg_latency_ms", 0.0),
                p50=summary.get("p50_latency_ms", 0.0),
                p95=summary.get("p95_latency_ms", 0.0),
                cpu=_fmt_opt(hardware.get("cpu_util_mean")),
                gpu=_fmt_opt(hardware.get("gpu_util_mean")),
                mem=_fmt_opt(hardware.get("gpu_mem_used_mb_mean")),
                dropped=summary.get("dropped_frames", 0),
                open_s=_fmt_opt(summary.get("stream_open_seconds")),
                first_s=_fmt_opt(summary.get("first_frame_seconds")),
                map_=_fmt_opt(detection.get("mAP")),
                prec=_fmt_opt(detection.get("Precision")),
                rec=_fmt_opt(detection.get("Recall")),
                f1=_fmt_opt(detection.get("F1")),
                events=output_load.get("saved_event_count", 0),
                mqtt=output_load.get("mqtt_crop_batch_count", 0),
                crop_b=output_load.get("avg_crop_jpeg_bytes", 0.0),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_opt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PT/ONNX/ENGINE backend comparison on the same video source.")
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = [args.pt_model, args.onnx_model, args.engine_model]
    summaries = [_run_single_benchmark(model_path=path, args=args, output_dir=output_dir) for path in model_paths]

    summary_json = output_dir / "comparison_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    _write_markdown_summary(output_dir / "comparison_summary.md", summaries)


if __name__ == "__main__":
    main()
