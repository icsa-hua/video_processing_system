from __future__ import annotations

import argparse
import time

import cv2
import torch

from collections.abc import Generator
from pathlib import Path
from typing import Any

from obs_system.detection_module.interface.factory import StreamerFactory
from obs_system.utils.global_config import CONF_THR, NMS_IOU
from obs_system.utils.benchmarking.backend_benchmark import (
    JetsonSampler,
    build_hardware_summary,
    dump_json,
    fmt_opt,
    latency_summary,
    read_csv_rows,
)
from obs_system.utils.benchmarking.metrics.pc_performance import (
    CPUMonitor,
    FramePerfLogger,
    GPUMonitor,
    PerfLogger,
    SlidingCounter,
    TimelineLogger,
)
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.checks import check_imgsz


DEFAULT_PT_MODEL = "assets/compressed_models/yolov8s.pt"
DEFAULT_ONNX_MODEL = "assets/compressed_models/yolov8s.onnx"
DEFAULT_ENGINE_MODEL = "assets/compressed_models/yolov8s.engine"


def _build_streamer(model_path: Path, batch_size: int, device: str, verbose: bool):
    factory = StreamerFactory(
        cfg=DEFAULT_CFG,
        overrides={
            "batch": int(batch_size),
            "device": device,
            "half": False,
            "verbose": bool(verbose),
        },
        callbacks=None,
    )
    model_name = model_path.name
    streamer, _ = factory.create(
        model_name=model_name,
        path_to_load=model_path,
        use_tensorrt=(model_path.suffix.lower() == ".engine"),
        opt="tracking",
    )
    streamer.args.batch = int(batch_size)
    streamer.args.device = device
    streamer.args.verbose = bool(verbose)
    streamer.args.half = False
    streamer.imgsz = check_imgsz(streamer.args.imgsz, stride=streamer.stride, min_dim=2)
    return streamer


def _iter_video_batches(video_source: str, batch_size: int, max_frames: int = 0) -> Generator[list[Any], None, None]:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {video_source}")

    yielded = 0
    try:
        while True:
            frames = []
            for _ in range(max(1, int(batch_size))):
                if max_frames > 0 and yielded >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
                yielded += 1

            if not frames:
                break
            yield frames

            if max_frames > 0 and yielded >= max_frames:
                break
    finally:
        cap.release()


def _synchronize_device(device: Any) -> None:
    if not torch.cuda.is_available():
        return
    try:
        if isinstance(device, torch.device):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
            return
        torch.cuda.synchronize()
    except Exception:
        pass


def _write_markdown_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Backend Inference Comparison",
        "",
        "| Model | FPS | Avg Latency ms | P50 ms | P95 ms | CPU % | GPU % | GPU Mem MB | Temp C | Power W | Power Mode | EMC MHz | Frames | Batches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]

    for summary in summaries:
        hardware = summary.get("hardware", {})
        lines.append(
            "| {model} | {fps:.2f} | {avg:.2f} | {p50:.2f} | {p95:.2f} | {cpu} | {gpu} | {mem} | {temp} | {power_w} | {power_mode} | {emc} | {frames} | {batches} |".format(
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
                frames=summary.get("frames_emitted", 0),
                batches=summary.get("batches_processed", 0),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single_benchmark(model_path: str, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    resolved_model_path = Path(model_path)
    run_dir = output_dir / resolved_model_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    streamer = _build_streamer(
        model_path=resolved_model_path,
        batch_size=args.batch_size,
        device=args.device,
        verbose=bool(args.verbose),
    )

    perf_logger = PerfLogger(str(run_dir / "perf_log.csv"))
    frame_logger = FramePerfLogger(str(run_dir / "perf_frames.csv"))
    timeline_logger = TimelineLogger(str(run_dir / "perf_timeline.jsonl"))
    gpu_mon = GPUMonitor(gpu_index=args.gpu_index)
    cpu_mon = CPUMonitor()
    jetson_sampler = JetsonSampler(interval_s=args.jetson_interval)
    infer_counter = SlidingCounter(window_s=1.0)
    fps_counter = SlidingCounter(window_s=1.0)

    frames_processed = 0
    batches_processed = 0
    inference_runtime_s = 0.0
    benchmark_wall_start = time.perf_counter()

    try:
        streamer.model.warmup(
            micro=max(1, int(args.batch_size)),
            warmup_sessions=max(1, int(args.warmup_sessions)),
        )
        jetson_sampler.start()

        for frames in _iter_video_batches(
            video_source=args.video_source,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
        ):
            batch_size = len(frames)
            images = streamer.preprocess(frames)
            gpu_stats = gpu_mon.sample()
            cpu_stats = cpu_mon.sample()

            _synchronize_device(streamer.device)
            t0 = time.perf_counter()
            infer_outputs = streamer.model(images, orig_imgs=frames, debug=bool(args.verbose))
            event = infer_outputs[1] if isinstance(infer_outputs, tuple) and len(infer_outputs) == 2 else None
            if event is not None and torch.cuda.is_available():
                torch.cuda.current_stream(device=streamer.device).wait_event(event)
            _synchronize_device(streamer.device)
            t1 = time.perf_counter()

            inference_runtime_s += t1 - t0
            frames_processed += batch_size
            batches_processed += 1

            infer_counter.add(t1, 1.0)
            fps_counter.add(t1, float(batch_size))

            inference_ms_per_frame = ((t1 - t0) * 1e3) / max(batch_size, 1)
            relative_t0 = t0 - benchmark_wall_start
            relative_t1 = t1 - benchmark_wall_start

            perf_logger.log(
                {
                    "t_wall": t1,
                    "batch_idx": batches_processed - 1,
                    "frames_in_batch": batch_size,
                    "res_w": int(frames[0].shape[1]),
                    "res_h": int(frames[0].shape[0]),
                    "motion_density": 0.0,
                    "avg_motion_score": 0.0,
                    "inference_ran": 1,
                    "frames_inferred": batch_size,
                    "infer_calls_per_sec": infer_counter.rate(t1),
                    "gpu_util": gpu_stats.get("gpu_util", float("nan")),
                    "gpu_mem_used_mb": gpu_stats.get("mem_used_mb", float("nan")),
                    "gpu_mem_total_mb": gpu_stats.get("mem_total_mb", float("nan")),
                    "cpu_util": cpu_stats.get("cpu_util", float("nan")),
                    "roi_ms_per_frame": 0.0,
                    "mog2_ms_per_frame": 0.0,
                    "preprocess_ms_per_frame": 0.0,
                    "inference_ms_per_frame": inference_ms_per_frame,
                    "postprocess_ms_per_frame": 0.0,
                    "total_ms_per_frame": inference_ms_per_frame,
                    "fps_sliding": fps_counter.rate(t1),
                }
            )

            for offset in range(batch_size):
                frame_logger.log(
                    {
                        "t_wall": t1,
                        "batch_idx": batches_processed - 1,
                        "frame_id": frames_processed - batch_size + offset,
                        "res_w": int(frames[offset].shape[1]),
                        "res_h": int(frames[offset].shape[0]),
                        "motion_passed": 1,
                        "motion_score": 1.0,
                        "gpu_util": gpu_stats.get("gpu_util", float("nan")),
                        "gpu_mem_used_mb": gpu_stats.get("mem_used_mb", float("nan")),
                        "cpu_util": cpu_stats.get("cpu_util", float("nan")),
                        "roi_ms": 0.0,
                        "mog2_ms": 0.0,
                        "preprocess_ms": 0.0,
                        "inference_ms": inference_ms_per_frame,
                        "postprocess_ms": 0.0,
                        "total_ms": inference_ms_per_frame,
                    }
                )

            timeline_logger.log_span(
                batches_processed - 1,
                "inference",
                relative_t0,
                relative_t1,
                {
                    "frames_in_batch": batch_size,
                    "model": resolved_model_path.name,
                },
            )
    finally:
        jetson_sampler.stop()
        perf_logger.close()
        frame_logger.close()
        timeline_logger.close()
        try:
            streamer.stop_save_worker()
            streamer.release_video_writers()
            streamer.release_dataset_resources()
        except Exception:
            pass

    frame_rows = read_csv_rows(run_dir / "perf_frames.csv")
    batch_rows = read_csv_rows(run_dir / "perf_log.csv")
    hardware = build_hardware_summary(
        batch_rows=batch_rows,
        jetson_samples=jetson_sampler.samples,
        jetson_context=jetson_sampler.static_context,
    )

    summary = {
        "benchmark_mode": "inference_only",
        "video_source": args.video_source,
        "model_path": model_path,
        "model_kind": resolved_model_path.suffix.lower().lstrip("."),
        "run_dir": str(run_dir),
        "batch_size": int(args.batch_size),
        "confidence_threshold": CONF_THR,
        "nms_iou": NMS_IOU,
        "runtime_seconds": inference_runtime_s,
        "fps": (frames_processed / inference_runtime_s) if inference_runtime_s > 0 else 0.0,
        **latency_summary(frame_rows),
        "frames_observed": frames_processed,
        "frames_emitted": frames_processed,
        "batches_processed": batches_processed,
        "hardware": hardware,
    }

    dump_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference-only PT/ONNX/ENGINE backend comparison on the same video.")
    parser.add_argument("--video-source", required=True, help="Input video path or stream URL.")
    parser.add_argument("--output-dir", default="assets/backend_inference_comparison", help="Directory where run summaries/logs are written.")
    parser.add_argument("--pt-model", default=DEFAULT_PT_MODEL, help="Path to the PT model.")
    parser.add_argument("--onnx-model", default=DEFAULT_ONNX_MODEL, help="Path to the ONNX model.")
    parser.add_argument("--engine-model", default=DEFAULT_ENGINE_MODEL, help="Path to the TensorRT engine.")
    parser.add_argument("--batch-size", type=int, default=16, help="Frames per inference batch.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap. Use 0 to process the full video.")
    parser.add_argument("--device", default="", help="Torch device override, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for utilization sampling.")
    parser.add_argument("--warmup-sessions", type=int, default=8, help="Warmup iterations before timing.")
    parser.add_argument("--jetson-interval", type=float, default=1.0, help="Sampling interval for Jetson telemetry in seconds.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose backend logging.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = [args.pt_model, args.onnx_model, args.engine_model]
    summaries = [_run_single_benchmark(model_path=path, args=args, output_dir=output_dir) for path in model_paths]

    dump_json(output_dir / "comparison_summary.json", summaries)
    _write_markdown_summary(output_dir / "comparison_summary.md", summaries)


if __name__ == "__main__":
    main()
