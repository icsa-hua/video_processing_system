from __future__ import annotations

import argparse
import time

import cv2
import torch
from torchvision.ops import batched_nms

from collections.abc import Generator
from pathlib import Path
from typing import Any

from obs_system.detection_module.interface.detection_batch import FrameDetections
from obs_system.detection_module.interface.factory import StreamerFactory
from obs_system.utils.global_config import CONF_THR, NMS_IOU
from obs_system.utils.benchmarking.backend_benchmark import (
    JetsonSampler,
    build_hardware_summary,
    dump_json,
    fmt_opt,
    latency_summary,
    probe_model_backend,
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
DEFAULT_FRAME_CAP = 1500
DEFAULT_VIDEO_SECONDS_CAP = 60.0


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


def _effective_frame_cap(video_source: str, requested_max_frames: int) -> int:
    hard_cap = _estimate_video_frame_cap(video_source)
    if int(requested_max_frames) > 0:
        return max(1, min(hard_cap, int(requested_max_frames)))
    return hard_cap


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


def _video_source_fps(video_source: str) -> float:
    cap = cv2.VideoCapture(video_source)
    try:
        if not cap.isOpened():
            return 0.0
        return float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
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


def _normalize_backend_outputs(
    streamer: Any,
    infer_outputs: Any,
    frames_bgr: list[Any],
    frame_start_idx: int,
) -> list[FrameDetections]:
    event = infer_outputs[1] if isinstance(infer_outputs, tuple) and len(infer_outputs) == 2 else None
    raw_outputs = infer_outputs[0] if event is not None else infer_outputs
    i_boxes, i_scores, i_classes = raw_outputs

    if event is not None and torch.cuda.is_available():
        torch.cuda.current_stream(device=streamer.device).wait_event(event)
        _synchronize_device(streamer.device)

    detections: list[FrameDetections] = []
    for batch_idx, frame_bgr in enumerate(frames_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_id = frame_start_idx + batch_idx

        boxes = i_boxes[batch_idx]
        scores = i_scores[batch_idx]
        classes = i_classes[batch_idx]
        if boxes is None or len(boxes) == 0:
            detections.append(FrameDetections.empty(frame_id=frame_id, batch_index=batch_idx, orig_img=frame_rgb))
            continue

        boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes)
        scores_t = scores if torch.is_tensor(scores) else torch.as_tensor(scores)
        classes_t = classes if torch.is_tensor(classes) else torch.as_tensor(classes)

        boxes_t = boxes_t.to(dtype=torch.float32)
        scores_t = scores_t.to(dtype=torch.float32)
        classes_t = classes_t.to(dtype=torch.int64)

        road_mask = streamer._get_road_filter_mask(classes_t)
        boxes_t = boxes_t[road_mask]
        scores_t = scores_t[road_mask]
        classes_t = classes_t[road_mask]

        if boxes_t.numel() == 0:
            detections.append(FrameDetections.empty(frame_id=frame_id, batch_index=batch_idx, orig_img=frame_rgb))
            continue

        keep = batched_nms(boxes_t, scores_t, classes_t.long(), iou_threshold=NMS_IOU)
        boxes_t = boxes_t[keep]
        scores_t = scores_t[keep]
        classes_t = classes_t[keep]

        if boxes_t.numel() == 0:
            detections.append(FrameDetections.empty(frame_id=frame_id, batch_index=batch_idx, orig_img=frame_rgb))
            continue

        detections.append(
            FrameDetections(
                frame_id=frame_id,
                batch_index=batch_idx,
                orig_img=frame_rgb,
                boxes=boxes_t,
                scores=scores_t,
                classes=classes_t,
            )
        )

    return detections


def _render_detection_frame(streamer: Any, detection: FrameDetections) -> Any:
    results = detection.to_results(streamer.converter.class_names)
    return streamer._render_prediction_frame(results)


def _write_markdown_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Backend Inference Comparison",
        "",
        f"- Evaluation cap: up to `{DEFAULT_FRAME_CAP}` frames or `{int(DEFAULT_VIDEO_SECONDS_CAP)}` seconds of source video, whichever is smaller.",
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

    try:
        streamer = _build_streamer(
            model_path=resolved_model_path,
            batch_size=args.batch_size,
            device=args.device,
            verbose=bool(args.verbose),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{model_path}': {exc}") from exc

    perf_logger = PerfLogger(str(run_dir / "perf_log.csv"), flush_every=64)
    frame_logger = FramePerfLogger(str(run_dir / "perf_frames.csv"), flush_every=128)
    timeline_logger = TimelineLogger(str(run_dir / "perf_timeline.jsonl"), flush_every=128)
    gpu_mon = GPUMonitor(gpu_index=args.gpu_index)
    cpu_mon = CPUMonitor()
    jetson_sampler = JetsonSampler(interval_s=args.jetson_interval)
    infer_counter = SlidingCounter(window_s=1.0)
    fps_counter = SlidingCounter(window_s=1.0)
    video_writer = None
    rendered_video_path = run_dir / "detections.mp4" if args.save_video else None
    video_fps = _video_source_fps(args.video_source)

    frames_processed = 0
    batches_processed = 0
    inference_runtime_s = 0.0
    benchmark_wall_start = time.perf_counter()
    effective_max_frames = _effective_frame_cap(args.video_source, args.max_frames)

    try:
        streamer.model.warmup(
            micro=max(1, int(args.batch_size)),
            warmup_sessions=max(1, int(args.warmup_sessions)),
        )
        jetson_sampler.start()

        for frames in _iter_video_batches(
            video_source=args.video_source,
            batch_size=args.batch_size,
            max_frames=effective_max_frames,
        ):
            batch_size = len(frames)
            images = streamer.preprocess(frames)
            gpu_stats = gpu_mon.sample()
            cpu_stats = cpu_mon.sample()
            batch_frame_start = frames_processed

            _synchronize_device(streamer.device)
            t0 = time.perf_counter()
            infer_outputs = streamer.model(images, orig_imgs=frames, debug=bool(args.verbose))
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

            if args.save_video:
                detections = _normalize_backend_outputs(
                    streamer=streamer,
                    infer_outputs=infer_outputs,
                    frames_bgr=frames,
                    frame_start_idx=batch_frame_start,
                )
                for detection in detections:
                    rendered_rgb = _render_detection_frame(streamer, detection)
                    if rendered_rgb is None:
                        continue
                    rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
                    if video_writer is None:
                        fps = video_fps if video_fps > 0.0 else 30.0
                        h, w = rendered_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(str(rendered_video_path), fourcc, fps, (w, h))
                        if not video_writer.isOpened():
                            raise RuntimeError(f"Failed to open detection video writer: {rendered_video_path}")
                    video_writer.write(rendered_bgr)

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
        if video_writer is not None:
            video_writer.release()
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
        "frame_cap": int(effective_max_frames),
        "video_seconds_cap": float(DEFAULT_VIDEO_SECONDS_CAP),
        "annotated_video_path": str(rendered_video_path) if args.save_video and rendered_video_path is not None else "",
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
    parser = argparse.ArgumentParser(
        description=(
            "Run inference-only PT/ONNX/ENGINE backend comparison on the same video. "
            "On WSL / non-Jetson hosts the .engine backend is automatically skipped "
            "when TensorRT is not installed; missing model files are also skipped. "
            "Recorded videos are capped to the first standardized evaluation window."
        )
    )
    parser.add_argument("--video-source", required=True, help="Input video path or stream URL.")
    parser.add_argument("--output-dir", default="assets/backend_inference_comparison", help="Directory where run summaries/logs are written.")
    parser.add_argument("--pt-model", default=DEFAULT_PT_MODEL, help="Path to the PT model.")
    parser.add_argument("--onnx-model", default=DEFAULT_ONNX_MODEL, help="Path to the ONNX model.")
    parser.add_argument("--engine-model", default=DEFAULT_ENGINE_MODEL, help="Path to the TensorRT engine.")
    parser.add_argument("--batch-size", type=int, default=16, help="Frames per inference batch.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional stricter frame cap. Effective limit is min(this value, standardized 1500-frame/60-second cap). Use 0 to rely only on the standardized cap.")
    parser.add_argument("--device", default="", help="Torch device override, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for utilization sampling.")
    parser.add_argument("--warmup-sessions", type=int, default=8, help="Warmup iterations before timing.")
    parser.add_argument("--jetson-interval", type=float, default=1.0, help="Sampling interval for Jetson telemetry in seconds.")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False, help="Save an annotated detection video per backend run for qualitative review.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose backend logging.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths = [args.pt_model, args.onnx_model, args.engine_model]

    # Filter to backends that can actually run in this environment.
    # On WSL / non-Jetson hosts this silently drops the .engine backend when
    # TensorRT is not installed and any model whose file is missing.
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

    dump_json(output_dir / "comparison_summary.json", summaries)
    _write_markdown_summary(output_dir / "comparison_summary.md", summaries)


if __name__ == "__main__":
    main()
