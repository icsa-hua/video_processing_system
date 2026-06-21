# Obstacle Recognition Edge AI

Obstacle recognition and tracking pipeline for roadside perception video streams, with support for PyTorch, ONNX, and TensorRT backends.

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)

## Repository

- Remote: `https://github.com/icsa-hua/video_processing_system.git`
- Current development branch: `claude_v2_deploy`

## Overview

This repository contains a video processing pipeline for static roadside cameras. It performs:

- motion gating with background subtraction
- object detection
- tracking
- hazard logic
- optional MQTT publishing
- optional preview / saved outputs
- backend benchmarking for `.pt`, `.onnx`, and `.engine`

The current pipeline and benchmark flows have been exercised on:

- WSL
- NVIDIA Jetson

## Requirements

### Software

- Python 3.10
- PyTorch
- Ultralytics
- OpenCV
- ONNX / ONNX Runtime
- paho-mqtt
- FastAPI
- Streamlit
- Shapely

### Installation

Use either:

```bash
pip install -r requirements.txt
```

or:

```bash
pip install -e .
```

### Hardware

The pipeline can run on CPU-only systems, but GPU execution is strongly recommended. Jetson devices are supported, and TensorRT is intended primarily for Jetson deployment.

## Getting Started

Clone the repository:

```bash
git clone -b claude_v2_deploy https://github.com/icsa-hua/video_processing_system.git
cd video_processing_system
```

Using a virtual environment is recommended.

If module resolution is inconsistent in your shell, set:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

## Main Pipeline Execution

The main entry point is:

```bash
python3 scripts/obs_pipeline.py
```

### Common examples

Run the pipeline with a specific video:

```bash
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4
```

Run with a specific model:

```bash
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4 --model_name assets/compressed_models/yolov8s.pt
```

Run a TensorRT engine:

```bash
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4 --model_name assets/compressed_models/yolov8s.engine --use_TRT
```

For the 720×720 overhead fisheye road camera, select the isolated road-focused
tangent-view profile:

```bash
python3 scripts/obs_pipeline.py \
  --video_source assets/runs/jetson_2_recording.mp4 \
  --model_name assets/compressed_models/edi_jetson_model.engine \
  --use_TRT \
  --fep \
  --fisheye_profile jetson_2_road
```

The profile keeps the existing six-view inference budget, focuses those views
on the upper road and upper-left approach, persists recently active views for
eight frames, and performs a complete road-view refresh every 50 frames.
Omitting `--fisheye_profile` preserves the previous fisheye behaviour.

Show preview:

```bash
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4 --show
```

Verbose detections and logs:

```bash
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4 --verbose
```

Enable ROI:

```bash
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4 --roi
```

Select a specific ROI profile explicitly:

```bash
python3 scripts/obs_pipeline.py --video_source samples/test_samples/highway.mp4 --roi --roi_profile highway
```

Enable MQTT:

```bash
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4 --mqtt
```

Enable fisheye projection:

```bash
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4 --fep
```

Save rendered outputs:

```bash
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4 --save
```

Enable benchmark accounting:

```bash
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4 --bench --bench-labels samples/labels --plot_perf
```

Run the Jetson execution branch:

```bash
python3 scripts/obs_pipeline.py \
  --video_source samples/sample_video.mp4 \
  --model_name assets/compressed_models/yolov8s.engine \
  --use_TRT \
  --jetson_profile \
  --jetson_hazard_scale 0.5 \
  --jetson_cpu_threads 2
```

Launch the web interface:

```bash
python3 scripts/obs_pipeline.py --gui
```

## Main Pipeline CLI Choices

The primary execution script supports the following options through `obs_system/application_module/dummy_application/pipeline_config.py`:

- `--model_name`
  Model path or model file to load. Supports `.pt`, `.onnx`, and `.engine`.
- `--video_source`
  Local video path, RTSP source, or configured stream source.
- `--type`
  Pipeline mode. Current default is `tracking`.
- `--gui` / `--no-gui`
  Launch the Streamlit/FastAPI interface instead of direct CLI execution.
- `--mqtt` / `--no-mqtt`
  Enable or disable MQTT publishing.
- `--show` / `--no-show`
  Enable preview generation.
- `--verbose` / `--no-verbose`
  Enable detailed console logging.
- `--port_address`
  GUI / service port.
- `--host_address`
  GUI / service host address.
- `--save` / `--no-save`
  Save rendered outputs.
- `--roi` / `--no-roi`
  Enable region-of-interest cropping.
- `--roi_profile`
  Optional ROI profile key from [obs_system/utils/roi_profiles.json](/Users/jimborg/WorkSpace/edgeai/video_processing_system/obs_system/utils/roi_profiles.json). If omitted, the pipeline tries the source path, filename, and stem automatically.
- `--half` / `--no-half`
  Enable reduced resource / half-style execution path where supported.
- `--fep` / `--no-fep`
  Enable fisheye projection logic.
- `--bench` / `--no-bench`
  Enable benchmark scoring against ground truth labels.
- `--bench-labels`
  Ground-truth label directory for benchmark mode.
- `--use_TRT` / `--no-use_TRT`
  Required when using `.engine` TensorRT models.
- `--plot_perf` / `--no-plot_perf`
  Save pipeline performance logs for later analysis.
- `--only_FPS` / `--no-only_FPS`
  Track FPS-focused execution without full plotting requirements.
- `--stream_limit_hours`
  Runtime cap for live streams. `0` disables the cap.
- `--lane_recalibration_interval_frames`
  Periodic lane recalibration interval for long-running streams.
- `--jetson_profile` / `--no-jetson_profile`
  Enable the Jetson-optimized execution branch.
- `--jetson_hazard_scale`
  Downscale factor for Jetson hazard-mask processing. Valid range: `(0, 1]`.
- `--jetson_cpu_threads`
  CPU thread cap for the Jetson branch. `0` keeps the default runtime behavior.

## ROI Profiles For Experiments

Per-video ROI overrides live in [obs_system/utils/roi_profiles.json](/Users/jimborg/WorkSpace/edgeai/video_processing_system/obs_system/utils/roi_profiles.json).

- The `default` entry preserves the existing static ROI rectangle.
- Profiles can be selected explicitly with `--roi_profile`, or matched automatically from the video source path, filename, or stem.
- Each profile stores `x1`, `y1`, `x2`, `y2` together with `reference_width` and `reference_height`.
- The sample-video profiles are keyed for `MVI_39401`, `highway`, `fisheye`, and `short_1920_12fps`.

## Benchmark and Test Scripts

### 1. Full pipeline backend comparison

```bash
python3 scripts/run_backend_comparison.py --video-source samples/sample_video.mp4
```

This runs `.pt`, `.onnx`, and `.engine` through the main pipeline on the same video and stores:

- per-backend `summary.json`
- `perf_log.csv`
- `perf_frames.csv`
- `perf_timeline.jsonl`
- `comparison_summary.json`
- `comparison_summary.md`

Important toggles:

- `--roi` / `--no-roi`
- `--fep` / `--no-fep`
- `--mqtt` / `--no-mqtt`
- `--save-outputs` / `--no-save-outputs`
- `--verbose` / `--no-verbose`
- `--labels-dir`
- `--stream-limit-hours`
- `--jetson-profile`
- `--jetson-hazard-scale`
- `--jetson-cpu-threads`

### 2. Inference-only backend comparison

```bash
python3 scripts/test_backend_inference.py --video-source samples/sample_video.mp4
```

This compares `.pt`, `.onnx`, and `.engine` on the same video while timing backend inference only. It is useful for separating raw model throughput from the end-to-end pipeline.

### 3. Other scripts

Additional helper scripts live in `scripts/`, including:

- `plot_perf_comparisons.py`
- `plot_perf_extended.py`
- `bench_postprocess.py`
- `fine_tuned_tester.py`
- `test_fisheye.py`

## Docker

This repository includes:

- `Dockerfile`
- `docker-compose.yml`

### Build and start

```bash
docker compose up --build -d
```

Enter the container:

```bash
docker exec -it dev_cont bash
```

Inside the container, the repository is mounted at:

```bash
/workspace
```

Typical in-container execution:

```bash
cd /workspace
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4
```

### Docker notes

- The compose setup is configured for NVIDIA runtime use.
- The current image is Jetson-oriented and based on `ultralytics/ultralytics:latest-jetson-jetpack6`.
- The repository is bind-mounted into the container, so code edits on the host are immediately visible inside the container.
- `jtop` socket passthrough is configured in `docker-compose.yml` for Jetson telemetry access when available.

## Important Jetson TensorRT Note

When first going to use the Docker environment on a Jetson device, the TensorRT options require a model generated on your specific Jetson.

Recommended workflow:

1. Enter the Docker environment.
2. Export a YOLO model to ONNX from Python.
3. Convert that ONNX model to a TensorRT engine on the Jetson.
4. Pass that resulting `.engine` file through `--model_name`.

Example ONNX export from Python with Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(format="onnx")
```

Then convert ONNX to TensorRT inside the container, typically with:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=/path/to/model.onnx --saveEngine=/path/to/model.engine
```

Then run the pipeline with:

```bash
python3 scripts/obs_pipeline.py --video_source samples/sample_video.mp4 --model_name /path/to/model.engine --use_TRT
```

## Notes on Model Selection

The pipeline accepts:

- `.pt`
- `.onnx`
- `.engine`

For TensorRT:

- `.engine` requires `--use_TRT`
- on Jetson, the `.engine` should be built on the target device for best compatibility

## Project Structure

- `scripts/`
  Main execution and benchmarking scripts.
- `obs_system/application_module/`
  Application setup, configuration, GUI/backend integration.
- `obs_system/detection_module/`
  Streamers, model adapters, inference flow.
- `obs_system/logic_module/`
  ROI, subtractor, hazard logic, tracking helpers.
- `obs_system/communication_module/`
  MQTT integration.
- `assets/`
  Models, benchmark outputs, saved MQTT payloads, hazard outputs.

## Troubleshooting

### TensorRT engine does not load

- Ensure the engine was built for the same Jetson device and software stack.
- Ensure you passed `--use_TRT`.
- Ensure `--model_name` points to the `.engine` file.

### Benchmark numbers differ between scripts

That is expected:

- `scripts/test_backend_inference.py` measures inference-only throughput.
- `scripts/run_backend_comparison.py` measures end-to-end pipeline throughput.

### Labels path error in benchmark mode

If `--bench` is enabled, ensure `--bench-labels` points to an existing label directory.

### Preview / GUI issues in Docker

- Use the web interface path with `--gui` when appropriate.
- For headless Docker usage, prefer saved outputs or the browser-based interface instead of direct local window display.

### MQTT issues

- Verify broker certificates and MQTT configuration under `obs_system/communication_module/mqtt_com/`.
- For pure throughput testing, keep `--no-mqtt`.
