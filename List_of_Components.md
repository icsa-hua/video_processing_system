# List of Components

## Python packages apparent in the repository
- `ultralytics`: primary YOLO integration for `.pt` models, `Results` handling, inference-source loading, image-size validation, plotting helpers, and callback wiring.
- `torch`: tensor preparation, CUDA execution, warmup, device transfers, GPU stream/event synchronization, and handoff to ONNX Runtime or TensorRT wrappers.
- `torchvision`: NMS helpers via `torchvision.ops.nms` and `torchvision.ops.batched_nms`.
- `opencv-python` / `cv2`: video capture, RTSP handling, background subtraction, morphology, connected components, contour analysis, image saving, JPEG encoding/decoding, and fisheye geometry operations.
- `numpy`: frame arrays, geometry, mask processing, vectorized hazard logic, and performance/statistics support.
- `onnx`: declared dependency for model export/interchange.
- `onnxruntime`: ONNX backend execution through `InferenceSession`, preferring `CUDAExecutionProvider` and falling back to `CPUExecutionProvider`.
- `tensorrt`: engine build/load/inference for TensorRT deployment, optimization-profile setup, FP16/INT8 configuration, and serialized engine handling.
- `supervision`: tracking-oriented `Detections`, `ByteTrack`, annotators, and track-id-aware rendering support.
- `trackers`: alternate `SORTTracker` support retained beside the default `ByteTrack` path.
- `fastapi`: backend API for starting/stopping inference workers and serving MJPEG preview streams.
- `pydantic`: request/response validation for FastAPI routes.
- `streamlit`: browser UI for source/model selection and live preview control.
- `requests`: Streamlit-to-FastAPI control-plane calls.
- `paho-mqtt`: MQTT client for publishing detection or crop payloads.
- `cbor2`: CBOR serialization for MQTT messages carrying JPEG crops and metadata.
- `shapely`: ROI polygon representation and point-in-polygon region counting.
- `Pillow` / `PIL`: image loading for the UI/static assets.
- `psutil`: CPU/RAM/process monitoring for performance logging.
- `pynvml` / `nvidia-ml-py`: desktop NVIDIA GPU monitoring.
- `jtop` / `jetson-stats`: Jetson hardware monitoring.
- `memory_profiler`: decorators for memory profiling around streaming paths.
- `viztracer`: profiling support referenced by the repository.
- `matplotlib`: plotting support in benchmarking and comparison scripts.
- `pandas`: tabular analysis in performance/benchmark scripts.
- `tqdm`: progress reporting in utility scripts.
- `yaml` / `PyYAML`: dataset/config conversion utilities.

## Top-level pipeline entrypoints
- `scripts/obs_pipeline.py`: primary CLI entrypoint; builds `PipelineConfig`, validates runtime options, and launches either direct pipeline execution or the GUI/API pair.
- `obs_system/application_module/dummy_application/dummy_app.py` -> `Application`: top-level orchestrator that wires source selection, model/backend setup, logic modules, optional MQTT, execution mode, and cleanup.
- `obs_system/application_module/dummy_application/pipeline_config.py` -> `PipelineConfig`: central runtime dataclass and CLI parser for model path, stream source, ROI, MQTT, preview, benchmarking, TensorRT mode, stream limit, lane recalibration, Jetson profile, and forced tiling.

## Application / UI / API components
- `obs_system/application_module/dummy_application/backend.py`: FastAPI control service; starts/stops inference in worker processes, keeps bounded preview queues, and exposes lifecycle/status endpoints.
- `obs_system/application_module/dummy_application/intermediary.py`: launches and coordinates `uvicorn` and `streamlit run`, including shutdown handling for both processes.
- `obs_system/application_module/dummy_application/web_interface.py`: Streamlit UI for local-video/RTSP selection, model/backend choice, ROI/MQTT/TensorRT toggles, and embedded MJPEG preview.
- `obs_system/application_module/dummy_application/stream_examiner.py`: lightweight stream validator that checks whether a stream opens and yields frames before full inference is started.
- `obs_system/application_module/dummy_application/camera_config.py`: hardcoded RTSP camera definitions and derived stream URLs used by default configuration.

## Detection/model-loading components
- `obs_system/detection_module/interface/model_registry.py` -> `ModelRegistry`: extension-to-backend registry. Current default mapping is `.pt -> pt`, `.onnx -> onnx` with optional TensorRT override, and `.engine -> trt` with TensorRT required.
- `obs_system/detection_module/interface/factory.py` -> `StreamerFactory`: central builder that resolves the requested backend through the registry and returns a configured `UnifiedModelStreamer`.
- `obs_system/detection_module/dummy_predictor/stream_unified.py` -> `UnifiedModelStreamer`: main current inference path. It unifies `.pt`, `.onnx`, and `.engine` execution behind one streamer and reuses the optimized batched pipeline.
- `UnifiedModelStreamer` implementation details:
- selects backend-specific adapters instead of branching throughout the pipeline.
- keeps a single preprocessing path with backend-aware letterboxing rules.
- attaches tracking only when `type=tracking`.
- intentionally defaults to the non-tiled streaming route (`force_streaming_no_tiles = True`) unless the pipeline explicitly forces tiles elsewhere.
- `_PtAdapter` in `stream_unified.py`: wraps Ultralytics `YOLO.predict`, converts `Results` into per-frame box/score/class tensors, and supports CUDA warmup.
- `_OnnxAdapter` in `stream_unified.py`: wraps `CompressedYOLO`, returning raw detections from ONNX Runtime plus optional GPU warmup.
- `_TensorRTAdapter` in `stream_unified.py`: wraps `TensorRTYOLO`, returning TensorRT detections and CUDA completion events.
- `obs_system/detection_module/interface/streamer.py` -> `Streamer`: shared execution framework for source setup, runtime limits, preview publication, hazard recording, MQTT publication, output saving, scene-mask caching, and high-attention hazard state.
- `Streamer` implementation details:
- maintains async workers for saving, rendering/preview encoding, and I/O-heavy tasks such as MQTT publishing and hazard-event persistence.
- caches lane/crosswalk masks and their integral images to avoid repeated per-frame preprocessing.
- tracks run-level metrics such as dropped frames, first-frame time, preview emissions, and MQTT/save counters.
- supports a live-stream runtime cap and clean early termination when that cap is reached.
- `obs_system/detection_module/interface/streaming_compressed.py` -> `OptimizedStreamer`: main batched execution engine. It performs batch acquisition, motion gating, optional ROI crop, optional fisheye correction, batched inference, NMS, tracking, hazard analysis, benchmarking, and preview/output emission.
- `OptimizedStreamer` implementation details:
- stage A acquires frames, applies ROI crop and subtractor gating, and can skip batches for warmup or no-motion cases.
- stage B preprocesses, runs backend inference, applies per-frame NMS, and maps boxes back to original coordinates.
- stage C applies tracking, hazard logic, preview/output generation, MQTT publication, and performance logging.
- uses `DetectionBatch`/`FrameDetections` to keep batched detections aligned with original frames and frame ids.
- `obs_system/detection_module/interface/streaming_default.py` -> `YOLOStreamer`: older/default Ultralytics-style path retained as a legacy implementation.

## Legacy / alternate detection backends still present
- `obs_system/detection_module/dummy_predictor/stream_yolov8.py`: older YOLOv8 path based on `YOLO.track(..., tracker="bytetrack.yaml")`.
- `obs_system/detection_module/dummy_predictor/stream_yolov5.py`: older YOLOv5-oriented path with legacy `torch.hub` loading.
- `obs_system/detection_module/dummy_predictor/stream_y8_onnx.py`: older ONNX execution path with tile reconstruction and custom NMS.
- `obs_system/detection_module/dummy_predictor/stream_trt.py`: older TensorRT path with explicit tiling and GPU event synchronization.

## Model/backend implementation components
- `obs_system/compressed/interface/compressed_yolo.py` -> `CompressedYOLO`: ONNX Runtime wrapper for YOLO-style models. It prepares inputs, runs ONNX Runtime with IO binding when possible, applies confidence filtering, multiclass NMS, and returns per-image boxes/scores/classes.
- `CompressedYOLO` implementation details:
- prefers zero-copy-ish GPU IO binding when the input tensor is CUDA-backed and contiguous.
- falls back to `session.run()` if IO binding is unavailable or fails.
- handles both tensor and numpy inference inputs.
- exposes a dedicated warmup path for CUDA-backed sessions.
- `obs_system/compressed/interface/tensor_yolo.py` -> `TensorRTYOLO`: TensorRT wrapper that can build an engine from ONNX, deserialize `.engine` files, bind CUDA buffers, execute inference, and return outputs synchronized with CUDA events.
- `TensorRTYOLO` implementation details:
- builds engines under `assets/compressed_models` when a matching engine is not already available.
- uses a fixed optimization profile matching the configured batch/input shape.
- supports FP16 as the main active optimization path and retains INT8 calibrator support in code.
- stores its own CUDA stream, input/output bindings, and execution context.
- `obs_system/compressed/interface/convert_to_Results.py` -> `ConverterResults`: converts backend outputs into Ultralytics-like `Results` objects and carries the class-name vocabulary used by the rest of the pipeline.
- `obs_system/compressed/interface/utils.py`: helper utilities for lightweight NMS and multiclass NMS used by compressed backends.
- `obs_system/detection_module/interface/detection_batch.py` -> `DetectionBatch` / `FrameDetections`: lightweight containers used by the batched pipeline to represent empty/non-empty detections per frame and preserve batch index, frame id, and original image alignment.

## Logic / scene-understanding components
- `obs_system/logic_module/dummy_logic/subtractor.py` -> `Subtractor`: current scene-mask module for motion gating, lane extraction, crosswalk extraction, startup warmup, and runtime lane recalibration.
- `Subtractor` current behavior:
- uses a single MOG2 background model both for per-frame motion gating and for calibration accumulation.
- supports static-background warmup from an empty image for video or stream startup.
- supports stream-specific startup warmup where inference is held until the calibration window has completed.
- supports periodic runtime lane recalibration for long-running streams.
- keeps motion scores per frame and exposes them to the streamer for logging/metrics.
- `Subtractor` lane extraction updates:
- now includes static lane detection for sparse-traffic videos through `detect_static_lanes()` in `obs_system/utils/common.py`.
- samples a stabilized background image after a warmup window and extracts lane corridors from bright road markings.
- treats the static-lane mask as the primary source when motion density is too low for reliable vehicle-path accumulation.
- blends static geometry with accumulated motion masks when traffic is dense enough, yielding a hybrid lane mask.
- `Subtractor` implementation components:
- MOG2 foreground extraction with resolution-cached thresholds.
- hysteresis/hold logic to avoid rapid motion-state flapping.
- calibration accumulator that reuses raw MOG2 output instead of running a second subtractor.
- hybrid lane-mask merge between motion accumulation and static-road-marking extraction.
- crosswalk extraction from the calibrated lane region using classical image-processing heuristics.
- scene-mask serving through `get_scene_masks()`, including optional dilation for high-attention mode.
- `obs_system/utils/common.py` -> `detect_static_lanes`: static-lane helper that enhances bright road markings with CLAHE and top-hat morphology, dilates them into lane-width corridors, and filters components by area. This is the main recent addition for sparse-traffic scenes.
- `obs_system/logic_module/dummy_logic/region_setter.py` -> `RegionSetter`: ROI module that defines a rectangular road-only region, crops frames before inference, translates detections back to full-frame coordinates, and counts tracked centroids inside the ROI.
- `RegionSetter` implementation details:
- scales hardcoded `640x640` reference coordinates to the actual frame size.
- falls back to full-frame inference if the configured ROI is invalid for the current source.
- handles both frame cropping and post-inference box translation.
- `obs_system/logic_module/dummy_logic/obstacle_filtering.py` -> `analyze_lane_hazards`: rule-based hazard classifier operating on detections plus lane/crosswalk masks.
- `Obstacle filtering` implementation details:
- uses vectorized overlap tests with optional integral-image acceleration.
- filters out allowed vehicles and allowed pedestrian-in-crosswalk cases.
- classifies lane obstacles into categories such as `pedestrian_in_lane`, `animal_on_road`, `small_debris`, `edge_small_object`, `large_static_object`, and `unknown_obstruction`.
- derives risk from vertical position inside the lane mask and from size/edge heuristics.
- emits structured metadata used for preview overlays, CSV logging, and MQTT action hints.
- `obs_system/logic_module/dummy_logic/tracker_sv.py` -> `TrackerHandler`: tracking wrapper currently defaulting to `supervision.ByteTrack`, with optional `SORTTracker` retained.
- `TrackerHandler` implementation details:
- converts raw detections into `supervision.Detections`.
- updates tracked detections and returns Ultralytics-like `Results`.
- stores recent per-track center history for visualization and trajectory persistence.
- can increase or reduce history persistence dynamically when the streamer enters or leaves high-attention hazard mode.
- `obs_system/logic_module/dummy_logic/fisheye.py` -> `FishEyeProjection`: fisheye-handling module for view remapping and box projection between corrected and original coordinates.
- `obs_system/logic_module/dummy_logic/overlap_detection.py`: auxiliary overlap/geometry placeholder.
- `obs_system/logic_module/dummy_logic/homography.py`: homography-related placeholder.

## Communication/output components
- `obs_system/communication_module/mqtt_com/message_transmitter.py` -> `RealMQTT`: basic MQTT interface wrapper.
- `obs_system/communication_module/mqtt_com/message_transmitter.py` -> `CBORMQTTCropClientCV2`: current MQTT crop publisher/subscriber path. It encodes crops with OpenCV, wraps them in CBOR, and publishes/decodes detection batches.
- `obs_system/communication_module/mqtt_com/config.py`: broker, topic, TLS, QoS, JPEG, and asset-path configuration.
- `obs_system/detection_module/interface/streamer.py` save/output pipeline:
- asynchronous save worker for non-blocking disk output.
- hazard frame and crop persistence under `assets/hazard_events`.
- CSV event logging with timestamp, class, category, risk, overlaps, and saved artifact paths.
- MJPEG preview publication through bounded multiprocessing queues.
- dedicated async I/O path for MQTT results and no-detection messages.

## Benchmarking / profiling / monitoring components
- `obs_system/utils/benchmarking/metrics/model_performance.py` -> `ModelPerf`: detection benchmarking with precision, recall, F1, IoU thresholds, and optional COCO-style interpolation.
- `obs_system/utils/benchmarking/metrics/pc_performance.py`: per-run and per-frame performance logging, sliding counters, CPU monitors, GPU monitors, and timeline logging.
- `obs_system/utils/appraisal.py`: `StepContext` timing wrappers and utility helpers used to measure setup and stage durations.
- `obs_system/utils/common.py`: shared helpers for GPU existence checks, model alias resolution, frame-id extraction, empty-result creation, static-lane detection, and general pipeline utilities.
- `obs_system/utils/tiles.py`: tile construction, padding, reconstruction metadata, and letterboxing helpers for the tiled inference path.
- repository scripts for benchmarking/plots:
- `scripts/run_backend_comparison.py`: end-to-end comparison of `.pt`, `.onnx`, and `.engine` on the same source.
- `scripts/test_backend_inference.py`: backend-only inference comparison without the full pipeline.
- `scripts/bench_postprocess.py`: focused micro-benchmarks for hazard logic, scene-mask caching, and postprocess hot paths.
- `scripts/plot_perf_extended.py`
- `scripts/plot_perf_comparisons.py`
- `scripts/fine_tuned_tester.py`
- `scripts/sampler.py`

## Data-prep / conversion scripts apparent in repo
- `scripts/convert_fisheye8k_to_yolo.py`: converts fisheye-oriented datasets into YOLO format.
- `scripts/create_unified_yaml.py`: dataset/class YAML generation utility.
- `scripts/convert_png_jpg_and_store.py`: image conversion helper.
- `scripts/zone_cleaner.py`: annotation/zone cleanup utility.
- `scripts/test_fisheye.py`: fisheye-processing test helper.

## Models, trackers, and inference methods explicitly apparent
- YOLOv8 `.pt` inference through Ultralytics.
- YOLO-style `.onnx` inference through `CompressedYOLO` + ONNX Runtime.
- TensorRT `.engine` inference through `TensorRTYOLO`.
- `.onnx` models can also be routed through TensorRT when `use_TRT` is enabled.
- ByteTrack is the current default tracker through `supervision.ByteTrack`.
- SORT remains as an alternate tracker implementation.
- MOG2 remains the motion gate and one source of lane-path accumulation.
- lane and crosswalk reasoning remain classical CV / heuristic methods rather than learned segmentation.
- static lane detection from a stabilized background image is now a first-class path for sparse-traffic scenes and is merged with motion-based calibration when useful.

## Hardcoded defaults, assumptions, and code admissions
- The system is designed around static roadside cameras; ROI coordinates, lane masks, and crosswalk heuristics are camera-view-specific.
- `PipelineConfig.DEFAULT_MODEL` is `assets/compressed_models/yolov8s.engine`.
- `PipelineConfig.DEFAULT_VIDEO_SOURCE` is `RECTILINEAR_RTSP` from `camera_config.py`.
- RTSP capture is forced to TCP through `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`.
- `stream_limit_hours` defaults to `1.0` for live streams and `0` disables the limit.
- preview defaults are `preview_max_width=960`, `preview_jpeg_quality=70`, `preview_fps=8.0`.
- `lane_recalibration_interval_frames` defaults to `0`, meaning runtime lane recalibration is disabled unless explicitly enabled.
- Jetson profile support is present through `jetson_profile`, `jetson_hazard_scale`, and `jetson_cpu_threads`.
- `force_tiles` exists in configuration, but the main unified streamer still defaults to the non-tiled route unless tile use is explicitly forced by execution logic.
- model validation rules are explicit:
- `.engine` requires TensorRT mode.
- TensorRT mode is allowed only for `.onnx` or `.engine`.
- default thresholds in `global_config.py` are `CONF_THR=0.25`, `NMS_IOU=0.45`, `CLASS_AGNOSTIC=True`.
- batching defaults are `BATCH_SIZE=16` and `WARM_UP_SESSIONS=8`.
- tiling defaults are `TILE_SIZE=640`, `TILE_OVERLAP=0.25`, `TILE_THR=3`.
- ROI defaults are hardcoded from a `640x640` reference frame:
- `ROI_X1=0`, `ROI_Y1=639`, `ROI_X2=430`, `ROI_Y2=300`.
- invalid ROI bounds fall back to full-frame inference.
- subtractor/motion defaults are still code-driven:
- `TRIALS=10`
- `HISTORY=300`
- `VARTHRESHOLD=16`
- `THR_RATIO=0.2`
- `K_CONSECUTIVE=3`
- `HOLD_FRAMES=10`
- `MIN_OBJ_AREA=0.003`
- the non-stream warmup path still assumes an empty-background image may be provided, commonly `samples/highway_rescaled.png`.
- live streams can hold inference during startup warmup while the subtractor accumulates enough frames to produce a usable scene mask.
- sparse-traffic handling now assumes a stable enough background image can reveal lane markings; static-lane extraction is based on bright painted markings rather than learned semantics.
- static-lane extraction details currently include:
- CLAHE enhancement of the background image.
- top-hat morphology with a large rectangular kernel to isolate bright markings.
- dilation/closing to expand markings into lane-width corridors.
- connected-component filtering using a minimum area ratio.
- hybrid lane fusion uses a rolling motion-density window:
- low-density traffic keeps the static-lane mask as primary.
- denser traffic merges motion accumulation with the static mask.
- crosswalk extraction remains non-ML and assumes stripe-like bright blobs inside the lane mask.
- hazard classification policy remains explicit:
- persons in crosswalk are tolerated.
- persons in lane outside crosswalk are hazards.
- allowed vehicles are not lane hazards by default.
- animals and debris-like objects are escalated according to overlap, size, and position heuristics.
- high-attention hazard mode is hardcoded:
- default countdown `18` frames.
- risk-based bonus frames `+10` high, `+6` medium, `+3` low.
- lane dilation in high-attention mode is `14` pixels.
- tracker history increases from `30` to `60` in high-attention mode, or to smaller capped values when Jetson profile is active.
- ONNX Runtime provider preference is:
- `CUDAExecutionProvider`
- `CPUExecutionProvider`
- TensorRT engine-building assumptions are explicit:
- workspace memory pool limit `1 << 30`.
- runtime input shape `[16, 3, 640, 640]`.
- auto-derived engine filenames stored under `assets/compressed_models`.
- MQTT defaults currently assume:
- broker `edgejet3vpn.edi.lv`
- port `8884`
- topic `reid-vehicle-detection`
- QoS `1`
- keepalive `60`
- JPEG quality `75`
- sender TLS assets under `assets/mqtt_credentials/...`
- `CREATE_SUBSCRIBER = False` by default.
- the web UI exposes three hardcoded model choices:
- TensorRT engine: `assets/compressed_models/mixed_dataset_trained_yolov8s_mixed_batch_trt_fp16_noint8.engine`
- YOLOv8 ONNX: `assets/compressed_models/mixed_dataset_trained_yolov8s.onnx`
- YOLOv8 PT: `assets/compressed_models/yolov8s.pt`
- `camera_config.py` still contains hardcoded camera credentials, hosts, and ports.
- `stream_y8_onnx.py` still contains a hardcoded ONNX path independent of the passed model argument.
- the README still notes that a modified Ultralytics `loaders.py` may be required for the intended crop/zoom behavior.

## Output artifacts produced by the pipeline
- `runs/det/...`: Ultralytics-style rendered detection outputs.
- `assets/background_check/`: subtractor and scene-mask debug images.
- `assets/hazard_events/frames`: saved hazard frames.
- `assets/hazard_events/crops`: saved hazard-object crops.
- `assets/hazard_events/hazard_events.csv`: persisted structured hazard log.
- `assets/mqtt/saved_publishes.cbor`: archived outbound MQTT payloads.
- `assets/mqtt/received_vehicle_crops`: decoded inbound MQTT JPEG crops.
- `assets/perf_logs/*.csv` and `*.jsonl`: performance logs, frame-level logs, and timeline traces.
- `assets/trace_jsons/*.json`: optional profiler traces exported during verbose first-batch backend profiling.
