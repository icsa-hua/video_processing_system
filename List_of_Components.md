# List of Components

## Python packages apparent in the repository
- `ultralytics`: primary YOLO integration, `YOLO` model loading, `Results`, letterboxing, source loaders, plotting helpers, callback integration, image-size checks.
- `torch`: tensor handling, CUDA execution, warmup, tensor conversion, GPU stream/event synchronization, TensorRT/ONNX handoff.
- `torchvision`: NMS utilities via `torchvision.ops.nms` and `torchvision.ops.batched_nms`.
- `opencv-python` / `cv2`: video capture, RTSP handling, background subtraction, morphology, contour extraction, encoding/decoding JPEG, drawing, connected components, fisheye geometry.
- `numpy`: frame arrays, geometry, thresholds, statistics, mask processing, benchmarking math.
- `onnx`: declared dependency for model export/interchange.
- `onnxruntime`: ONNX inference backend through `InferenceSession` with `CUDAExecutionProvider` and `CPUExecutionProvider`.
- `tensorrt`: TensorRT engine build/load/inference, optimization profiles, FP16/INT8 configuration.
- `supervision`: `ByteTrack`, `Detections`, `BoxAnnotator`, color palette for tracked outputs.
- `trackers`: alternate tracker support via `SORTTracker`.
- `fastapi`: backend API for starting/stopping inference and serving MJPEG preview streams.
- `pydantic`: request schemas for FastAPI.
- `streamlit`: web UI for selecting source/model/options and viewing the preview stream.
- `requests`: Streamlit frontend -> FastAPI backend HTTP calls.
- `paho-mqtt`: MQTT publisher/subscriber client.
- `cbor2`: CBOR serialization for MQTT payloads containing JPEG crops and metadata.
- `shapely`: ROI polygon representation and point-in-polygon checks.
- `Pillow` / `PIL`: Streamlit logo/image handling.
- `psutil`: RAM/process statistics and CPU monitoring.
- `pynvml` / `nvidia-ml-py`: desktop NVIDIA GPU memory/utilization monitoring.
- `jtop` / `jetson-stats`: Jetson hardware monitoring.
- `memory_profiler`: decorators on streaming methods.
- `viztracer`: declared/per-README profiling support.
- `matplotlib`: plotting support in repo scripts.
- `pandas`: imported in plotting/benchmark scripts.
- `tqdm`: progress support in scripts.
- `yaml` / `PyYAML`: YAML conversion scripts.

## Top-level pipeline entrypoints
- `scripts/obs_pipeline.py`: CLI entrypoint; builds `PipelineConfig`, warns that changing the video source requires changing the background subtractor image, launches GUI mode or direct application mode.
- `obs_system/application_module/dummy_application/dummy_app.py` -> `Application`: orchestrates source setup, model setup, logic setup, optional MQTT setup, stream execution, runtime statistics, cleanup.
- `obs_system/application_module/dummy_application/pipeline_config.py` -> `PipelineConfig`: central runtime configuration for model, source, ROI, MQTT, TensorRT, GUI, preview, benchmarking, live-stream runtime cap.

## Application / UI / API components
- `obs_system/application_module/dummy_application/backend.py`: FastAPI service; spawns separate worker processes for inference and stream examination, keeps MJPEG preview queues, exposes start/stop/status routes.
- `obs_system/application_module/dummy_application/intermediary.py`: launches `uvicorn` for FastAPI and `streamlit run` for the UI, handles shutdown of both processes.
- `obs_system/application_module/dummy_application/web_interface.py`: Streamlit UI; provides local-video/live-stream source selection, model selection, ROI/MQTT/TensorRT toggles, preview embedding, and stop controls.
- `obs_system/application_module/dummy_application/stream_examiner.py`: lightweight RTSP/stream validator and previewer using OpenCV only; used to verify that a stream can open and produce frames before/without full inference.
- `obs_system/application_module/dummy_application/camera_config.py`: hardcoded RTSP camera credentials, ports, and derived stream URLs.

## Detection/model-loading components
- `obs_system/detection_module/interface/model_registry.py`: model extension registry; resolves `.pt` -> PyTorch, `.onnx` -> ONNX or TensorRT mode, `.engine` -> TensorRT-only.
- `obs_system/detection_module/interface/factory.py` -> `StreamerFactory`: builds the unified streamer after backend resolution.
- `obs_system/detection_module/dummy_predictor/stream_unified.py` -> `UnifiedModelStreamer`: main current inference path; wraps PT/ONNX/TRT backends under one streamer and attaches optional tracking.
- `_PtAdapter` in `stream_unified.py`: loads Ultralytics `YOLO`, runs `.predict()`, returns boxes/scores/classes tensors.
- `_OnnxAdapter` in `stream_unified.py`: loads `CompressedYOLO`, returns ONNX detections.
- `_TensorRTAdapter` in `stream_unified.py`: loads `TensorRTYOLO`, runs TensorRT FP16 inference.
- `obs_system/detection_module/interface/streamer.py` -> `Streamer`: common inference framework for source setup, preview publication, hazard recording, save queue, MQTT publishing, runtime limit enforcement.
- `obs_system/detection_module/interface/streaming_compressed.py` -> `OptimizedStreamer`: main batched/compressed inference implementation with ROI cropping, motion gating, optional tiling path, performance logging, postprocessing.
- `obs_system/detection_module/interface/streaming_default.py` -> `YOLOStreamer`: older/default Ultralytics-style streamer path with ROI and subtractor integration.

## Legacy / alternate detection backends still present
- `obs_system/detection_module/dummy_predictor/stream_yolov8.py`: older YOLOv8 streamer using `YOLO.track(..., tracker="bytetrack.yaml")`.
- `obs_system/detection_module/dummy_predictor/stream_yolov5.py`: older YOLOv5 streamer; legacy `torch.hub` path for autoshape and tracking path through Ultralytics `YOLO`.
- `obs_system/detection_module/dummy_predictor/stream_y8_onnx.py`: older ONNX streamer with tile reconstruction and NMS.
- `obs_system/detection_module/dummy_predictor/stream_trt.py`: older TensorRT streamer with tiling and GPU event synchronization.

## Model/backend implementation components
- `obs_system/compressed/interface/compressed_yolo.py` -> `CompressedYOLO`: ONNX Runtime YOLO wrapper; uses IO binding, per-image postprocessing, confidence filtering, multiclass NMS, optional CUDA warmup.
- `obs_system/compressed/interface/tensor_yolo.py` -> `TensorRTYOLO`: TensorRT engine wrapper; can build an engine from ONNX, deserialize engines, bind buffers, run FP16/INT8 inference, and return CUDA completion events.
- `obs_system/compressed/interface/convert_to_Results.py` -> `ConverterResults`: converts raw detection tensors into Ultralytics-like `Results`, stores COCO-style class-name vocabulary.
- `obs_system/compressed/interface/utils.py`: lightweight NMS and multiclass NMS helpers used by compressed backends.

## Logic / scene-understanding components
- `obs_system/logic_module/dummy_logic/subtractor.py` -> `Subtractor`: OpenCV MOG2 motion-gating + lane-mask calibration + non-ML crosswalk extraction.
- `Subtractor` techniques/methods:
- `cv2.createBackgroundSubtractorMOG2` for motion gating and separate calibration background model.
- binary thresholding, morphological open/close, contour filtering, hysteresis, foreground-ratio scoring.
- accumulated motion mask calibration to derive lane region.
- non-ML crosswalk detection using grayscale masking, Gaussian blur, top-hat morphology, Otsu thresholding, contour heuristics, connected components.
- `obs_system/logic_module/dummy_logic/region_setter.py` -> `RegionSetter`: ROI cropper using a rectangular `shapely.Polygon`; translates cropped-image detections back to full-frame coordinates.
- `RegionSetter` techniques/methods:
- ratio-based ROI scaling from `640x640` reference coordinates to actual frame size.
- crop-only inference region.
- point-in-polygon counting for tracked centroids.
- `obs_system/logic_module/dummy_logic/obstacle_filtering.py` -> `analyze_lane_hazards`: rule-based hazard classification from detections + lane/crosswalk masks.
- `Obstacle filtering` techniques/methods:
- overlap-ratio tests against lane/crosswalk masks.
- explicit class allowlists/denylists for vehicles, animals, debris.
- risk estimation from vertical position in the lane mask.
- size heuristics and lane-edge proximity heuristics.
- `obs_system/logic_module/dummy_logic/tracker_sv.py` -> `TrackerHandler`: tracking wrapper; defaults to `supervision.ByteTrack`, keeps per-track history, emits Ultralytics `Results`.
- `obs_system/logic_module/dummy_logic/fisheye.py` -> `FishEyeProjection`: geometry-aware fisheye handling with remap-view generation and back-projection of boxes to original fisheye coordinates.
- `obs_system/logic_module/dummy_logic/overlap_detection.py`: auxiliary geometry/overlap logic placeholder.
- `obs_system/logic_module/dummy_logic/homography.py`: homography-related placeholder module.

## Communication/output components
- `obs_system/communication_module/mqtt_com/message_transmitter.py` -> `RealMQTT`: simple MQTT interface wrapper.
- `obs_system/communication_module/mqtt_com/message_transmitter.py` -> `CBORMQTTCropClientCV2`: main MQTT crop publisher/subscriber; encodes JPEG crops with OpenCV, wraps them in CBOR, publishes/decodes batches.
- `obs_system/communication_module/mqtt_com/config.py`: broker/topic/TLS/QoS/JPEG settings and asset paths.
- `obs_system/detection_module/interface/streamer.py` save pipeline:
- asynchronous save worker thread.
- frame/video saving through OpenCV `VideoWriter`.
- hazard-event frame/crop saving to `assets/hazard_events`.
- hazard CSV append logic.
- MJPEG browser preview publishing through bounded multiprocessing queues.

## Benchmarking / profiling / monitoring components
- `obs_system/utils/benchmarking/metrics/model_performance.py` -> `ModelPerf`: mAP, precision, recall, F1 benchmarking using IoU thresholds and optional COCO-style 101-point interpolation.
- `obs_system/utils/benchmarking/metrics/pc_performance.py`: CSV/JSONL performance logging, sliding inference-rate counter, CPU/GPU monitors.
- `obs_system/utils/appraisal.py`: step/performance context helpers used around setup/inference stages.
- `obs_system/utils/common.py`: GPU existence check, model alias resolution, frame-id extraction, misc utility helpers.
- `obs_system/utils/tiles.py`: tiling, padding, letterboxing, tile reconstruction metadata.
- repo scripts for benchmarking/plots:
- `scripts/plot_perf_extended.py`
- `scripts/plot_perf_comparisons.py`
- `scripts/fine_tuned_tester.py`
- `scripts/sampler.py`

## Data-prep / conversion scripts apparent in repo
- `scripts/convert_fisheye8k_to_yolo.py`: dataset conversion toward YOLO format.
- `scripts/create_unified_yaml.py`: dataset/class YAML creation.
- `scripts/convert_png_jpg_and_store.py`: image conversion utility.
- `scripts/zone_cleaner.py`: annotation/zone-cleaning helper.
- `scripts/test_fisheye.py`: fisheye testing utility.

## Models, trackers, and inference methods explicitly apparent
- YOLOv8 pretrained models via Ultralytics.
- YOLOv5 support still present in legacy streamers.
- ONNX-compressed YOLO inference through ONNX Runtime.
- TensorRT FP16 engine inference through custom wrapper.
- optional TensorRT INT8 calibration class exists (`YOLOInt8Calibrator`) but main unified path uses FP16.
- ByteTrack appears in two ways:
- current unified tracker path: `supervision.ByteTrack`.
- older Ultralytics path: `YOLO.track(..., tracker="bytetrack.yaml")`.
- SORT tracker support exists as an alternative in `tracker_sv.py`.
- MOG2 background subtraction is the motion gate and lane-calibration basis.
- Lane and crosswalk reasoning are classical CV / heuristic methods, not learned segmentation models.

## Hardcoded defaults, assumptions, and code admissions
- The code explicitly warns: if the input video source changes, the background subtractor image should also change, otherwise frames may be classified incorrectly as “without movement”.
- `PipelineConfig.DEFAULT_MODEL` is `assets/compressed_models/yolov8s.engine`.
- `PipelineConfig.DEFAULT_VIDEO_SOURCE` is the hardcoded `RECTILINEAR_RTSP` stream from `camera_config.py`.
- RTSP capture is forced to TCP via `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`.
- `stream_limit_hours` defaults to `1.0` for live streams.
- preview defaults are `preview_max_width=960`, `preview_jpeg_quality=70`, `preview_fps=8.0`.
- live-stream examination timeouts are hardcoded to `15s` for open and `20s` for first frame.
- FastAPI preview queues are bounded to size `2`.
- model validation assumes:
- `.engine` models must be used with `--use_TRT`.
- TensorRT mode is only allowed for `.onnx` or `.engine`.
- default detection thresholds in `global_config.py` are `CONF_THR=0.25`, `NMS_IOU=0.45`, `CLASS_AGNOSTIC=True`.
- batching defaults are `BATCH_SIZE=16` and `WARM_UP_SESSIONS=8`.
- tiling defaults are `TILE_SIZE=640`, `TILE_OVERLAP=0.25`, `TILE_THR=3`.
- the unified streamer currently sets `force_streaming_no_tiles = True`, so the tiled path exists but is intentionally bypassed in the main unified route.
- ROI defaults are hardcoded from a `640x640` reference frame:
- `ROI_X1=0`, `ROI_Y1=639`, `ROI_X2=430`, `ROI_Y2=300`.
- the ROI code explicitly states that the region is camera-feed-specific and meant to cover the road network only.
- if the ROI resolves to an invalid rectangle, the logic falls back to full-frame inference.
- background subtraction defaults are:
- `TRIALS=10`
- `HISTORY=300`
- `VARTHRESHOLD=16`
- `THR_RATIO=0.2`
- `K_CONSECUTIVE=3`
- `HOLD_FRAMES=10`
- `MIN_OBJ_AREA=0.003`
- `EMPTY_IMAGE_PATH="samples/highway_rescaled.png"`
- the non-stream warmup path assumes a static empty-background image is available at `samples/highway_rescaled.png`.
- live streams use subtractor startup warmup based on `accum_time=500` frames before inference is considered ready if no static background has been preloaded.
- lane calibration uses:
- accumulated motion masks.
- connected-component minimum area `15000`.
- Gaussian blur `(11,11)` and threshold `50` for final lane mask.
- motion gating logic assumes motion if either:
- foreground pixels exceed `threshold_ratio * frame_pixels`.
- or the largest contour area exceeds `MIN_OBJ_AREA * frame_pixels`.
- subtractor code explicitly notes an edge case: a single moving car may fail the motion test.
- crosswalk detection is explicitly non-ML and assumes:
- zebra-like bright stripes exist inside the calibrated lane mask.
- top-hat kernel `(17,17)`.
- contour aspect ratio must exceed `2.2`.
- contour extent must exceed `0.30`.
- at least `3` valid stripe blobs must be found.
- crosswalk connected-component area must exceed `max(200, 0.0025 * lane_area)`.
- hazard classification policy in code is explicit:
- person in crosswalk is allowed.
- person in lane outside crosswalk is a hazard.
- allowed vehicles are ignored as hazards.
- non-allowed objects in lane are hazards.
- lane and crosswalk overlap thresholds are both `0.20`.
- hazard risk is position-based:
- lower in the lane image -> higher risk.
- `y` position >= `70%` of lane span -> `high`.
- `y` position >= `40%` of lane span -> `medium`.
- otherwise `low`.
- hazard size/placement heuristics are explicit:
- debris with area ratio `< 0.01` -> `small_debris`.
- near-lane-edge and area ratio `< 0.003` -> `edge_small_object`.
- “near edge” means within `15%` of the left or right lane boundary.
- high-attention mode assumptions are hardcoded:
- default countdown `18` frames.
- bonus frames: `+10` for high risk, `+6` for medium, `+3` for low.
- lane dilation in high-attention mode is `14` pixels.
- tracker history increases from `30` to `60`.
- ONNX Runtime inference prefers providers in this order:
- `CUDAExecutionProvider`
- `CPUExecutionProvider`
- TensorRT engine-building assumptions are hardcoded:
- workspace memory pool limit is `1 << 30` bytes.
- runtime input shape is fixed to `[16, 3, 640, 640]`.
- engine filenames are auto-derived and stored under `assets/compressed_models`.
- the MQTT setup assumes:
- broker `edgejet3vpn.edi.lv`
- port `8884`
- topic `reid-vehicle-detection`
- QoS `1`
- keepalive `60`
- JPEG quality `75`
- sender TLS certificates are expected under `assets/mqtt_credentials/...`.
- `CREATE_SUBSCRIBER = False` by default.
- the web UI hardcodes three model choices:
- TensorRT engine: `assets/compressed_models/mixed_dataset_trained_yolov8s_mixed_batch_trt_fp16_noint8.engine`
- YOLOv8 ONNX: `assets/compressed_models/mixed_dataset_trained_yolov8s.onnx`
- YOLOv8 PT: `assets/compressed_models/yolov8s.pt`
- `camera_config.py` contains hardcoded camera usernames, passwords, hosts, and ports.
- `stream_y8_onnx.py` contains a hardcoded ONNX model path `obs_system/compressed/yolov8s_original.onnx`, independent of the passed model name/path.
- the README explicitly states that a modified Ultralytics `loaders.py` may need to be copied into the environment for the intended crop/zoom behavior.

## Output artifacts produced by the pipeline
- `runs/det/...`: Ultralytics-style detection outputs.
- `assets/background_check/`: optional subtractor debug images.
- `assets/hazard_events/frames`: saved hazard frames.
- `assets/hazard_events/crops`: saved hazard-object crops.
- `assets/hazard_events/hazard_events.csv`: hazard event table.
- `assets/mqtt/saved_publishes.cbor`: archived outbound MQTT payloads.
- `assets/mqtt/received_vehicle_crops`: decoded inbound MQTT JPEG crops.
- `assets/perf_logs/*.csv` and `*.jsonl`: performance logs/timelines.
