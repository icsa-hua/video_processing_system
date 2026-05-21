from obs_system.communication_module.interface import mqtt_interface
from obs_system.utils.common import _get_gt, _empty_dets_numpy, _empty_results
from obs_system.compressed.interface.compressed_yolo import CompressedYOLO
from obs_system.compressed.interface.tensor_yolo import TensorRTYOLO
from obs_system.detection_module.interface.detection_batch import DetectionBatch, FrameDetections
from obs_system.detection_module.interface.streamer import Streamer
from obs_system.utils.benchmarking.metrics.model_performance import ModelPerf
from obs_system.utils.benchmarking.metrics.pc_performance import PerfLogger, SlidingCounter, CPUMonitor, GPUMonitor, FramePerfLogger, TimelineLogger
from obs_system.utils.global_config import BATCH_SIZE
from obs_system.utils.appraisal import StepContext, frame_list
from obs_system.utils.global_config import *
from obs_system.utils.common import *
from obs_system.utils.tiles import * 

import os 
import cv2 
import time
import glob
import torch 
import numpy as np
import queue
import threading
import collections 

from typing import Union, List, Any, Generator, Optional
from pathlib import Path 
from memory_profiler import profile as mem_profile
from torch.profiler import profile
from abc import abstractmethod
from torch.profiler import  ProfilerActivity
from ultralytics.data.augment import LetterBox
from ultralytics.utils.torch_utils import smart_inference_mode

from ultralytics.utils import ops


class OptimizedStreamer(Streamer): 


    def __init__(self, cfg: str, overrides:dict, _callbacks:Any)->None: 
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self._benchmark_label_files: List[Path] = []
        self._benchmark_gt_by_stem: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._benchmark_labels_loaded = False
        

    def __call__(self, source:str, model:str, logic_module=None, mqtt_broker=None, producer_flag=None, preview_queue=None, *args, **kwargs)->None:
        self.mqtt_interface = mqtt_broker 
        self.args.stream_buffer = True 
        self.logic_module = logic_module 
        self.lanes_final = None

        try: 
            self.predict_cli(source=os.path.normpath(os.path.abspath(source)) if os.path.isfile(source) else source, 
                model=model, 
                producer_flag=producer_flag, 
                preview_queue=preview_queue
            )

        except KeyboardInterrupt as ke: 

            if producer_flag is not None: 
                producer_flag.value=False 
            
            try:    
                cv2.destroyAllWindows() 
            except cv2.error: 
                pass 
            Streamer.logger.exception(f"KeyboardInterrupt: {ke}")
        
        return 


    def pre_transform(self, im:List[np.ndarray])->List: 

        pt = None 

        if isinstance(self.model, CompressedYOLO) or isinstance(self.model, TensorRTYOLO): 
            pt = True 
            self.stride = 16 if self.args.half else 32 

        else: 
            raise ValueError("The type of the model parsed is incorrect")

        same_shapes = len({x.shape for x in im}) == 1 
        letterbox = LetterBox(self.imgsz, auto=same_shapes ^ pt, stride=self.stride)
        return [letterbox(image=x) for x in im]


    def preprocess(self, im: Union[torch.Tensor, List[np.ndarray]])-> torch.Tensor | List[np.ndarray]:
        return super().preprocess(im)

    
    def postprocess(self, preds:Any, orig_image:Any)->Any : 
        return super().postprocess(preds, orig_image=orig_image) 

    def _get_batch_frame_ids(self, labels: List[str], frame_count: int) -> List[int]:
        labels = list(labels or [])
        if len(labels) < frame_count:
            labels.extend([""] * (frame_count - len(labels)))
        else:
            labels = labels[:frame_count]

        fallback_start = getattr(self, "_next_generated_frame_id", 0)
        frame_ids = get_frame_ids(labels=labels, fallback_start=fallback_start)
        if frame_ids:
            self._next_generated_frame_id = max(fallback_start + len(frame_ids), max(frame_ids) + 1)
        return frame_ids


    def _map_boxes_to_original_frame(
        self,
        boxes: torch.Tensor,
        model_input_shape: tuple[int, int],
        original_shape: tuple[int, int],
        inference_shape: tuple[int, int],
    ) -> torch.Tensor:
        if boxes is None or boxes.numel() == 0:
            return boxes

        if self.use_roi:
            return self.logic_module["ROI"].translate_bounding_boxes(
                results=boxes,
                orig_img_shape=original_shape,
                input_img_shape=model_input_shape,
                crop_img_shape=inference_shape,
            )

        return ops.scale_boxes(model_input_shape, boxes.clone(), original_shape)


    def _publish_no_motion_preview(self, original_images, preview_queue, producer_flag) -> None:
        if not self.args.show or not original_images:
            return
        self._enqueue_async_sink(
            ("preview_frame", original_images[-1].copy(), preview_queue, producer_flag),
            stage="preview_encode_ms",
            drop_if_full=True,
        )


    def _sync_subtractor_warmup_state(self) -> bool:
        subtractor = None if self.logic_module is None else self.logic_module.get("SUBTRACTOR")
        if subtractor is None or not hasattr(subtractor, "is_ready_for_inference"):
            self.done_warmup = True
            return self.done_warmup

        self.done_warmup = bool(subtractor.is_ready_for_inference())
        return self.done_warmup

    def _stage_a_acquire_and_gate(self, batch_payload, frame_read_ms, stream_start, timeline_logger, batch_idx):
        paths, im0s, s = batch_payload
        frame_ids = self._get_batch_frame_ids(labels=s, frame_count=len(im0s))
        self._reset_stage_metrics(frame_ids)
        self._record_stage_time("frame_read_ms", frame_read_ms, frame_ids=frame_ids)
        self._note_frame_ids(frame_ids)
        original_images_bgr = [im.copy() for im in im0s]
        original_images = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in original_images_bgr]
        cropped_original_images = original_images

        roi_ms = 0.0
        mog2_ms = 0.0
        defish_ms = 0.0
        _t0 = 0.0
        _t0_rel = 0.0
        res_h, res_w = im0s[0].shape[:2] if len(im0s) else (0, 0)

        if self.use_roi:
            with StepContext(name="ROI Cropping", catch=(RuntimeError,), verbose=self.args.verbose):
                if self.args.plot_performance:
                    _t0 = time.perf_counter()
                    _t0_rel = _t0 - stream_start
                im0s = self.logic_module["ROI"].crop_image(im0s)
                cropped_original_images = self.logic_module["ROI"].crop_image(original_images)
                if self.args.plot_performance:
                    roi_ms = (time.perf_counter() - _t0) * 1e3
                    self._record_stage_time("roi_ms", roi_ms, frame_ids=frame_ids)
                    _t1_rel = time.perf_counter() - stream_start
                    timeline_logger.log_span(batch_idx, "roi", _t0_rel, _t1_rel)
                    res_h, res_w = im0s[0].shape[:2] if len(im0s) else (0, 0)

        with StepContext(name="BackGround Subtractor  (Motion-Gating)", catch=(RuntimeError, Exception), verbose=self.args.verbose):
            if self.args.plot_performance:
                _t0 = time.perf_counter()
                _t0_rel = _t0 - stream_start
            mfgs, lanes_final = self.logic_module["SUBTRACTOR"].detect(im0s, save_img=True)
            if self.args.plot_performance:
                mog2_ms = (time.perf_counter() - _t0) * 1e3
                self._record_stage_time("mog2_ms", mog2_ms, frame_ids=frame_ids)
                _t1_rel = time.perf_counter() - stream_start
                timeline_logger.log_span(batch_idx, "mog2", _t0_rel, _t1_rel)
            if lanes_final is not None:
                self.lanes_final = lanes_final

        warmup_pending = not self.done_warmup
        if warmup_pending:
            if self._sync_subtractor_warmup_state():
                Streamer.logger.info("Background subtractor warmup completed. Detection pipeline enabled for the next batch.")
            self.step_attention_state()
            return {"skip_reason": "warmup"}

        if self.should_force_inference_all_frames():
            mfgs = [True] * len(mfgs)

        with StepContext(name="FishEyE Processing (Defish)", catch=(RuntimeError,), verbose=self.args.verbose):
            if self.logic_module["FEP"] is not None:
                Streamer.logger.debug("FEP enabled")
                t_defish0 = time.perf_counter()
                im0s = self.logic_module["FEP"]._defish(im0s)
                defish_ms = (time.perf_counter() - t_defish0) * 1e3
                self._record_stage_time("defish_ms", defish_ms, frame_ids=frame_ids)

        return {
            "paths": paths,
            "im0s": im0s,
            "frame_ids": frame_ids,
            "frame_read_ms": frame_read_ms,
            "original_images": original_images,
            "original_images_bgr": original_images_bgr,
            "cropped_original_images": cropped_original_images,
            "mfgs": mfgs,
            "roi_ms": roi_ms,
            "mog2_ms": mog2_ms,
            "defish_ms": defish_ms,
            "res_h": res_h,
            "res_w": res_w,
            "skip_reason": "no_motion" if not any(mfgs) else None,
        }

    def _stage_b_inference_and_nms(self, stage_a, model, profilers, activities, stream_start, timeline_logger, batch_idx):
        im0s = stage_a["im0s"]
        original_images = stage_a["original_images"]
        original_images_bgr = stage_a["original_images_bgr"]
        cropped_original_images = stage_a["cropped_original_images"]
        mfgs = stage_a["mfgs"]
        frame_ids = stage_a["frame_ids"]

        for i, keep_frame in enumerate(mfgs):
            if not keep_frame:
                im0s[i] = empty_image(im0s[i])
                original_images[i] = empty_image(original_images[i])
                original_images_bgr[i] = empty_image(original_images_bgr[i])

        if self.args.plot_performance:
            _t0 = time.perf_counter()
            _t0_rel = _t0 - stream_start

        with profilers[0]:
            images = self.preprocess(im0s)

        preprocess_ms = 0.0
        if self.args.plot_performance:
            preprocess_ms = (time.perf_counter() - _t0) * 1e3
            self._record_stage_time("preprocess_ms", preprocess_ms, frame_ids=frame_ids)
            timeline_logger.log_span(batch_idx, "preprocess", _t0_rel, time.perf_counter() - stream_start)

        if self.args.plot_performance:
            _t0 = time.perf_counter()
            _t0_rel = _t0 - stream_start

        with profilers[1]:
            event = None
            if self.seen == 0 and self.args.verbose:
                with profile(activities=activities) as prof:
                    infer_outputs = self.model(
                        images,
                        orig_imgs=original_images if not self.use_roi else cropped_original_images,
                        debug=self.args.verbose,
                    )
                if not os.path.exists("assets/trace_jsons"):
                    os.mkdir("assets/trace_jsons")
                model_tag = getattr(self, "model_tag", type(self.model).__name__)
                prof.export_chrome_trace(f"assets/trace_jsons/trace_{model_tag}.json")
            else:
                infer_outputs = self.model(
                    images,
                    orig_imgs=original_images if not self.use_roi else cropped_original_images,
                    debug=self.args.verbose,
                )
            if isinstance(infer_outputs, tuple) and len(infer_outputs) == 2 and isinstance(infer_outputs[0], tuple):
                (i_boxes, i_scores, i_classes), event = infer_outputs
            else:
                i_boxes, i_scores, i_classes = infer_outputs

        inference_ms = 0.0
        if self.args.plot_performance:
            inference_ms = (time.perf_counter() - _t0) * 1e3
            self._record_stage_time("inference_ms", inference_ms, frame_ids=frame_ids)
            timeline_logger.log_span(batch_idx, "inference", _t0_rel, time.perf_counter() - stream_start)

        if event is not None and torch.cuda.is_available():
            torch.cuda.current_stream().wait_event(event)

        frames: List[FrameDetections] = []
        nms_ms = 0.0
        for bni, fid in enumerate(frame_ids):
            orig_img = original_images[bni]
            boxes = i_boxes[bni]
            scores = i_scores[bni]
            cls_ = i_classes[bni]

            if not mfgs[bni] or boxes is None or len(boxes) == 0:
                frames.append(FrameDetections.empty(frame_id=fid, batch_index=bni, orig_img=orig_img))
                continue

            boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes)
            scores_t = scores if torch.is_tensor(scores) else torch.as_tensor(scores)
            classes_t = cls_ if torch.is_tensor(cls_) else torch.as_tensor(cls_)
            scores_t = scores_t.to(dtype=torch.float32)
            classes_t = classes_t.to(dtype=torch.int64)
            boxes_t = boxes_t.to(dtype=torch.float32)

            t_nms0 = time.perf_counter()
            keep = batched_nms(boxes_t, scores_t, classes_t.long(), iou_threshold=NMS_IOU)
            boxes_t, scores_t, classes_t = boxes_t[keep], scores_t[keep], classes_t[keep]
            nms_elapsed_ms = (time.perf_counter() - t_nms0) * 1e3
            nms_ms += nms_elapsed_ms
            self._record_stage_time("nms_ms", nms_elapsed_ms, frame_id=fid)

            if self.use_roi:
                boxes_t = self.logic_module["ROI"].translate_bounding_boxes(
                    results=boxes_t,
                    orig_img_shape=self.original_imgsz,
                )

            if boxes_t.numel() == 0:
                frames.append(FrameDetections.empty(frame_id=fid, batch_index=bni, orig_img=orig_img))
                continue

            frames.append(
                FrameDetections(
                    frame_id=fid,
                    batch_index=bni,
                    orig_img=orig_img,
                    boxes=boxes_t,
                    scores=scores_t,
                    classes=classes_t,
                )
            )

        stage_a["im0s"] = im0s
        return {
            "detections": DetectionBatch(frames=frames),
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "nms_ms": nms_ms,
        }

    def _stage_c_tracking_and_hazard_logic(self, detection_batch: DetectionBatch, orig_images_bgr, profilers):
        tracked_results = []
        frame_bundles = []
        for frame in detection_batch:
            frame_bundles.append(
                {
                    "frame_id": frame.frame_id,
                    "orig_img": frame.orig_img,
                    "bni": frame.batch_index,
                    "empty": frame.is_empty,
                }
            )
            if self.tracker_model is not None and not frame.is_empty:
                tracked = self.tracker_model.detect_compact(
                    frame=frame,
                    class_names=self.converter.class_names,
                )
                tracked_results.append(tracked)
            else:
                tracked_results.append(frame.to_results(self.converter.class_names))

        with profilers[2]:
            postprocessed = self.postprocess_batch(tracked_results, orig_images=orig_images_bgr)
        return frame_bundles, postprocessed

    def _stage_d_dispatch_optional_sinks(
        self,
        preds,
        frame_bundles,
        paths,
        original_images_bgr,
        preview_queue,
        producer_flag,
    ):
        frame_log_rows = []
        mqtt_preds = []
        mqtt_images = []
        mqtt_frame_ids = []

        for fb, result in zip(frame_bundles, preds):
            fid = fb["frame_id"]
            bni = fb["bni"]
            filename = Path(paths[bni])

            if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                self._enqueue_async_sink(
                    ("result_outputs", result, filename, int(getattr(self.dataset, "count", fid)), preview_queue, producer_flag, getattr(result, "track_points", None)),
                    stage="preview_encode_ms" if self.args.show else None,
                    frame_id=fid,
                    drop_if_full=bool(self.args.show and not (self.args.save or self.args.save_txt or self.args.save_crop)),
                )

            hazards = getattr(result, "hazard_events", None) or []
            if hazards:
                self._enqueue_async_sink(
                    ("hazard_event", result, original_images_bgr[bni], list(hazards), fid),
                    stage="event_saving_ms",
                    frame_id=fid,
                )

            if self.mqtt_interface is not None and result.boxes is not None and result.boxes.xyxy.numel() > 0:
                mqtt_preds.append(result)
                mqtt_images.append(original_images_bgr[bni])
                mqtt_frame_ids.append(fid)

            frame_log_rows.append(
                {
                    "frame_id": fid,
                    "bni": bni,
                    "motion_passed": int(not fb["empty"]),
                }
            )

        if mqtt_preds:
            self._enqueue_async_sink(
                ("mqtt_results", mqtt_preds, mqtt_images, mqtt_frame_ids),
                stage="mqtt_ms",
                frame_ids=mqtt_frame_ids,
            )

        return frame_log_rows


    def _ensure_benchmark_labels_loaded(self) -> None:
        if self._benchmark_labels_loaded or not self.args.bench:
            return

        label_dir = str(getattr(self.args, "bench_labels", "") or "")
        label_files = [Path(filename) for filename in sorted(glob.glob(f"{label_dir}/*.txt"), key=key_func)]
        self._benchmark_label_files = label_files
        self._benchmark_gt_by_stem = build_gt_index(
            [str(path) for path in label_files],
            fixed_size=(FIXED_WIDTH, FIXED_HEIGHT),
        )
        self._benchmark_labels_loaded = True


    def _resolve_benchmark_gt(self, frame_id: Any) -> tuple[np.ndarray, np.ndarray]:
        if not self.args.bench:
            return (np.zeros((0,), np.float32), np.zeros((0, 4), np.float32))

        self._ensure_benchmark_labels_loaded()

        frame_key = str(frame_id)
        gt = _get_gt(frame_key, self._benchmark_gt_by_stem)
        if gt[1].size:
            return gt

        if isinstance(frame_id, (int, np.integer)):
            index = int(frame_id)
            for candidate in (index, index - 1):
                if 0 <= candidate < len(self._benchmark_label_files):
                    stem = self._benchmark_label_files[candidate].stem
                    return _get_gt(stem, self._benchmark_gt_by_stem)

        return (np.zeros((0,), np.float32), np.zeros((0, 4), np.float32))


    def setup_model(self, model_name:str, path_to_load:Optional[str|Path], opt:str)->None:
        pass 


    @smart_inference_mode()
    def stream_inference(self, source:str, model:str, producer_flag:Any, preview_queue:Any, *args, **kwargs)->Generator[Optional[Any], None, None]:

        self.source = source 

        if self.args.verbose : Streamer.logger.info(" ")

        with self._lock: 
            with StepContext(name="Set up Dataloader Process", catch=(RuntimeError, ), verbose=self.args.verbose):
                self.setup_source(source if source is not None else self.args.source)
                self.dataset.bs = BATCH_SIZE

            self.seen = 0  
            self.batch = None 
            self.mp = None 
            self._next_generated_frame_id = 0
                       
            self.use_roi = bool(
                self.args.roi
                and self.logic_module is not None
                and self.logic_module.get("ROI") is not None
            )
            # self.use_roi = self.args.roi

            profilers = (
                ops.Profile(device=self.device), 
                ops.Profile(device=self.device),
                ops.Profile(device=self.device)
            )

            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            start_time = time.perf_counter() 

            first_batch = next(iter(self.dataset)) #Keeps the original pointer without moving it "peeking" to the first frame 
            _, im0s, _ = first_batch
            self._note_first_frame_ready()

            self.original_imgsz = im0s[0].shape[:2] 
            self.orig_height, self.orig_width = self.original_imgsz
            subtractor = None if self.logic_module is None else self.logic_module.get("SUBTRACTOR")
            if self.use_roi :
                with StepContext(name="ROI Cropping", catch=(RuntimeError, ), verbose=self.args.verbose): 
                    
                    self.logic_module['ROI'].set_regions(im0s[0]) 
                    self.cropped_imgsz = ((self.logic_module['ROI'].y_end - self.logic_module['ROI'].y_start),(self.logic_module['ROI'].x_end - self.logic_module['ROI'].x_start) )
                    cropped_frame = self.logic_module['ROI'].crop_image(im0s[0])
                    
                    if self.args.show: 
                        self.logic_module['ROI']._show_regions(cropped_frame.copy())
                    
                    self.orig_height, self.orig_width = cropped_frame.shape[:2]

            if subtractor is not None and hasattr(subtractor, "configure_source_warmup"):
                subtractor.configure_source_warmup(source_is_stream=bool(getattr(self.source_type, "stream", False)))
                self.done_warmup = subtractor.is_ready_for_inference()
            else:
                self.done_warmup = True

            if subtractor is not None and hasattr(subtractor, "configure_runtime_recalibration"):
                subtractor.configure_runtime_recalibration(
                    enabled=bool(getattr(self.source_type, "stream", False)),
                    interval_frames=getattr(self.args, "lane_recalibration_interval_frames", None),
                )

            if not getattr(self.source_type, "stream", False):
                empty_image = f"{EMPTY_IMAGE_PATH}"
                if not os.path.exists(empty_image):
                    raise FileNotFoundError(f"Empty image path for background subtraction does not exist: {empty_image}")

                empty_image = cv2.imread(empty_image)
                self.logic_module['SUBTRACTOR'].warm_up(empty_image, trials=TRIALS)
                self._sync_subtractor_warmup_state()
            
            tile_flag = True if (self.orig_width // TILE_SIZE) > TILE_THR or (self.orig_height // TILE_SIZE) >= TILE_THR else False
            force_no_tiles = bool(getattr(self, "force_streaming_no_tiles", False))
            if tile_flag and not force_no_tiles:
                Streamer.logger.info("Run Inference with Tiles")
                return self._stream_inference_impl_tiles(
                    model=model,
                    producer_flag=producer_flag,
                    preview_queue=preview_queue,
                    profilers=profilers,
                    activities=activities,
                    start_time=start_time,
                )

            Streamer.logger.info("Run Inference without Tiles")
            return self._stream_inference_impl(
                model=model,
                producer_flag=producer_flag,
                preview_queue=preview_queue,
                profilers=profilers,
                activities=activities,
                start_time=start_time,
            )


    @abstractmethod
    @mem_profile
    def _stream_inference_impl(self, **kwargs): 

        FPS_WINDOW = 100  # sliding window size
        fps_times = collections.deque(maxlen=FPS_WINDOW)
        fps=0
        stream_start = time.perf_counter()
        last_fps_log = stream_start
        total_frames = 0

        # Performance logging (for plots: FPS vs motion density, inference calls/sec, latency breakdown)
        perf_log_path = getattr(self.args, 'perf_log', None) or 'assets/perf_logs/perf_log.csv'
        perf_flush_every = int(getattr(self.args, 'perf_log_flush_every', 64) or 64)
        frame_flush_every = int(getattr(self.args, 'perf_frame_log_flush_every', 128) or 128)
        timeline_flush_every = int(getattr(self.args, 'perf_timeline_flush_every', 128) or 128)
        perf_logger = PerfLogger(perf_log_path, flush_every=perf_flush_every)
        frame_log_path = getattr(self.args, 'perf_log_frames', None) or 'assets/perf_logs/perf_frames.csv'
        frame_logger = FramePerfLogger(frame_log_path, flush_every=frame_flush_every)

        timeline_path = getattr(self.args, 'perf_timeline', None) or 'assets/perf_logs/perf_timeline.jsonl'
        timeline_logger = TimelineLogger(timeline_path, flush_every=timeline_flush_every)

        gpu_index = int(getattr(self.args, 'gpu_index', 0))
        gpu_mon = GPUMonitor(gpu_index=gpu_index)
        cpu_mon = CPUMonitor()
        infer_counter = SlidingCounter(window_s=1.0)
        batch_idx = 0

        model = kwargs["model"] 
        producer_flag  = kwargs["producer_flag"] 
        preview_queue = kwargs["preview_queue"]
        profilers=kwargs["profilers"] 
        activities=kwargs["activities"]
        start_time = kwargs["start_time"]

        if self.args.bench: 
            self.mp = ModelPerf(
                class_ids=[i for i, _ in enumerate(self.converter.class_names)], 
                iou_thresholds=np.arange(0.50, 0.96, 0.05), 
                conf_threshold=CONF_THR, 
                use_101_point_interp=True
            )

        else: 
            self.mp = None 
    
        self.run_callbacks("on_predict_start") 
        with StepContext(name="Warmup Session", catch=(Exception, RuntimeError), verbose=self.args.verbose):
            if not self.model_warmup_done: 
                self.model.warmup(micro=BATCH_SIZE, warmup_sessions=WARM_UP_SESSIONS)
                self.model_warmup_done = True

        # Asynchronous batch loading to avoid stalls
        batch_queue = queue.Queue(maxsize=8)

        def producer():
            try:
                dataset_iter = iter(self.dataset)
                while True:
                    t_read0 = time.perf_counter()
                    try:
                        batch = next(dataset_iter)
                    except StopIteration:
                        break
                    read_ms = (time.perf_counter() - t_read0) * 1e3
                    batch_queue.put((batch, read_ms))
            except Exception as e:
                Streamer.logger.error(f"Error in batch producer: {e}")
            finally:
                batch_queue.put(None)  # sentinel

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        last_frame_id = None

        while True:

            t_batch_start = time.perf_counter()
            # sample resource utilization at batch start (best-effort)
            gpu_stats = gpu_mon.sample()
            cpu_stats = cpu_mon.sample()
            frame_read_ms = 0.0
            roi_ms = 0.0
            mog2_ms = 0.0
            defish_ms = 0.0
            preprocess_ms = 0.0
            inference_ms = 0.0
            postprocess_ms = 0.0
            nms_ms = 0.0
            tracking_ms = 0.0

            if self.runtime_limit_reached():
                break

            try:
                self.batch = batch_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.batch is None:
                break

            self.run_callbacks("on_predict_batch_start")

            batch_payload, frame_read_ms = self.batch
            self.batch = batch_payload
            stage_a = self._stage_a_acquire_and_gate(batch_payload, frame_read_ms, stream_start, timeline_logger, batch_idx)
            if stage_a.get("skip_reason") == "warmup":
                continue

            paths = stage_a["paths"]
            im0s = stage_a["im0s"]
            frame_ids = stage_a["frame_ids"]
            original_images_bgr = stage_a["original_images_bgr"]
            roi_ms = stage_a["roi_ms"]
            mog2_ms = stage_a["mog2_ms"]
            defish_ms = stage_a["defish_ms"]
            res_h = stage_a["res_h"]
            res_w = stage_a["res_w"]
            mfgs = stage_a["mfgs"]

            if stage_a.get("skip_reason") == "no_motion":
                Streamer.logger.debug("No motion detected in the batch - skipping inference")
                empty_preds = return_no_motion_frames(
                    im0s=im0s,
                    batch_size=len(im0s),
                )
                if self.mp is not None and self.args.bench:
                    for fid in frame_ids:
                        gt_cls, gt_bbs = self._resolve_benchmark_gt(fid)
                        empty_boxes, empty_scores, empty_cls = _empty_dets_numpy()
                        self.mp.update(
                            boxes_xyxy=empty_boxes,
                            scores=empty_scores,
                            classes=empty_cls,
                            gt_boxes_xyxy=gt_bbs.astype(np.float32),
                            gt_classes=gt_cls.astype(np.int64),
                        )
                self._note_emitted_result(len(empty_preds))
                yield empty_preds
                self._enqueue_async_sink(("mqtt_no_detection", empty_preds, frame_ids), stage="mqtt_ms", frame_ids=frame_ids)
                self._publish_no_motion_preview(original_images_bgr, preview_queue, producer_flag)
                if self.args.plot_performance:
                    t_now = time.perf_counter()
                    scores = getattr(self.logic_module.get('SUBTRACTOR', None), 'last_motion_scores', None)
                    avg_motion_score = float(np.mean(scores)) if scores else 0.0
                    motion_density = 0.0
                    fps_sliding = fps
                    total_ms = (t_now - t_batch_start) * 1e3
                    infer_calls_per_sec = infer_counter.rate(t_now)
                    batch_stage_metrics = self._get_batch_stage_metrics()
                    perf_logger.log({
                        't_wall': t_now,
                        'batch_idx': batch_idx,
                        'frames_in_batch': len(im0s),
                        'res_w': res_w,
                        'res_h': res_h,
                        'motion_density': motion_density,
                        'avg_motion_score': avg_motion_score,
                        'inference_ran': 0,
                        'frames_inferred': 0,
                        'infer_calls_per_sec': infer_calls_per_sec,
                        'gpu_util': gpu_stats.get('gpu_util', float('nan')),
                        'gpu_mem_used_mb': gpu_stats.get('mem_used_mb', float('nan')),
                        'gpu_mem_total_mb': gpu_stats.get('mem_total_mb', float('nan')),
                        'cpu_util': cpu_stats.get('cpu_util', float('nan')),
                        'frame_read_ms_per_frame': frame_read_ms / max(len(im0s), 1),
                        'roi_ms_per_frame': roi_ms / max(len(im0s), 1),
                        'mog2_ms_per_frame': mog2_ms / max(len(im0s), 1),
                        'defish_ms_per_frame': defish_ms / max(len(im0s), 1),
                        'preprocess_ms_per_frame': 0.0,
                        'inference_ms_per_frame': 0.0,
                        'postprocess_ms_per_frame': 0.0,
                        'nms_ms_per_frame': 0.0,
                        'tracking_ms_per_frame': 0.0,
                        'hazard_logic_ms_per_frame': 0.0,
                        'preview_encode_ms_per_frame': batch_stage_metrics.get('preview_encode_ms', 0.0) / max(len(im0s), 1),
                        'mqtt_ms_per_frame': batch_stage_metrics.get('mqtt_ms', 0.0) / max(len(im0s), 1),
                        'event_saving_ms_per_frame': 0.0,
                        'total_ms_per_frame': total_ms / max(len(im0s), 1),
                        'fps_sliding': fps_sliding,
                    })

                    scores_list = getattr(self.logic_module.get('SUBTRACTOR', None), 'last_motion_scores', None) or [0.0] * len(im0s)
                    per_roi = roi_ms / max(len(im0s), 1)
                    per_mog2 = mog2_ms / max(len(im0s), 1)
                    per_read = frame_read_ms / max(len(im0s), 1)
                    per_defish = defish_ms / max(len(im0s), 1)
                    for i, fid in enumerate(frame_ids):
                        frame_stage_metrics = self._get_frame_stage_metrics(fid)
                        frame_logger.log({
                            't_wall': t_now,
                            'batch_idx': batch_idx,
                            'frame_id': int(fid) if str(fid).isdigit() else fid,
                            'res_w': res_w,
                            'res_h': res_h,
                            'motion_passed': 0,
                            'motion_score': float(scores_list[i]) if i < len(scores_list) else 0.0,
                            'gpu_util': gpu_stats.get('gpu_util', float('nan')),
                            'gpu_mem_used_mb': gpu_stats.get('mem_used_mb', float('nan')),
                            'cpu_util': cpu_stats.get('cpu_util', float('nan')),
                            'frame_read_ms': per_read,
                            'roi_ms': per_roi,
                            'mog2_ms': per_mog2,
                            'defish_ms': per_defish,
                            'preprocess_ms': 0.0,
                            'inference_ms': 0.0,
                            'postprocess_ms': 0.0,
                            'nms_ms': 0.0,
                            'tracking_ms': 0.0,
                            'hazard_logic_ms': 0.0,
                            'preview_encode_ms': frame_stage_metrics.get('preview_encode_ms', 0.0),
                            'mqtt_ms': frame_stage_metrics.get('mqtt_ms', 0.0),
                            'event_saving_ms': 0.0,
                            'total_ms': (
                                per_read
                                + per_roi
                                + per_mog2
                                + per_defish
                                + frame_stage_metrics.get('preview_encode_ms', 0.0)
                                + frame_stage_metrics.get('mqtt_ms', 0.0)
                            ),
                        })

                    timeline_logger.log_span(batch_idx, 'batch_total', t_batch_start - stream_start, t_now - stream_start, {
                        'inference_ran': 0,
                        'frames_in_batch': int(len(im0s)),
                    })
                    batch_idx += 1
                self.step_attention_state()
                continue

            stage_b = self._stage_b_inference_and_nms(stage_a, model, profilers, activities, stream_start, timeline_logger, batch_idx)
            preprocess_ms = stage_b["preprocess_ms"]
            inference_ms = stage_b["inference_ms"]
            nms_ms = stage_b["nms_ms"]
            detection_batch = stage_b["detections"]
            postprocess_total_t0 = time.perf_counter()
            if self.args.plot_performance:
                _tpost = time.perf_counter()
                _tpost_rel = _tpost - stream_start
            frame_bundles, preds = self._stage_c_tracking_and_hazard_logic(detection_batch, original_images_bgr, profilers)
            frame_log_rows = self._stage_d_dispatch_optional_sinks(
                preds,
                frame_bundles,
                paths,
                original_images_bgr,
                preview_queue,
                producer_flag,
            )

            for fb, r in zip(frame_bundles, preds):
                r.speed = {
                    "preprocess": profilers[0].dt * 1e3/len(im0s),
                    "inference": profilers[1].dt * 1e3/len(im0s),
                    "postprocess": profilers[2].dt * 1e3/len(im0s)
                }
                self.log_detection_snapshot(r)
                fid = fb["frame_id"]
                bni = fb["bni"]

                if self.mp is not None and self.args.bench:
                    gt_cls, gt_bbs = self._resolve_benchmark_gt(fid)
                    if r.boxes is None or r.boxes.xyxy.numel() == 0:
                        det_boxes, det_scores, det_classes = _empty_dets_numpy()
                    else:
                        detections = getattr(r, "sv_detections", None)
                        if detections is not None:
                            det_boxes = np.asarray(detections.xyxy, dtype=np.float32)
                            det_scores = np.asarray(detections.confidence, dtype=np.float32)
                            det_classes = np.asarray(detections.class_id, dtype=np.int64)
                        else:
                            det_boxes = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                            det_scores = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
                            det_classes = r.boxes.cls.detach().cpu().numpy().astype(np.int64)
                    self.mp.update(
                        boxes_xyxy=det_boxes,
                        scores=det_scores,
                        classes=det_classes,
                        gt_boxes_xyxy=gt_bbs.astype(np.float32),
                        gt_classes=gt_cls.astype(np.int64),
                    )
                 
                # if fid != last_frame_id:
                #     continue

                self._note_emitted_result()
                yield r

                if (self.args.only_FPS and not self.args.plot_performance) or self.args.plot_performance:
                    now = time.perf_counter()
                    fps_times.append(now)
                    total_frames += 1
                    if len(fps_times) > 1:
                        fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                    if now - last_fps_log >= 1.0:
                        elapsed = now - stream_start
                        avg_fps = total_frames / elapsed
                        Streamer.logger.info(
                            f"[FPS] End-to-end FPS: {fps:.2f} | "
                            f"Average FPS since start: {avg_fps:.2f}"
                        )
                        last_fps_log = now

            self.run_callbacks("on_predict_postprocess_end")
            
            if self.args.plot_performance:

                # ---- perf log (batch processed) ----
                t_now = time.perf_counter()
                frames_in_batch = len(im0s)
                frames_inferred = int(sum(bool(x) for x in mfgs)) if 'mfgs' in locals() else 0
                motion_density = (frames_inferred / frames_in_batch) if frames_in_batch else 0.0
                scores = getattr(self.logic_module.get('SUBTRACTOR', None), 'last_motion_scores', None)
                avg_motion_score = float(np.mean(scores)) if scores else 0.0
                inference_ran = 1 if frames_inferred > 0 else 0
                if inference_ran:
                    infer_counter.add(t_now, 1.0)  # one model call per batch
                infer_calls_per_sec = infer_counter.rate(t_now)
                total_ms = (t_now - t_batch_start) * 1e3
                batch_stage_metrics = self._get_batch_stage_metrics()
                postprocess_ms = max(
                    0.0,
                    ((t_now - postprocess_total_t0) * 1e3)
                    - batch_stage_metrics.get('preview_encode_ms', 0.0)
                    - batch_stage_metrics.get('mqtt_ms', 0.0),
                )
                timeline_logger.log_span(batch_idx, 'postprocess', _tpost_rel, t_now - stream_start, {'frame_in_batch': int(bni)})
                per_read = frame_read_ms / max(frames_in_batch, 1)
                per_roi = roi_ms / max(frames_in_batch, 1)
                per_mog2 = mog2_ms / max(frames_in_batch, 1)
                per_defish = defish_ms / max(frames_in_batch, 1)
                per_pre = preprocess_ms / max(frames_in_batch, 1)
                per_inf = inference_ms / max(frames_in_batch, 1)
                perf_logger.log({
                    't_wall': t_now,
                    'batch_idx': batch_idx,
                    'frames_in_batch': frames_in_batch,
                    'res_w': res_w,
                    'res_h': res_h,
                    'motion_density': motion_density,
                    'avg_motion_score': avg_motion_score,
                    'inference_ran': inference_ran,
                    'frames_inferred': frames_inferred,
                    'infer_calls_per_sec': infer_calls_per_sec,
                    'gpu_util': gpu_stats.get('gpu_util', float('nan')),
                    'gpu_mem_used_mb': gpu_stats.get('mem_used_mb', float('nan')),
                    'gpu_mem_total_mb': gpu_stats.get('mem_total_mb', float('nan')),
                    'cpu_util': cpu_stats.get('cpu_util', float('nan')),
                    'frame_read_ms_per_frame': per_read,
                    'roi_ms_per_frame': per_roi,
                    'mog2_ms_per_frame': per_mog2,
                    'defish_ms_per_frame': per_defish,
                    'preprocess_ms_per_frame': per_pre,
                    'inference_ms_per_frame': per_inf,
                    'postprocess_ms_per_frame': postprocess_ms / max(frames_in_batch, 1),
                    'nms_ms_per_frame': batch_stage_metrics.get('nms_ms', 0.0) / max(frames_in_batch, 1),
                    'tracking_ms_per_frame': batch_stage_metrics.get('tracking_ms', 0.0) / max(frames_in_batch, 1),
                    'hazard_logic_ms_per_frame': batch_stage_metrics.get('hazard_logic_ms', 0.0) / max(frames_in_batch, 1),
                    'preview_encode_ms_per_frame': batch_stage_metrics.get('preview_encode_ms', 0.0) / max(frames_in_batch, 1),
                    'mqtt_ms_per_frame': batch_stage_metrics.get('mqtt_ms', 0.0) / max(frames_in_batch, 1),
                    'event_saving_ms_per_frame': batch_stage_metrics.get('event_saving_ms', 0.0) / max(frames_in_batch, 1),
                    'total_ms_per_frame': total_ms / max(frames_in_batch, 1),
                    'fps_sliding': fps,
                })
                scores_list = getattr(self.logic_module.get('SUBTRACTOR', None), 'last_motion_scores', None) or []
                for row in frame_log_rows:
                    fid = row["frame_id"]
                    bni = row["bni"]
                    frame_stage_metrics = self._get_frame_stage_metrics(fid)
                    motion_score = float(scores_list[bni]) if bni < len(scores_list) else float('nan')
                    frame_logger.log({
                        't_wall': t_now,
                        'batch_idx': batch_idx,
                        'frame_id': int(fid) if str(fid).isdigit() else fid,
                        'res_w': res_w,
                        'res_h': res_h,
                        'motion_passed': row["motion_passed"],
                        'motion_score': motion_score,
                        'gpu_util': gpu_stats.get('gpu_util', float('nan')),
                        'gpu_mem_used_mb': gpu_stats.get('mem_used_mb', float('nan')),
                        'cpu_util': cpu_stats.get('cpu_util', float('nan')),
                        'frame_read_ms': per_read,
                        'roi_ms': per_roi,
                        'mog2_ms': per_mog2,
                        'defish_ms': per_defish,
                        'preprocess_ms': per_pre,
                        'inference_ms': per_inf,
                        'postprocess_ms': postprocess_ms / max(frames_in_batch, 1),
                        'nms_ms': frame_stage_metrics.get('nms_ms', 0.0),
                        'tracking_ms': frame_stage_metrics.get('tracking_ms', 0.0),
                        'hazard_logic_ms': frame_stage_metrics.get('hazard_logic_ms', 0.0),
                        'preview_encode_ms': frame_stage_metrics.get('preview_encode_ms', 0.0),
                        'mqtt_ms': frame_stage_metrics.get('mqtt_ms', 0.0),
                        'event_saving_ms': frame_stage_metrics.get('event_saving_ms', 0.0),
                        'total_ms': (
                            per_read
                            + per_roi
                            + per_mog2
                            + per_defish
                            + per_pre
                            + per_inf
                            + (postprocess_ms / max(frames_in_batch, 1))
                            + frame_stage_metrics.get('preview_encode_ms', 0.0)
                            + frame_stage_metrics.get('mqtt_ms', 0.0)
                        ),
                    })
                timeline_logger.log_span(batch_idx, 'batch_total', t_batch_start - stream_start, t_now - stream_start, {
                    'inference_ran': int(inference_ran),
                    'frames_in_batch': int(frames_in_batch),
                })
                batch_idx += 1
                # ----------------------------------
           
            self.run_callbacks("on_predict_batch_end")
            self.step_attention_state()

        if self.stop_reason != "stream_limit":
            producer_thread.join()

        if self.args.bench and self.mp is not None: 
            self.mp.finalize() 
            Streamer.logger.info(self.mp.results())

        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
        

        if self.args.verbose and self.seen: 
            t = tuple(x.t / self.seen * 1e3 for x in profilers) 
            Streamer.logger.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, BATCH_SIZE)}" % t
            )

        if (self.args.only_FPS and not self.args.plot_performance) or (self.args.plot_performance): 

            # ---------------- CLEANUP LOG ----------------
            total_time = time.perf_counter() - stream_start
            if total_frames > 0:
                print(
                    f"[FPS] FINAL Average FPS: {total_frames / total_time:.2f}"
                )

        if self.args.plot_performance:
            # Close performance loggers
            try:
                perf_logger.close()
                frame_logger.close()
                timeline_logger.close()
            except Exception:
                pass

        self.release_session_resources(preview_queue=preview_queue, producer_flag=producer_flag)
        self.run_callbacks("on_predict_end")


    @abstractmethod 
    @mem_profile
    def _stream_inference_impl_tiles(self, **kwargs)->Generator[Optional[Any], None, None]: 
        pass


    def _frames_to_tiles(self, frame_iter, tile_size:int, overlap_ratio:float): 
        overlap_px = max(0, int(round(tile_size * overlap_ratio)))
        for f_id, img in frame_iter: 
            self.frame_images[f_id] = img
            for t, m in split_image_gen(img, f_id, tile_size=tile_size, overlap=overlap_px):
                yield t, m


    def iter_data(self): 

        """ Yields (frame_id, image) lazily from self.dataset after ROI + MOG2 gating """
        
        self.dataset = iter(self.dataset) 
        while True: 
            if self.runtime_limit_reached():
                return

            try: 
                self.batch = next(self.dataset)
            except StopIteration: 
                return 

            _, im0s, s = self.batch 
            frame_ids = self._get_batch_frame_ids(labels=s, frame_count=len(im0s))
            self._note_frame_ids(frame_ids)
            original_images = im0s.copy() 
            if self.use_roi:
                with StepContext(name="ROI Cropping", catch=(RuntimeError, ), verbose=self.args.verbose): 
                    im0s = self.logic_module['ROI'].crop_image(im0s)

            with StepContext(name="FishEyE Processing (Defish)", catch=(RuntimeError, ), verbose=self.args.verbose):    
                # Defish FishEye camera frames to increase accuracy
                if self.logic_module["FEP"] is not None: 
                    Streamer.logger.debug("FEP enabled")
                    im0s = self.logic_module["FEP"]._defish(im0s)

            with StepContext(name="BackGround Subtractor (Motion-Gating)", catch=(RuntimeError, Exception), verbose=self.args.verbose):
                # Motion gate (vectorized over the mini batch) 
                mfgs, lanes_final = self.logic_module["SUBTRACTOR"].detect(im0s, save_img=False)
                if lanes_final is not None: 
                    self.lanes_final = lanes_final

            warmup_pending = not self.done_warmup
            if warmup_pending:
                if self._sync_subtractor_warmup_state():
                    Streamer.logger.info("Background subtractor warmup completed. Detection pipeline enabled for the next batch.")
                self.step_attention_state()
                continue

            if self.should_force_inference_all_frames():
                mfgs = [True] * len(mfgs)

            if not any(mfgs):
                print("No motion detected in the batch - skipping inference")
                empty_preds = return_no_motion_frames(
                    im0s=im0s,
                    batch_size=len(im0s),
                )
                self._enqueue_async_sink(("mqtt_no_detection", empty_preds, frame_ids), stage="mqtt_ms", frame_ids=frame_ids)
                self.step_attention_state()
                continue 

            for i, keep_frame in enumerate(mfgs):
                if not keep_frame:
                    im0s[i] = empty_image(im0s[i])
                    original_images[i] = empty_image(original_images[i])

            if self.args.bench and self.mp is not None: 
                self._ensure_benchmark_labels_loaded()

                for passed, f_id, img in zip(mfgs, frame_ids, im0s): 
                    gt_cls, gt_bbs = self._resolve_benchmark_gt(f_id)

                    if passed : 
                        yield (int(f_id), img)

                    else: 
                        empty_boxes, empty_scores, empty_cls = _empty_dets_numpy() 
                        self.mp.update(
                            boxes_xyxy=empty_boxes,
                            scores=empty_scores,
                            classes=empty_cls,
                            gt_boxes_xyxy=gt_bbs.astype(np.float32),
                            gt_classes=gt_cls.astype(np.int64),
                        )                
                        continue

            else: 
                for passed, f_id, img in zip(mfgs, frame_ids, im0s): 
                    if passed: 
                        yield (int(f_id), img)

            self.step_attention_state()



    def _publish_mqtt_message(self, preds, mqtt_messages, frame_ids)->None: 
        super()._publish_mqtt_message(preds, mqtt_messages, frame_ids)


    def _publish_mqtt_message_no_detection(self, preds, frame_index)->None: 
        super()._publish_mqtt_message_no_detection(preds, frame_index)
