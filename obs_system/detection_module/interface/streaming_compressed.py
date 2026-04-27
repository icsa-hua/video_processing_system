from obs_system.communication_module.interface import mqtt_interface
from obs_system.utils.common import _get_gt, _empty_dets_numpy, _empty_results
from obs_system.compressed.interface.compressed_yolo import CompressedYOLO
from obs_system.compressed.interface.tensor_yolo import TensorRTYOLO
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

            cv2.destroyAllWindows() 
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

        preview = original_images[-1].copy()
        if self.use_roi and self.logic_module is not None and self.logic_module.get("ROI") is not None:
            self.logic_module["ROI"]._show_regions(preview)

        self.proc_image = self._encode_preview_frame(cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        self.publish_preview(preview_queue, producer_flag)


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

            self.original_imgsz = im0s[0].shape[:2] 
            self.orig_height, self.orig_width = self.original_imgsz

            empty_image = f"{EMPTY_IMAGE_PATH}"
            if not os.path.exists(empty_image):
                raise FileNotFoundError(f"Empty image path for background subtraction does not exist: {empty_image}")

            empty_image = cv2.imread(empty_image)
            if self.use_roi :
                with StepContext(name="ROI Cropping", catch=(RuntimeError, ), verbose=self.args.verbose): 
                    
                    self.logic_module['ROI'].set_regions(im0s[0]) 
                    self.cropped_imgsz = ((self.logic_module['ROI'].y_end - self.logic_module['ROI'].y_start),(self.logic_module['ROI'].x_end - self.logic_module['ROI'].x_start) )
                    cropped_frame = self.logic_module['ROI'].crop_image(im0s[0])
                    
                    if self.args.show: 
                        self.logic_module['ROI']._show_regions(cropped_frame.copy())
                    
                    self.orig_height, self.orig_width = cropped_frame.shape[:2]
                self.logic_module['SUBTRACTOR'].warm_up(empty_image, trials=TRIALS)
            
            else: 
                self.logic_module['SUBTRACTOR'].warm_up(empty_image, trials=TRIALS)
            
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
        perf_logger = PerfLogger(perf_log_path)
        frame_log_path = getattr(self.args, 'perf_log_frames', None) or 'assets/perf_logs/perf_frames.csv'
        frame_logger = FramePerfLogger(frame_log_path)

        timeline_path = getattr(self.args, 'perf_timeline', None) or 'assets/perf_logs/perf_timeline.jsonl'
        timeline_logger = TimelineLogger(timeline_path)

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
            if not self.done_warmup: 
                self.model.warmup(micro=BATCH_SIZE, warmup_sessions=WARM_UP_SESSIONS)
                self.done_warmup = True

        # Asynchronous batch loading to avoid stalls
        batch_queue = queue.Queue(maxsize=8)

        def producer():
            try:
                for batch in self.dataset:
                    batch_queue.put(batch)
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
            roi_ms = 0.0
            mog2_ms = 0.0
            preprocess_ms = 0.0
            inference_ms = 0.0
            postprocess_ms = 0.0

            if self.runtime_limit_reached():
                break

            try:
                self.batch = batch_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.batch is None:
                break

            self.run_callbacks("on_predict_batch_start")

            paths, im0s, s = self.batch
            frame_ids = self._get_batch_frame_ids(labels=s, frame_count=len(im0s))
            original_images = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in im0s.copy()]
            
            _t0, _t0_rel = 0.0,0.0

            res_h, res_w = im0s[0].shape[:2] if len(im0s) else (0, 0)

            #use_roi = True if self.args.roi and self.logic_module["ROI"] is not None else False 
            if self.use_roi :
                with StepContext(name="ROI Cropping", catch=(RuntimeError, ), verbose=self.args.verbose): 
                    if self.args.plot_performance:
                        _t0 = time.perf_counter()
                        _t0_rel = _t0 - stream_start
                    
                    im0s = self.logic_module['ROI'].crop_image(im0s)
                    
                    if self.args.plot_performance:
                        roi_ms = (time.perf_counter() - _t0) * 1e3
                        _t1_rel = time.perf_counter() - stream_start
                        timeline_logger.log_span(batch_idx, 'roi', _t0_rel, _t1_rel)
                        res_h, res_w = im0s[0].shape[:2] if len(im0s) else (0, 0)
                
            # Required here to capture the cropped frames, if cropping happens
            if self.use_roi:
                cropped_original_images = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in im0s.copy()]
                   
            with StepContext(name="BackGround Subtractor  (Motion-Gating)", catch=(RuntimeError, Exception), verbose=self.args.verbose):
                # Motion gate (vectorized over the mini batch) 
                if self.args.plot_performance:

                    _t0 = time.perf_counter()
                    _t0_rel = _t0 - stream_start
               
                mfgs, lanes_final = self.logic_module["SUBTRACTOR"].detect(im0s, save_img=True)
                
                if self.args.plot_performance:
                    mog2_ms = (time.perf_counter() - _t0) * 1e3
                    _t1_rel = time.perf_counter() - stream_start
                    timeline_logger.log_span(batch_idx, 'mog2', _t0_rel, _t1_rel)
                if lanes_final is not None: 
                    self.lanes_final = lanes_final

            if self.should_force_inference_all_frames():
                mfgs = [True] * len(mfgs)

            with StepContext(name="FishEyE Processing (Defish)", catch=(RuntimeError, ), verbose=self.args.verbose):    
                # Defish FishEye camera frames to increase accuracy
                if self.logic_module["FEP"] is not None: 
                    Streamer.logger.debug("FEP enabled")
                    im0s = self.logic_module["FEP"]._defish(im0s)  

                    
            # For rectilinear images, motion gating seems to only work with ROI.  
            # Speeds up the process when no motion is detected in the incoming batch. 
            if not any(mfgs):
                Streamer.logger.debug("No motion detected in the batch - skipping inference")
                empty_preds = return_no_motion_frames(
                    im0s=im0s,
                    batch_size=len(im0s),
                )
                yield empty_preds 
                if self.args.plot_performance:

                    # ---- perf log (batch skipped) ----
                    t_now = time.perf_counter()
                    scores = getattr(self.logic_module.get('SUBTRACTOR', None), 'last_motion_scores', None)
                    avg_motion_score = float(np.mean(scores)) if scores else 0.0
                    motion_density = 0.0
                    fps_sliding = fps
                    total_ms = (t_now - t_batch_start) * 1e3
                    infer_calls_per_sec = infer_counter.rate(t_now)
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
                        'roi_ms_per_frame': roi_ms / max(len(im0s), 1),
                        'mog2_ms_per_frame': mog2_ms / max(len(im0s), 1),
                        'preprocess_ms_per_frame': 0.0,
                        'inference_ms_per_frame': 0.0,
                        'postprocess_ms_per_frame': 0.0,
                        'total_ms_per_frame': total_ms / max(len(im0s), 1),
                        'fps_sliding': fps_sliding,
                    })
                
                    # per-frame latency log (skipped inference)
                    scores_list = getattr(self.logic_module.get('SUBTRACTOR', None), 'last_motion_scores', None) or [0.0]*len(im0s)
                    per_roi = roi_ms / max(len(im0s), 1)
                    per_mog2 = mog2_ms / max(len(im0s), 1)
                    per_total = total_ms / max(len(im0s), 1)
                    for i, fid in enumerate(frame_ids):
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
                            'roi_ms': per_roi,
                            'mog2_ms': per_mog2,
                            'preprocess_ms': 0.0,
                            'inference_ms': 0.0,
                            'postprocess_ms': 0.0,
                            'total_ms': per_total,
                        })

                    # timeline span for skipped batch
                    timeline_logger.log_span(batch_idx, 'batch_total', t_batch_start - stream_start, t_now - stream_start, {
                        'inference_ran': 0,
                        'frames_in_batch': int(len(im0s)),
                    })
                    batch_idx += 1

                self._publish_mqtt_message_no_detection(preds=empty_preds, frame_index=frame_ids)
                self._publish_no_motion_preview(original_images, preview_queue, producer_flag)
                self.step_attention_state()
                continue # to the next batch 
                
            for i, keep_frame in enumerate(mfgs):
                if not keep_frame:
                    im0s[i] = empty_image(im0s[i])
                    original_images[i] = empty_image(original_images[i])

            if self.args.plot_performance:
                _t0 = time.perf_counter()
                _t0_rel = _t0 - stream_start

            with profilers[0]: 
                # if cropped the im0s here have a (572, 1290, 3) shape. Otherwise same shape with original images
                images = self.preprocess(im0s) 
                model_input_shape = tuple(int(v) for v in images.shape[-2:])
                            
            if self.args.plot_performance:
                preprocess_ms = (time.perf_counter() - _t0) * 1e3 
                timeline_logger.log_span(batch_idx, 'preprocess', _t0_rel, time.perf_counter() - stream_start)

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
                    # The shape of image is (3, 640, 640) | original_images shape is (1080, 1920, 3)
                    # If ROI is enabled then original_images shape should be the cropped size.
                    infer_outputs = self.model(
                        images,
                        orig_imgs=original_images if not self.use_roi else cropped_original_images,
                        debug=self.args.verbose,
                    )

                if isinstance(infer_outputs, tuple) and len(infer_outputs) == 2 and isinstance(infer_outputs[0], tuple):
                    (i_boxes, i_scores, i_classes), event = infer_outputs
                else:
                    i_boxes, i_scores, i_classes = infer_outputs
            
            if self.args.plot_performance:
                inference_ms = (time.perf_counter() - _t0) * 1e3
                timeline_logger.log_span(batch_idx, 'inference', _t0_rel, time.perf_counter() - stream_start)

            if event is not None and torch.cuda.is_available():
                # Synchronize only when the backend provides a CUDA completion event.
                torch.cuda.current_stream().wait_event(event)

            if self.args.plot_performance:
                _tpost = time.perf_counter()
                _tpost_rel = _tpost - stream_start

            frame_bundles = [] # one entry per frame in batch 
            for bni in range(len(original_images)): 

                fid = frame_ids[bni]
                orig_img = original_images[bni]

                # If self.use_roi then boxes here have the coordinates of the cropped images. 
                boxes = i_boxes[bni] 
                scores = i_scores[bni]
                cls_ = i_classes[bni]

                if not mfgs[bni]: 
                    frame_bundles.append({
                        "frame_id": fid, 
                        "orig_img": orig_img, 
                        "empty": True, 
                        "boxes": None, 
                        "scores": None, 
                        "classes": None,
                        "bni":bni
                    })
                    continue


                # preds = None 
                # self.seen = bni
                # current_frame_id = frame_ids[self.seen]
                # No result returned from the object detection model

                if boxes is None or len(boxes) == 0: 
                    frame_bundles.append({
                        "frame_id": fid, 
                        "orig_img": orig_img, 
                        "empty": True, 
                        "boxes": None, 
                        "scores": None, 
                        "classes": None,
                        "bni": bni
                    })
                    continue

                boxes_t = torch.as_tensor(boxes) 
                scores_t = torch.as_tensor(scores) 
                classes_t = torch.as_tensor(cls_).long()

                keep = batched_nms(
                        boxes_t, 
                        scores_t, 
                        classes_t.long(), 
                        iou_threshold=(1.0-NMS_IOU)
                    )
                    
                # keep = keep_pc if keep_pc.numel()==0 else keep_pc[nms(boxes_t[keep_pc],scores_t[keep_pc], iou_threshold=(1-NMS_IOU))]
                boxes_t, scores_t, classes_t = boxes_t[keep], scores_t[keep], classes_t[keep] 
               
                inference_shape = (
                    cropped_original_images[bni].shape[:2]
                    if self.use_roi
                    else orig_img.shape[:2]
                )
                boxes_t = self._map_boxes_to_original_frame(
                    boxes=boxes_t,
                    model_input_shape=model_input_shape,
                    original_shape=orig_img.shape[:2],
                    inference_shape=inference_shape,
                )

                frame_bundles.append({
                    "frame_id": fid, 
                    "orig_img": orig_img, 
                    "empty": False, 
                    "boxes": boxes_t, 
                    "scores": scores_t, 
                    "classes": classes_t,
                    "bni": bni
                })

            results_list = []  # Results only for non-empty frames
            results_map = {} # frame_id -> Results 

            for fb in frame_bundles: 

                if fb["empty"]: 
                    r = _empty_results(orig_image=fb["orig_img"]) 
                else: 

                    inf_results = torch.cat(
                            [fb["boxes"],
                             fb["scores"][:,None],
                             fb["classes"][:,None].float()],
                            dim=1
                    )
                    
                    # len(r) = number of detections. r.boxes.xyxy.shape is (4,4) e.g. 
                    # len(frame_bundles) = Number of frames. 
                    r = Results(
                        orig_img=fb["orig_img"],
                        path=f"image_{fb['frame_id']}.jpg",
                        names=self.converter.class_names,
                        boxes=inf_results, 
                        speed={}
                    )

                results_list.append(r) 
                results_map[fb["frame_id"]] = r

                # ================= FPS MEASUREMENT =================
                if (self.args.only_FPS and not self.args.plot_performance) or (self.args.plot_performance): 
                    now = time.perf_counter()
                    fps_times.append(now)
                    total_frames += 1
                
                    # Sliding-window FPS (end-to-end)
                    if len(fps_times) > 1:
                        fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])

                    # Log FPS once per second
                    if now - last_fps_log >= 1.0:
                        elapsed = now - stream_start
                        avg_fps = total_frames / elapsed
                        Streamer.logger.info(
                            f"[FPS] End-to-end FPS: {fps:.2f} | "
                            f"Average FPS since start: {avg_fps:.2f}"
                        )
                        last_fps_log = now
                    # ===================================================

                if self.tracker_model is not None:
                    if fb["empty"] : continue 
                    
                    fid = fb["frame_id"] 
                    results_map[fid] = self.tracker_model.detect(
                        predictions=results_map[fid], 
                        save=False, 
                        orig_frame=fb["orig_img"], 
                        f_id=fid, 
                        class_names=self.converter.class_names
                    )
                        
                    # with StepContext(name="Tracking Frame", catch=(RuntimeError, ), verbose=self.args.verbose):
                    #     preds = self.tracker_model.detect(predictions=preds,save=False,orig_frame=orig_img,f_id=frame_ids[self.seen],class_names=self.converter.class_names)

            with profilers[2]:
                out = self.postprocess_batch(results_list, orig_images=original_images)

                # if self.mp is not None and self.args.bench: 
                #     gt_cls, gt_bbs = self.__gt_labels.pop(self.seen, (np.zeros((0,), np.int64),np.zeros((0,4), np.float32)))
                #     self.mp.update(
                #         boxes_xyxy = boxes_t.cpu().numpy().astype(np.float32), 
                #         scores = scores_t.cpu().numpy().astype(np.float32), 
                #         classes = classes_t.cpu().numpy().astype(np.int32), 
                #         gt_boxes_xyxy=gt_bbs.astype(np.float32), 
                #         gt_classes = gt_cls.astype(np.int64)
                #     ) 

            preds = [] 
            mqtt_messages = [] 
            for fb, (r,mqtt_mess) in zip(frame_bundles,out):
                r.speed = {
                    "preprocess": profilers[0].dt * 1e3/len(im0s),
                    "inference": profilers[1].dt * 1e3/len(im0s),
                    "postprocess": profilers[2].dt * 1e3/len(im0s)
                }
                fid = fb["frame_id"]
                bni = fb["bni"]
                 
                # if fid != last_frame_id:
                #     continue

                yield r 

                if self.args.plot_performance:

                    # --- per-frame logging (approximate stage attribution) ---
                    t_now_f = time.perf_counter()
                    scores_list = getattr(self.logic_module.get('SUBTRACTOR', None), 'last_motion_scores', None) or []
                    motion_score = float(scores_list[bni]) if bni < len(scores_list) else float('nan')
                    per_roi = roi_ms / max(len(im0s), 1)
                    per_mog2 = mog2_ms / max(len(im0s), 1)
                    per_pre = preprocess_ms / max(len(im0s), 1)
                    per_inf = inference_ms / max(len(im0s), 1)
                    # postprocess_ms is measured per-frame
                    total_est = per_roi + per_mog2 + per_pre + per_inf + postprocess_ms
                    frame_logger.log({
                        't_wall': t_now_f,
                        'batch_idx': batch_idx,
                        'frame_id': int(fid) if str(fid).isdigit() else fid,
                        'res_w': res_w,
                        'res_h': res_h,
                        'motion_passed': int(bool(mfgs[bni])) if bni < len(mfgs) else 1,
                        'motion_score': motion_score,
                        'gpu_util': gpu_stats.get('gpu_util', float('nan')),
                        'gpu_mem_used_mb': gpu_stats.get('mem_used_mb', float('nan')),
                        'cpu_util': cpu_stats.get('cpu_util', float('nan')),
                        'roi_ms': per_roi,
                        'mog2_ms': per_mog2,
                        'preprocess_ms': per_pre,
                        'inference_ms': per_inf,
                        'postprocess_ms': postprocess_ms,
                        'total_ms': total_est,
                    })
                    # --------------------------------------------------------

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    filename=Path(paths[bni])

                    if self.args.save or self.args.show:
                        self.optional_save_or_show(r, filename)

                    self.save_queue.put(("save_results", r, filename, fid))

                    if self.args.save_txt or self.args.save_crop:
                        with StepContext(name="Save Results", catch=(Exception, RuntimeError), verbose=self.args.verbose):
                            self.batch[2][bni] += self.write_results(
                                preds=r, 
                                i = bni,
                                im= original_images,
                        )

                self.publish_preview(preview_queue, producer_flag)

                preds.append(r)
                mqtt_messages.append(mqtt_mess)
                    # if self.seen == len(im0s)-1 and self.args.verbose: 
                    #     elapsed_time=time.perf_counter() - start_time 
                    #     Streamer.logger.info(f"Time from capturing batch to meaningful information: {elapsed_time:.2f}")
            self._publish_mqtt_message(preds=preds, mqtt_messages=mqtt_messages, frame_ids=frame_ids) 
            last_frame_id = bni 
        
            if self.args.plot_performance:
                postprocess_ms = (time.perf_counter() - _tpost) *1e3
                timeline_logger.log_span(batch_idx, 'postprocess', _tpost_rel, time.perf_counter() - stream_start, {'frame_in_batch': int(bni)})

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
                    'roi_ms_per_frame': roi_ms / max(frames_in_batch, 1),
                    'mog2_ms_per_frame': mog2_ms / max(frames_in_batch, 1),
                    'preprocess_ms_per_frame': preprocess_ms / max(frames_in_batch, 1),
                    'inference_ms_per_frame': inference_ms / max(frames_in_batch, 1),
                    'postprocess_ms_per_frame': postprocess_ms / max(frames_in_batch, 1),
                    'total_ms_per_frame': total_ms / max(frames_in_batch, 1),
                    'fps_sliding': fps,
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

            if self.should_force_inference_all_frames():
                mfgs = [True] * len(mfgs)

            if not any(mfgs):
                print("No motion detected in the batch - skipping inference")
                empty_preds = return_no_motion_frames(
                    im0s=im0s,
                    batch_size=len(im0s),
                )
                yield empty_preds 

                self._publish_mqtt_message_no_detection(preds=empty_preds, frame_index=frame_ids)
                self.step_attention_state()
                continue 

            for i, keep_frame in enumerate(mfgs):
                if not keep_frame:
                    im0s[i] = empty_image(im0s[i])
                    original_images[i] = empty_image(original_images[i])

            if self.args.bench and self.mp is not None: 
 
                labels = [] 
                for filename in sorted(glob.glob(f'{self.args.bench_labels}/*.txt'), key=key_func):
                    labels.append(filename)

                self.__gt_labels = build_gt_index(labels, fixed_size=(FIXED_WIDTH, FIXED_HEIGHT))

                for passed, f_id, img, gt_file in zip(mfgs, frame_ids, im0s, labels): 
                    
                    stem = Path(gt_file).stem
                    gt_cls, gt_bbs = _get_gt(stem, self.__gt_labels)

                    if passed : 
                        self.__gt_labels[int(f_id)] = (gt_cls, gt_bbs)
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
