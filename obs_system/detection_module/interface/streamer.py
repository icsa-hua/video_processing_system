from obs_system.compressed.interface.convert_to_Results import ConverterResults 
from obs_system.logic_module.dummy_logic.obstacle_filtering import analyze_lane_hazards
from obs_system.utils.logger import get_logger
from obs_system.utils.common import *
#
import os
import cv2
import torch
import logging
import platform
import threading 
import numpy as np 
import time
import queue 
import csv

from pathlib import Path
from io import StringIO
from collections import Counter, defaultdict
from typing import Union, List, Any, Optional, final, Generator, Tuple, Dict
from abc import ABC, abstractmethod
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils.plotting import colors
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils import DEFAULT_CFG, callbacks
from ultralytics.data.build import load_inference_source
from ultralytics.utils.torch_utils import smart_inference_mode
from datetime import datetime, timezone


class Streamer(ABC): 
    
    logger = get_logger("obs_system"+__name__)

    STREAM_WARNING = """
    WARNING ⚠️ Infefrence results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
    errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

    Example:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
    """

    @abstractmethod
    def __init__(self, cfg:str, overrides:dict, _callbacks:Any)->None:

        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        self.docker_flag = False 
        self.done_warmup = True
        self.model_warmup_done = False
        
        self.save_queue = queue.Queue(maxsize=10)
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        self.seen = 0 
        self.speed = {} 
        self.windows = [] 
        self.vid_writer = {} 
        self.frame_images = {} 
        
        self.__check_docker_env() 

        self.mp: Any = None 
        self.model: Any = None 
        self.stride: Any = None 
        self.imgsz: Any = None 
        self.cropped_imgsz:Any = None
        self.original_imgsz:Any = None 
        self.use_roi:bool = False
        self.device: Any = None 
        self.dataset: Any = None
        self.plotted_img: Any = None 
        self.lanes_final: Any = None
        self.batch: Any = None 
        self.source_type: Any = None 
        self.results: Optional[List[Any]] = []
        # self.txt_path: Optional[str] = None 
        self.proc_image: Optional[bytes] = None 
        self.mqtt_interface:Any = None 
        self.logic_module: Any = None 
        self.tracker_model: Any = None 
        self.points = dict() 
        self.preview_last_emit_ts = 0.0
        self.current_hazards: List[Dict[str, Any]] = []
        self.last_scene_masks: Dict[str, Any] = {"lane_mask": None, "crosswalk_mask": None}
        self.high_attention_countdown = 0
        self.high_attention_default_frames = 18
        self.normal_lane_expand_px = 0
        self.high_attention_lane_expand_px = 14
        self.normal_tracker_history = 30
        self.high_attention_tracker_history = 60
        self._perf_batch_stage_ms: Dict[str, float] = {}
        self._perf_frame_stage_ms: Dict[str, Dict[str, float]] = {}
        self._perf_active_frame_id: Optional[str] = None
        self.stream_limit_hours = 0.0
        self.stream_limit_seconds = 0.0
        self.stream_limit_deadline: Optional[float] = None
        self.stream_limit_active = False
        self.stop_reason: Optional[str] = None
        self._save_worker_stopped = False
        self._session_resources_released = False
        self._source_setup_started_at: Optional[float] = None
        self._prev_numeric_frame_id: Optional[int] = None
        self.run_metrics: Dict[str, Any] = {
            "stream_open_seconds": None,
            "first_frame_seconds": None,
            "frames_observed": 0,
            "frames_emitted": 0,
            "dropped_frames": 0,
            "mqtt_batches": 0,
            "mqtt_no_detection_batches": 0,
            "mqtt_hazard_messages": 0,
            "saved_hazard_events": 0,
            "saved_hazard_crops": 0,
            "crop_count": 0,
            "crop_total_bytes": 0,
            "crop_total_pixels": 0,
        }

        self._lock = threading.Lock() 
        self.converter = ConverterResults() 
        self.callbacks = _callbacks or callbacks.get_default_callbacks() 
        self.cropped_image_dirname = f'cropped_trial_{np.random.randint(44)}'
        self._init_hazard_store()
        
        callbacks.add_integration_callbacks(self) 

    def _reset_stage_metrics(self, frame_ids: Optional[List[Any]] = None) -> None:
        self._perf_batch_stage_ms = {}
        self._perf_frame_stage_ms = {}
        self._perf_active_frame_id = None
        for frame_id in frame_ids or []:
            self._perf_frame_stage_ms[str(frame_id)] = {}

    def _record_stage_time(
        self,
        stage: str,
        elapsed_ms: float,
        *,
        frame_id: Any = None,
        frame_ids: Optional[List[Any]] = None,
    ) -> None:
        ms = max(0.0, float(elapsed_ms))
        self._perf_batch_stage_ms[stage] = self._perf_batch_stage_ms.get(stage, 0.0) + ms

        targets: List[str] = []
        if frame_id is not None:
            targets = [str(frame_id)]
        elif frame_ids:
            targets = [str(fid) for fid in frame_ids]

        if not targets:
            return

        share = ms / max(len(targets), 1)
        for target in targets:
            frame_metrics = self._perf_frame_stage_ms.setdefault(target, {})
            frame_metrics[stage] = frame_metrics.get(stage, 0.0) + share

    def _get_batch_stage_metrics(self) -> Dict[str, float]:
        return dict(self._perf_batch_stage_ms)

    def _get_frame_stage_metrics(self, frame_id: Any) -> Dict[str, float]:
        return dict(self._perf_frame_stage_ms.get(str(frame_id), {}))

    def _set_perf_active_frame(self, frame_id: Any = None) -> None:
        self._perf_active_frame_id = None if frame_id is None else str(frame_id)

    def _note_source_setup_started(self) -> None:
        self._source_setup_started_at = time.perf_counter()
        self.run_metrics["stream_open_seconds"] = None
        self.run_metrics["first_frame_seconds"] = None

    def _note_source_setup_finished(self) -> None:
        if self._source_setup_started_at is None:
            return
        self.run_metrics["stream_open_seconds"] = time.perf_counter() - self._source_setup_started_at

    def _note_first_frame_ready(self) -> None:
        if self._source_setup_started_at is None:
            return
        if self.run_metrics["first_frame_seconds"] is None:
            self.run_metrics["first_frame_seconds"] = time.perf_counter() - self._source_setup_started_at

    def _note_frame_ids(self, frame_ids: List[Any]) -> None:
        self.run_metrics["frames_observed"] += len(frame_ids)

        for frame_id in frame_ids:
            if not isinstance(frame_id, (int, np.integer)):
                continue

            current = int(frame_id)
            previous = self._prev_numeric_frame_id
            if previous is not None and current > previous + 1:
                self.run_metrics["dropped_frames"] += current - previous - 1
            self._prev_numeric_frame_id = current

    def _note_emitted_result(self, count: int = 1) -> None:
        self.run_metrics["frames_emitted"] += max(0, int(count))
        

    def __check_docker_env(self): 
        if self.args.show:
            if getattr(self.args, "gui", False):
                # Browser preview does not require a local display backend.
                self.args.show = True
                if os.path.exists("/.dockerenv") or os.getenv("container") == "docker":
                    self.docker_flag = True
                return

            if os.path.exists("/.dockerenv") or os.getenv("container") == "docker":
                self.docker_flag = True 
                log_stream = StringIO() 
                log_handler = logging.StreamHandler(log_stream)
                logger_ultra = logging.getLogger("ultralytics")
                logger_ultra.addHandler(log_handler)
                self.args.show = check_imshow(warn=True)
                log_handler.flush()
                log_contents = log_stream.getvalue()
                logger_ultra.removeHandler(log_handler)
                    
                if "WARNING ⚠️" in log_contents:
                    self.args.show = True # Probably we are on a docker, where with streamlit we can show the images. 
            else: 
                self.args.show = True
                #self.args.show = check_imshow(warn=True)

    def _init_hazard_store(self) -> None:
        self.hazard_root = Path("assets") / "hazard_events"
        self.hazard_frames_dir = self.hazard_root / "frames"
        self.hazard_crops_dir = self.hazard_root / "crops"
        _ensure_dir(self.hazard_frames_dir)
        _ensure_dir(self.hazard_crops_dir)
        self.hazard_csv_path = self.hazard_root / "hazard_events.csv"
        if not self.hazard_csv_path.exists():
            with self.hazard_csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp_utc",
                        "frame_id",
                        "class_name",
                        "category",
                        "risk",
                        "kind",
                        "action",
                        "bbox_x1",
                        "bbox_y1",
                        "bbox_x2",
                        "bbox_y2",
                        "lane_overlap",
                        "crosswalk_overlap",
                        "event_image",
                        "crop_image",
                    ]
                )

    def _resolve_scene_masks(self, target_hw: Optional[Tuple[int, int]] = None) -> Dict[str, Optional[np.ndarray]]:
        subtractor = None
        if self.logic_module is not None:
            subtractor = self.logic_module.get("SUBTRACTOR")

        if subtractor is None or not hasattr(subtractor, "get_scene_masks"):
            self.last_scene_masks = {"lane_mask": self.lanes_final, "crosswalk_mask": None}
            return self.last_scene_masks

        expand_px = self.high_attention_lane_expand_px if self.high_attention_countdown > 0 else self.normal_lane_expand_px
        scene_masks = subtractor.get_scene_masks(expand_px=expand_px)
        if self.lanes_final is not None and (scene_masks.get("lane_mask") is None):
            scene_masks["lane_mask"] = self.lanes_final

        if self.use_roi and self.logic_module is not None and self.logic_module.get("ROI") is not None:
            roi = self.logic_module["ROI"]
            lane = scene_masks.get("lane_mask")
            lane_shape = None if lane is None else lane.shape[:2]
            target_shape = tuple(target_hw) if target_hw is not None else None
            should_embed_to_full = (
                target_shape is not None
                and self.original_imgsz is not None
                and target_shape == tuple(self.original_imgsz)
                and lane_shape != target_shape
            )

            if should_embed_to_full:
                full_h, full_w = self.original_imgsz

                def _embed(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
                    if mask is None:
                        return None
                    out = np.zeros((full_h, full_w), dtype=np.uint8)
                    y0, y1 = int(roi.y_start), int(roi.y_end)
                    x0, x1 = int(roi.x_start), int(roi.x_end)
                    h = max(0, min(y1 - y0, mask.shape[0]))
                    w = max(0, min(x1 - x0, mask.shape[1]))
                    if h > 0 and w > 0:
                        out[y0 : y0 + h, x0 : x0 + w] = mask[:h, :w]
                    return out

                scene_masks["lane_mask"] = _embed(scene_masks.get("lane_mask"))
                scene_masks["crosswalk_mask"] = _embed(scene_masks.get("crosswalk_mask"))

        self.last_scene_masks = scene_masks
        return scene_masks

    def _activate_high_attention(self, hazards: List[Dict[str, Any]]) -> None:
        if not hazards:
            return
        max_bonus = 0
        for hz in hazards:
            risk = str(hz.get("risk", "")).lower()
            if risk == "high":
                max_bonus = max(max_bonus, 10)
            elif risk == "medium":
                max_bonus = max(max_bonus, 6)
            else:
                max_bonus = max(max_bonus, 3)
        self.high_attention_countdown = max(self.high_attention_countdown, self.high_attention_default_frames + max_bonus)
        if self.tracker_model is not None and hasattr(self.tracker_model, "set_history_persistence"):
            self.tracker_model.set_history_persistence(self.high_attention_tracker_history)

    def step_attention_state(self) -> None:
        if self.high_attention_countdown > 0:
            self.high_attention_countdown -= 1
            if self.high_attention_countdown == 0:
                if self.tracker_model is not None and hasattr(self.tracker_model, "set_history_persistence"):
                    self.tracker_model.set_history_persistence(self.normal_tracker_history)

    def should_force_inference_all_frames(self) -> bool:
        return self.high_attention_countdown > 0


    def from_numpy(self, x:np.ndarray)->torch.Tensor:
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x


    @abstractmethod
    def __call__(self, source:str, model:str, logic_module=None, mqtt_broker=None, producer_flag=None, preview_queue=None, *args, **kwargs)->None:
        pass


    @abstractmethod
    def pre_transform(self, im:List[np.ndarray])->list: 
        pass  


    @abstractmethod
    def preprocess(self, im: Union[torch.Tensor, List[np.ndarray]])-> torch.Tensor | List[np.ndarray]:
        pass


    @abstractmethod
    def postprocess(self, preds:Any, orig_image:Any)->Any : 
        if not isinstance(preds, Results): 
            raise ValueError("Not using ultralytics.Results class in postprocess of Streamer.") 

        preds.save_dir = self.save_dir.__str__() 
        frame_id = self._extract_frame_id(preds)

        hazards: List[Dict[str, Any]] = []
        target_hw = None if orig_image is None else orig_image.shape[:2]
        scene_masks = self._resolve_scene_masks(target_hw=target_hw)
        lane_mask = scene_masks.get("lane_mask")
        crosswalk_mask = scene_masks.get("crosswalk_mask")

        with StepContext(
            name="Hazard Logic",
            catch=(RuntimeError, Exception),
            verbose=self.args.verbose,
            on_complete=lambda _name, ms: self._record_stage_time("hazard_logic_ms", ms, frame_id=frame_id),
        ):
            if preds.boxes is not None and preds.boxes.xyxy.numel() > 0:
                hazards = analyze_lane_hazards(
                    boxes=preds.boxes.xyxy,
                    classes=preds.boxes.cls,
                    class_names=self.converter.class_names,
                    lane_mask=lane_mask,
                    crosswalk_mask=crosswalk_mask,
                )
                self._activate_high_attention(hazards)

        try:  
            self.points.clear()
        except: 
            pass
        
        if getattr(self, "tracker_model", None) is not None: 
            with StepContext(
                name="Tracking History",
                catch=(RuntimeError, Exception),
                verbose=self.args.verbose,
                on_complete=lambda _name, ms: self._record_stage_time("tracking_ms", ms, frame_id=frame_id),
            ):
                self.points = self.tracker_model.update_tracker_history(preds, logic_module=self.logic_module)


        mqtt_batch_messages = {"crops": [], "boxes_xyxy": np.zeros((0, 4), dtype=np.int32), "paths": []}
        try: 
            # TODO: Don't have only the option to save the image but instead also be able to transmit them through mqtt. 
            mqtt_batch_messages = self.capture_object_boxes(
                    image=orig_image,
                    results=preds,
                    save=self.args.save, 
                    return_crops=True
            )
        except IndexError as ie: 
            Streamer.logger.exception(ie)

        self.current_hazards = hazards
        preds.hazard_events = hazards
        preds.hazard_mode = "high_attention" if self.high_attention_countdown > 0 else "normal"

        if hazards:
            self._record_hazard_evidence(preds=preds, image=orig_image, hazards=hazards, frame_id=frame_id)

        return preds, mqtt_batch_messages

    
    def postprocess_batch(self, preds_list: List[Results], orig_images: List[Any])-> List[Results]: 
        out = [] 
        for preds, im in zip(preds_list, orig_images): 
            results, mqtts = self.postprocess(preds, im)
            out.append((results, mqtts))
        return out


    @final
    def predict_cli(self, source:str, model:str, producer_flag:Any=None, preview_queue:Any=None)->None: 
        """
        Method used for Command Line Interface (CLI) prediction.

        This function is designed to run predictions using the CLI. It sets up the source and model, then processes
        the inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
        generator without storing results.

        Note:
            Do not modify this function or remove the generator. The generator ensures that no outputs are
            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
        """
        gen = self.stream_inference(source=source, model=model, producer_flag=producer_flag, preview_queue=preview_queue)
        for _ in gen: 
            pass 


    def setup_source(self, source:str)->None: 
        """Sets up source and inference mode."""

        self._note_source_setup_started()
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.stride,min_dim=2) 

        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer
        )
        
        self.source_type = self.dataset.source_type
        self._note_source_setup_finished()
        self.configure_runtime_limit()
        if not getattr(self,"stream", True ) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000 # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ): 
            Streamer.logger.warning(Streamer.STREAM_WARNING)

        if self.args.verbose:
            Streamer.logger.debug("Dataset Source Type | {}".format(self.source_type))


    def configure_runtime_limit(self) -> None:
        raw_limit = float(getattr(self.args, "stream_limit_hours", 0.0) or 0.0)
        self.stream_limit_hours = max(0.0, raw_limit)
        self.stream_limit_seconds = self.stream_limit_hours * 3600.0
        self.stop_reason = None
        self.stream_limit_active = bool(
            self.stream_limit_seconds > 0.0
            and self.source_type is not None
            and getattr(self.source_type, "stream", False)
        )
        self.stream_limit_deadline = (
            time.monotonic() + self.stream_limit_seconds if self.stream_limit_active else None
        )

        if self.stream_limit_active:
            Streamer.logger.info(
                "Live stream runtime limit enabled: %.2f hour(s)",
                self.stream_limit_hours,
            )


    def runtime_limit_reached(self) -> bool:
        if not self.stream_limit_active or self.stream_limit_deadline is None:
            return False

        if time.monotonic() < self.stream_limit_deadline:
            return False

        self.stream_limit_active = False
        self.stop_reason = "stream_limit"
        Streamer.logger.warning(
            "Live stream runtime limit of %.2f hour(s) reached. Stopping inference and releasing resources.",
            self.stream_limit_hours,
        )
        return True


    def stop_save_worker(self) -> None:
        if self._save_worker_stopped:
            return

        self._save_worker_stopped = True
        try:
            self.save_queue.put_nowait(None)
        except queue.Full:
            try:
                self.save_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.save_queue.put_nowait(None)
            except queue.Full:
                pass

        if self.save_thread.is_alive():
            self.save_thread.join(timeout=5)


    def release_dataset_resources(self) -> None:
        if self.dataset is None:
            return

        close_fn = getattr(self.dataset, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                Streamer.logger.debug("Dataset close() raised during cleanup", exc_info=True)

        release_targets = []
        for attr_name in ("cap", "caps", "vid_cap", "vid_caps"):
            resource = getattr(self.dataset, attr_name, None)
            if resource is None:
                continue
            if isinstance(resource, (list, tuple)):
                release_targets.extend(resource)
            else:
                release_targets.append(resource)

        for resource in release_targets:
            release_fn = getattr(resource, "release", None)
            if callable(release_fn):
                try:
                    release_fn()
                except Exception:
                    Streamer.logger.debug("Dataset release() raised during cleanup", exc_info=True)


    def release_video_writers(self) -> None:
        for writer in self.vid_writer.values():
            if isinstance(writer, cv2.VideoWriter):
                writer.release()
        self.vid_writer.clear()


    def release_session_resources(self, preview_queue: Any = None, producer_flag: Any = None) -> None:
        if self._session_resources_released:
            return

        self._session_resources_released = True

        if producer_flag is not None:
            producer_flag.value = False

        self.close_preview_stream(preview_queue)
        self.stop_save_worker()
        self.release_video_writers()
        self.release_dataset_resources()
        self.proc_image = None
        self.batch = None
        self.dataset = None
        self.frame_images.clear()
        self.points.clear()

        if isinstance(self.results, list):
            self.results.clear()

        if self.args.show or self.args.save:
            try: 
                cv2.destroyAllWindows()
            except cv2.error: 
                pass 


    @abstractmethod
    def setup_model(self, model:str, opt:str)-> None: 
        pass


    @abstractmethod 
    @smart_inference_mode()
    def stream_inference(self, source:str, model:str, producer_flag:Any, preview_queue:Any, *args, **kwargs)->Generator[Optional[Any], None, None]: 
        raise NotImplemented


    def write_results(self, preds:Any, i: Any, im:Any)->str: 
        """Write inference results to a file or directory."""
        
        string = "" 

        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            _ = self.dataset.count
        else:
            _ = getattr(self.dataset, "count", i)

        # # Ensure batch dimension
        if isinstance(im, list): 
            im = np.array(im) 
            
        if len(im.shape) == 3:
            im = im[None]  

        string = f"{i}: " if (self.source_type.stream or self.source_type.from_img or self.source_type.tensor) else ""
        string += "%gx%g " % im.shape[2:] 
        string += f"{preds.verbose()}{preds.speed.get('inference', 0.0): .1f}ms"
        return string


    def _save_worker(self):
        while True:
            task = self.save_queue.get()
            if task is None:
                break
            try: 
                self._do_save(task)
            except Exception: 
                Streamer.logger.exception("Save worker task failed: %s", task[0] if task else task)


    def _do_save(self, task):

        task_type = task[0] 

        if task_type == "save_frame" : 
            _, save_path, frame, im = task 
            self._do_save_frame(save_path, frame, im) 
            return

        if task_type == "save_results": 
            _, preds, p, frame = task 
            self._do_save_results(preds, p, frame)
            return 

        if task_type == "save_hazard_event":
            _, frame_path, annotated, crop_specs, csv_rows = task
            self._do_save_hazard_event(frame_path, annotated, crop_specs, csv_rows)
            return

        # if task_type == "save_crops": 
        #     _, crops_payload = task
        #     self._do_save_crops(crops_payload)
        #     return

        Streamer.logger.warning(f"Uknown save task type: {task_type}")


    def _do_save_frame(self, save_path, frame, im): 

        if im is None: 
            return 

        out_path = Path(save_path).expanduser()
        if out_path.name == "":
            Streamer.logger.error(f"Save predicted images: empty save path {save_path}")
            return

        ensure_dir(out_path.parent)

        if im.ndim == 3 and im.shape[2] == 3:
            bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        else:
            bgr = im

        is_stream_or_video = getattr(self.dataset, "mode", None) in {"stream", "video"}

        if is_stream_or_video:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            h, w = bgr.shape[:2]

            if h <= 0 or w <= 0:
                Streamer.logger.error("Invalid frame size")
                return

            vid_key = str(out_path.resolve())
            vw = self.vid_writer.get(vid_key)

            if vw is None:
                vw, opened_path, fourcc_used = open_writer(out_path, fps=fps, size_hw=(h, w))

                if vw is None:
                    Streamer.logger.error("VideoWriter failed to open for %s (fps=%s, size=%sx%s). "
                             "Check codec support in your OpenCV build.",
                             out_path, fps, w, h)
                    return

                self.vid_writer[vid_key] = vw

                Streamer.logger.info("Opened VideoWriter: %s (fourcc=%s)", opened_path, fourcc_used)

                if self.args.save_frames and opened_path is not None:
                    frames_dir = opened_path.with_suffix("").parent / (opened_path.stem + "_frames")
                    ensure_dir(frames_dir)
                    self._frames_dir_cache = getattr(self, "_frames_dir_cache", {})
                    self._frames_dir_cache[vid_key] = frames_dir

            self.vid_writer[vid_key].write(bgr)

            if getattr(self.args, "save_frames", False):
                frames_dir = getattr(self, "_frames_dir_cache", {}).get(vid_key)

                if frames_dir is None:
                    frames_dir = out_path.with_suffix("").parent / (out_path.stem + "_frames")
                    ensure_dir(frames_dir)
                    self._frames_dir_cache[vid_key] = frames_dir
                img_path = frames_dir / f"{int(frame):06d}.jpg"

                ok = cv2.imwrite(str(img_path), bgr)
                if not ok:
                    Streamer.logger.warning("cv2.imwrite failed: %s", img_path)

        else:
            # Save a single image
            img_path = out_path
            ok = cv2.imwrite(str(img_path), bgr)
            if not ok:
                Streamer.logger.error("cv2.imwrite failed: %s", img_path)


    def _do_save_results(self, preds: Any, p: Path, frame: int | None): 

        # Determine frame index
        txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))

        # self.__optional_save_or_show(preds, p)

        # Save results
        if self.args.save_txt:
            preds.save_txt(f"{txt_path}.txt", save_conf=self.args.save_conf)
        
        if self.args.save_crop:
            preds.save_crop(save_dir=self.save_dir / "crops", file_name=txt_path.stem if txt_path is not None else Path("unknown"))

    def _do_save_hazard_event(
        self,
        frame_path: Path,
        annotated: np.ndarray,
        crop_specs: List[Tuple[Path, np.ndarray]],
        csv_rows: List[List[Any]],
    ) -> None:
        cv2.imwrite(str(frame_path), annotated)
        for crop_path, crop in crop_specs:
            cv2.imwrite(str(crop_path), crop)
        self._append_hazard_csv(csv_rows)


    def save_predicted_images(self, save_path:str, frame:int) ->None: 
        im = self.plotted_img 
        
        if im is not None: 
            # self.save_queue.put((save_path, frame, im.copy()))
            self.save_queue.put(("save_frame", save_path, frame, im.copy()))


    def show(self, p:str)->None:
        im = self.plotted_img

        if im is None: 
            return 
        
        if self.use_roi and self.logic_module is not None and self.logic_module.get("ROI") is not None:
            self.logic_module["ROI"]._show_regions(im)

        for cls in self.points.keys(): 
            cv2.polylines(im, [self.points[cls]], isClosed=False, color=colors(cls, True), thickness=2)

        if self.docker_flag:
            self.proc_image = self._encode_preview_frame(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            return 

        elif platform.system() == "Linux" and p not in self.windows and not DEFAULT_CFG.gui: 
            self.windows.append(p)

            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        if DEFAULT_CFG.gui:
            try: 
                cv2.destroyAllWindows()
            except cv2.error: 
                pass
            self.proc_image = self._encode_preview_frame(im)

        elif self.args.show: 
            # cv2.imshow(winname=p, mat=im)
            # cv2.waitKey(300 if self.dataset.mode == 'image' else 1)
            pass


    def _encode_preview_frame(self, frame: np.ndarray) -> Optional[bytes]:
        t0 = time.perf_counter()
        if frame is None:
            return None

        preview_fps = float(getattr(self.args, "preview_fps", 8.0) or 0.0)
        now = time.perf_counter()
        if preview_fps > 0 and (now - self.preview_last_emit_ts) < (1.0 / preview_fps):
            return None

        self.preview_last_emit_ts = now

        max_width = int(getattr(self.args, "preview_max_width", 960) or 960)
        quality = int(getattr(self.args, "preview_jpeg_quality", 70) or 70)
        preview = frame

        if max_width > 0 and frame.shape[1] > max_width:
            scale = max_width / float(frame.shape[1])
            preview = cv2.resize(
                frame,
                (max_width, max(1, int(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded_image = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        self._record_stage_time(
            "preview_encode_ms",
            (time.perf_counter() - t0) * 1e3,
            frame_id=self._perf_active_frame_id,
        )
        return encoded_image.tobytes()


    def publish_preview(self, preview_queue: Any, producer_flag: Any = None) -> None:
        if preview_queue is None or self.proc_image is None:
            return

        if producer_flag is not None:
            producer_flag.value = True

        try:
            preview_queue.put_nowait(self.proc_image)
        except queue.Full:
            try:
                preview_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                preview_queue.put_nowait(self.proc_image)
            except queue.Full:
                pass


    def close_preview_stream(self, preview_queue: Any) -> None:
        if preview_queue is None:
            return

        try:
            preview_queue.put_nowait(None)
        except queue.Full:
            try:
                preview_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                preview_queue.put_nowait(None)
            except queue.Full:
                pass


    def run_callbacks(self, event:str)->None: 
        for cb in self.callbacks.get(event, []): 
            cb(self) 


    def add_callback(self, event: str, func:Any)->None: 
        self.callbacks[event].append(func)


    def capture_object_boxes(
        self,
        image:np.ndarray,
        results:Any,
        save:bool=True, 
        return_crops: bool = True, 
        max_objects: Optional[int] = None
    ): 

        if results is None or results.boxes is None:
            return {"crops": [], "boxes_xyxy": np.zeros((0,4), dtype=np.int32), "paths": []}

        # No detections 
        if results.boxes.xyxy.numel() == 0: 
            return {"crops": [], "boxes_xyxy": np.zeros((0,4), dtype=np.int32), "paths": []}

        if isinstance(image, torch.Tensor): 
            image = image.detach().cpu().numpy() 

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        orig_h, orig_w = image.shape[:2]
        infer_h, infer_w = self.cropped_imgsz if self.use_roi else results.orig_shape
        image = np.asarray(image) 
        xyxyi = _to_numpy_xyxy(results)
        # xyxyi = _unletterbox_xyxy_to_orig(xyxyi, (orig_h, orig_w), (infer_h, infer_w))

        xyxyi = xyxyi.round().astype(np.int32) 
        xyxyi[:, [0, 2]] = np.clip(xyxyi[:, [0, 2]], 0, orig_w)
        xyxyi[:, [1, 3]] = np.clip(xyxyi[:, [1, 3]], 0, orig_h)
        x1 = np.minimum(xyxyi[:, 0], xyxyi[:, 2])
        y1 = np.minimum(xyxyi[:, 1], xyxyi[:, 3])       
        x2 = np.maximum(xyxyi[:, 0], xyxyi[:, 2])
        y2 = np.maximum(xyxyi[:, 1], xyxyi[:, 3])
        xyxyi = np.stack([x1,y1,x2,y2], axis=1)
        #
        assert np.all(xyxyi[:, 0] <= xyxyi[:, 2]) 
        assert np.all(xyxyi[:, 1] <= xyxyi[:, 3]) 
        assert np.all(xyxyi[:, [0,2]] <= orig_w) 
        assert np.all(xyxyi[:, [1,3]] <= orig_h)
        #
        # # Optional Cap for performance boost
        if max_objects is not None and xyxyi.shape[0] > max_objects: 
            xyxyi = xyxyi[:max_objects]

        crops: List[np.ndarray] = [] 
        paths: List[str] = [] 

        out_dir = Path("assets") / self.cropped_image_dirname 
        if save: 
            _ensure_dir(out_dir)

        for idx, (x1, y1, x2, y2) in enumerate(xyxyi): 
            if x2 <= x1 or y2 <= y1: 
                continue 

            crop = image[y1:y2, x1:x2] 
            if return_crops: 
                crops.append(crop)
                self.run_metrics["crop_count"] += 1
                self.run_metrics["crop_total_bytes"] += int(crop.nbytes)
                self.run_metrics["crop_total_pixels"] += int(crop.shape[0] * crop.shape[1])

            if save: 
                fid = getattr(results, "path", "frame") 
                stem = Path(str(fid)).stem 
                fn = out_dir / f"{stem}_obj{idx}.jpg" 
                cv2.imwrite(str(fn), crop) 
                paths.append(str(fn))

        return {"crops": crops, "boxes_xyxy": xyxyi, "paths":paths}

    def _extract_frame_id(self, preds: Any) -> str:
        path = str(getattr(preds, "path", "frame"))
        stem = Path(path).stem
        if "_" in stem and stem.split("_")[-1].isdigit():
            return stem.split("_")[-1]
        return stem

    def log_detection_snapshot(self, preds: Any) -> None:
        if not getattr(self.args, "verbose", False):
            return

        boxes_obj = getattr(preds, "boxes", None)
        if boxes_obj is None or boxes_obj.xyxy is None or boxes_obj.xyxy.numel() == 0:
            return

        frame_id = self._extract_frame_id(preds)
        boxes = boxes_obj.xyxy.detach().cpu().tolist()
        confs = (
            boxes_obj.conf.detach().cpu().tolist()
            if getattr(boxes_obj, "conf", None) is not None
            else [0.0] * len(boxes)
        )
        cls_ids = (
            boxes_obj.cls.detach().cpu().tolist()
            if getattr(boxes_obj, "cls", None) is not None
            else [-1] * len(boxes)
        )

        summary = Counter()
        samples = []
        for idx, box in enumerate(boxes):
            cls_id = int(cls_ids[idx]) if idx < len(cls_ids) else -1
            if 0 <= cls_id < len(self.converter.class_names):
                class_name = self.converter.class_names[cls_id]
            else:
                class_name = f"class_{cls_id}"

            summary[class_name] += 1
            if idx < 3:
                x1, y1, x2, y2 = box
                conf = float(confs[idx]) if idx < len(confs) else 0.0
                samples.append(f"{class_name}@{conf:.2f}[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        Streamer.logger.info(
            "[Detections] frame=%s objects=%d summary=%s samples=%s",
            frame_id,
            len(boxes),
            ", ".join(f"{name}:{count}" for name, count in summary.most_common(4)),
            " | ".join(samples),
        )

    def _draw_scene_regions(self, image: np.ndarray) -> np.ndarray:
        lane = self.last_scene_masks.get("lane_mask")
        crosswalk = self.last_scene_masks.get("crosswalk_mask")
        if lane is None and crosswalk is None:
            return image

        out = image.copy()
        overlay = out.copy()

        if lane is not None and lane.size > 0 and cv2.countNonZero(lane) > 0:
            lane_contours, _ = cv2.findContours(lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lane_contours = [c for c in lane_contours if cv2.contourArea(c) >= 1000]
            if lane_contours:
                cv2.drawContours(overlay, lane_contours, -1, (0, 140, 255), thickness=-1)
                cv2.drawContours(out, lane_contours, -1, (0, 180, 255), thickness=2)

        if crosswalk is not None and crosswalk.size > 0 and cv2.countNonZero(crosswalk) > 0:
            cw_contours, _ = cv2.findContours(crosswalk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cw_contours = [c for c in cw_contours if cv2.contourArea(c) >= 300]
            if cw_contours:
                cv2.drawContours(overlay, cw_contours, -1, (255, 255, 0), thickness=-1)
                cv2.drawContours(out, cw_contours, -1, (255, 255, 0), thickness=2)

        return cv2.addWeighted(overlay, 0.18, out, 0.82, 0.0)

    def _draw_hazard_boxes(self, image: np.ndarray, hazards: List[Dict[str, Any]]) -> np.ndarray:
        out = image.copy()
        for hz in hazards:
            x1, y1, x2, y2 = hz.get("bbox_xyxy", [0, 0, 0, 0])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{hz.get('kind', 'Hazard')} | {str(hz.get('risk', '')).upper()}"
            cv2.putText(
                out,
                label,
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return out

    def _to_bgr(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return image
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy()
        return image

    def _append_hazard_csv(self, rows: List[List[Any]]) -> None:
        if not rows:
            return
        with self.hazard_csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def _record_hazard_evidence(self, preds: Any, image: np.ndarray, hazards: List[Dict[str, Any]], frame_id: Any = None) -> None:
        if image is None or not hazards:
            return

        frame_bgr = self._to_bgr(image)
        annotated = self._draw_scene_regions(frame_bgr)
        annotated = self._draw_hazard_boxes(annotated, hazards)

        event_ts = datetime.now(timezone.utc)
        event_ts_iso = event_ts.isoformat()
        ts = event_ts.strftime("%Y%m%dT%H%M%S.%fZ")
        frame_id = self._extract_frame_id(preds)
        frame_name = f"hazard_{ts}_f{frame_id}.jpg"
        frame_path = self.hazard_frames_dir / frame_name

        csv_rows: List[List[Any]] = []
        crop_specs: List[Tuple[Path, np.ndarray]] = []
        for idx, hz in enumerate(hazards):
            x1, y1, x2, y2 = hz.get("bbox_xyxy", [0, 0, 0, 0])
            x1 = int(np.clip(x1, 0, max(frame_bgr.shape[1] - 1, 0)))
            y1 = int(np.clip(y1, 0, max(frame_bgr.shape[0] - 1, 0)))
            x2 = int(np.clip(x2, 0, frame_bgr.shape[1]))
            y2 = int(np.clip(y2, 0, frame_bgr.shape[0]))
            crop_name = f"hazard_{ts}_f{frame_id}_obj{idx}.jpg"
            crop_path = self.hazard_crops_dir / crop_name
            if x2 > x1 and y2 > y1:
                crop = frame_bgr[y1:y2, x1:x2]
                crop_specs.append((crop_path, crop.copy()))
            else:
                crop_name = ""

            csv_rows.append(
                [
                    event_ts_iso,
                    frame_id,
                    hz.get("class_name", ""),
                    hz.get("category", ""),
                    hz.get("risk", ""),
                    hz.get("kind", ""),
                    hz.get("action", ""),
                    x1,
                    y1,
                    x2,
                    y2,
                    float(hz.get("lane_overlap", 0.0)),
                    float(hz.get("crosswalk_overlap", 0.0)),
                    frame_name,
                    crop_name,
                ]
            )

        t_save0 = time.perf_counter()
        self.save_queue.put(("save_hazard_event", frame_path, annotated.copy(), crop_specs, csv_rows))
        self._record_stage_time("event_saving_ms", (time.perf_counter() - t_save0) * 1e3, frame_id=frame_id)
        self.run_metrics["saved_hazard_events"] += 1
        self.run_metrics["saved_hazard_crops"] += len(crop_specs)
        self._publish_hazard_alert(preds=preds, hazards=hazards, frame_name=frame_name, frame_id=frame_id)

    def _publish_hazard_alert(self, preds: Any, hazards: List[Dict[str, Any]], frame_name: str, frame_id: Any = None) -> None:
        if self.mqtt_interface is None or not hazards:
            return

        risk_rank = {"low": 1, "medium": 2, "high": 3}
        highest = max((str(h.get("risk", "low")).lower() for h in hazards), key=lambda r: risk_rank.get(r, 1))
        payload = {
            "type": "hazard_event",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frame_id": self._extract_frame_id(preds),
            "attention_mode": "high_attention" if self.high_attention_countdown > 0 else "normal",
            "severity": highest,
            "event_image": frame_name,
            "hazards": [
                {
                    "class_name": h.get("class_name", ""),
                    "category": h.get("category", ""),
                    "risk": h.get("risk", ""),
                    "kind": h.get("kind", ""),
                    "bbox_xyxy": h.get("bbox_xyxy", []),
                    "action": h.get("action", ""),
                }
                for h in hazards
            ],
        }
        topic = f"{self.mqtt_interface.topic}/hazard"
        try:
            t_pub0 = time.perf_counter()
            self.mqtt_interface.publish(topic, payload)
            self._record_stage_time("mqtt_ms", (time.perf_counter() - t_pub0) * 1e3, frame_id=frame_id)
            self.run_metrics["mqtt_hazard_messages"] += 1
        except Exception:
            Streamer.logger.exception("Failed to publish hazard MQTT event")


    def optional_save_or_show(self, preds:Any, p:Any)-> None: 

        if self.args.save or self.args.show: 
            self.plotted_img = preds.plot(
                    line_width=self.args.line_width,
                    boxes=self.args.show_boxes,
                    conf=self.args.show_conf,
                    labels=self.args.show_labels,
            )
            if self.last_scene_masks.get("lane_mask") is not None:
                self.plotted_img = self._draw_scene_regions(self.plotted_img)
            hazards = getattr(preds, "hazard_events", None) or self.current_hazards
            if hazards:
                self.plotted_img = self._draw_hazard_boxes(self.plotted_img, hazards)

        if self.args.show:
            self.show(p)     

        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), int(self.dataset.count))    


    def __generate_mqtt_message(self, preds:Any, mqtt_messages, frame_index_list:list): 
            crops = defaultdict()
            for r, mes, fid in zip(preds, mqtt_messages, frame_index_list): 
                crops[fid] = []
                for bb in range(len(r.boxes.xyxy)):
                    cls_id = int(r.boxes.cls[bb].item())
                    cropped_detection = {
                        "img": mes["crops"][bb],
                        "bbox": mes["boxes_xyxy"][bb],
                        "cls":self.converter.class_names[cls_id],
                        "conf":r.boxes.conf[bb].item(),
                        "track_id":r.boxes.id[bb] if r.boxes.id is not None else None
                    }
                    crops[fid].append(cropped_detection)

            for cr_fr in crops: 
                self.mqtt_interface.publish_batch_from_crops(
                    crops=crops[cr_fr], 
                    cam_id="camera-1",
                    frame_id=cr_fr,
                    include_bbox=True
                ) 
                   
    def __generate_mqtt_message_no_motion(self, preds:Any, frame_index:list) -> Dict[str, Any]:
        frames = []
        for idx, frame_id in enumerate(frame_index): 
            boxes = preds[idx].boxes
            track_ids = boxes.id.tolist() if getattr(boxes, "id", None) is not None else []
            frames.append(
                {
                    "frame_id": frame_id,
                    "classes": boxes.cls.tolist(),
                    "boxes": boxes.xyxy.tolist(),
                    "track_ids": track_ids,
                }
            )

        return {
            "v": 1,
            "type": "no_detection_batch",
            "cam": "camera-1",
            "ts": int(time.time() * 1000),
            "inference_ran": False,
            "frames": frames,
        }
    

    @abstractmethod
    def _publish_mqtt_message(self, preds, mqtt_messages, frame_ids)->None: 
        if self.mqtt_interface is not None: 
            t0 = time.perf_counter()
            self.__generate_mqtt_message(preds, mqtt_messages, frame_ids)
            self._record_stage_time("mqtt_ms", (time.perf_counter() - t0) * 1e3, frame_ids=frame_ids)
            self.run_metrics["mqtt_batches"] += len(frame_ids)
            # self.mqtt_interface.publish(self.mqtt_interface.topic, message)


    @abstractmethod
    def _publish_mqtt_message_no_detection(self, preds, frame_index)->None: 
        if self.mqtt_interface is not None: 
            message = self.__generate_mqtt_message_no_motion(preds, frame_index)
            t0 = time.perf_counter()
            self.mqtt_interface.publish(self.mqtt_interface.topic, message)
            self._record_stage_time("mqtt_ms", (time.perf_counter() - t0) * 1e3, frame_ids=frame_index)
            self.run_metrics["mqtt_no_detection_batches"] += 1


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _to_numpy_xyxy(results) -> np.ndarray:
    """
    Returns Nx4 float32 xyxy in *inference/letterbox space* (whatever results.boxes.xyxy is).
    """
    xyxy = results.boxes.xyxy
    if isinstance(xyxy, torch.Tensor):
        xyxy = xyxy.detach().cpu().numpy()
    return np.asarray(xyxy, dtype=np.float32)


def _unletterbox_xyxy_to_orig(
    xyxy: np.ndarray,
    orig_hw: Tuple[int, int],
    infer_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Map boxes from letterboxed inference space -> original image pixel coords.
    """
    orig_h, orig_w = orig_hw
    infer_h, infer_w = infer_hw

    # Scale + padding used in letterbox
    scale = min(infer_w / orig_w, infer_h / orig_h)
    pad_w = (infer_w - orig_w * scale) / 2.0
    pad_h = (infer_h - orig_h * scale) / 2.0

    out = xyxy.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_w) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_h) / scale

    # Clip
    out[:, 0] = np.clip(out[:, 0], 0, orig_w)
    out[:, 2] = np.clip(out[:, 2], 0, orig_w)
    out[:, 1] = np.clip(out[:, 1], 0, orig_h)
    out[:, 3] = np.clip(out[:, 3], 0, orig_h)
    return out
