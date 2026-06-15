from obs_system.communication_module.interface import mqtt_interface
from obs_system.logic_module.dummy_logic.tile_kalman import TileDetectionSmoother
from obs_system.logic_module.dummy_logic.tile_activation import TileActivationWindow
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
        self.print_flag = True
        

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


    def postprocess(self, preds: Any, orig_image: Any) -> Any:
        return super().postprocess(preds, orig_image)


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


    def _publish_no_motion_preview(self, images_bgr, preview_queue, producer_flag) -> None:
        if not self.args.show or not images_bgr:
            return
        self._enqueue_async_sink(
            ("preview_frame", images_bgr[-1].copy(), preview_queue, producer_flag),
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
        cropped_original_images_bgr = original_images_bgr

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
                cropped_original_images_bgr = self.logic_module["ROI"].crop_image(original_images_bgr)
                if self.args.plot_performance:
                    roi_ms = (time.perf_counter() - _t0) * 1e3
                    self._record_stage_time("roi_ms", roi_ms, frame_ids=frame_ids)
                    _t1_rel = time.perf_counter() - stream_start
                    timeline_logger.log_span(batch_idx, "roi", _t0_rel, _t1_rel)
                    res_h, res_w = im0s[0].shape[:2] if len(im0s) else (0, 0)

        with StepContext(name="FishEyE Processing (Defish)", catch=(RuntimeError,), verbose=self.args.verbose):
            if self.logic_module["FEP"] is not None:
                Streamer.logger.debug("FEP enabled")
                fep = self.logic_module["FEP"]
                if not bool(getattr(fep, "use_tangent_views", False)):
                    t_defish0 = time.perf_counter()
                    im0s = fep._defish(im0s)
                    defish_ms = (time.perf_counter() - t_defish0) * 1e3
                    self._record_stage_time("defish_ms", defish_ms, frame_ids=frame_ids)

        with StepContext(name="BackGround Subtractor  (Motion-Gating)", catch=(RuntimeError, Exception), verbose=self.args.verbose):
            if self.args.plot_performance:
                _t0 = time.perf_counter()
                _t0_rel = _t0 - stream_start
            subtractor_inst = self.logic_module["SUBTRACTOR"]
            mfgs, lanes_final = subtractor_inst.detect(im0s, save_img=False)
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

        fg_masks = getattr(subtractor_inst, "_last_batch_fg_masks", None) or []

        # When FEP is active im0s has been remapped; YOLO boxes will be in that
        # undistorted coordinate space, so all display / postprocess paths must
        # also use the undistorted frames.  When FEP is off the two lists are
        # identical content (no copy needed).
        fep_active = (
            self.logic_module is not None
            and self.logic_module.get("FEP") is not None
            and not bool(getattr(self.logic_module.get("FEP"), "use_tangent_views", False))
        )
        display_images_bgr = list(im0s) if fep_active else original_images_bgr

        return {
            "paths": paths,
            "im0s": im0s,
            "frame_ids": frame_ids,
            "frame_read_ms": frame_read_ms,
            "original_images_bgr": original_images_bgr,
            "cropped_original_images_bgr": cropped_original_images_bgr,
            "display_images_bgr": display_images_bgr,
            "mfgs": mfgs,
            "fg_masks": fg_masks,
            "roi_ms": roi_ms,
            "mog2_ms": mog2_ms,
            "defish_ms": defish_ms,
            "res_h": res_h,
            "res_w": res_w,
            "skip_reason": "no_motion" if not any(mfgs) else None,
        }

    def _get_road_filter_mask(self, classes_t: "torch.Tensor") -> "torch.Tensor":
        """Return a boolean keep-mask that removes impossible road-scene classes."""
        class_names = getattr(getattr(self, "converter", None), "class_names", [])
        impossible_ids = {
            i for i, name in enumerate(class_names)
            if name.lower() in ROAD_IMPOSSIBLE_CLASSES
        }
        if not impossible_ids:
            return torch.ones(len(classes_t), dtype=torch.bool, device=classes_t.device)
        keep = torch.tensor(
            [int(c) not in impossible_ids for c in classes_t.tolist()],
            dtype=torch.bool,
            device=classes_t.device,
        )
        return keep

    def _extract_motion_detections(
        self,
        fg_mask: Optional[np.ndarray],
        im0_hw: tuple,
    ) -> tuple:
        """
        Extract bounding boxes from a binary MOG2 fg_mask and return them as
        low-confidence detections to be merged with YOLO detections.

        Confidence is fixed at MOTION_BOX_CONFIDENCE (below ByteTrack's
        track_activation_threshold=0.25) so these detections only reinforce
        existing tracks and never spawn new ones — the allow_spawn=False
        behaviour from CarDet_Dummy_EdgeAI.
        """
        if fg_mask is None or fg_mask.size == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)

        mask_h, mask_w = fg_mask.shape[:2]
        frame_h, frame_w = im0_hw
        sx = float(frame_w) / max(float(mask_w), 1.0)
        sy = float(frame_h) / max(float(mask_h), 1.0)
        total_px = float(mask_h * mask_w)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, scores, classes = [], [], []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            area_ratio = area / max(total_px, 1.0)
            if area_ratio < MOTION_BOX_MIN_AREA_RATIO or area_ratio > MOTION_BOX_MAX_AREA_RATIO:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / max(float(h), 1e-6)
            if aspect < MOTION_BOX_MIN_ASPECT or aspect > MOTION_BOX_MAX_ASPECT:
                continue
            boxes.append([x * sx, y * sy, (x + w) * sx, (y + h) * sy])
            scores.append(MOTION_BOX_CONFIDENCE)
            classes.append(0)

        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)

        return (
            np.array(boxes, dtype=np.float32),
            np.array(scores, dtype=np.float32),
            np.array(classes, dtype=np.int64),
        )

    def _merge_motion_boxes(
        self,
        frame_det: "FrameDetections",
        fg_mask: Optional[np.ndarray],
        im0_hw: tuple,
    ) -> "FrameDetections":
        """Merge allow_spawn=False motion boxes into frame_det at low confidence."""
        mot_boxes, mot_scores, mot_classes = self._extract_motion_detections(fg_mask, im0_hw)
        if len(mot_boxes) == 0:
            return frame_det

        # Stay on the same device as YOLO tensors; fall back to CUDA when the frame
        # is empty (no existing boxes to infer device from) so motion tensors never
        # force a host-side copy of the main detection pipeline.
        if not frame_det.is_empty:
            device = frame_det.boxes.device
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        mot_boxes_t = torch.as_tensor(mot_boxes, dtype=torch.float32).to(device)
        if self.use_roi and self.logic_module is not None and self.logic_module.get("ROI") is not None:
            mot_boxes_t = self.logic_module["ROI"].translate_bounding_boxes(
                results=mot_boxes_t,
                orig_img_shape=self.original_imgsz,
            )
            if mot_boxes_t is None or mot_boxes_t.numel() == 0:
                return frame_det
            mot_boxes_t = mot_boxes_t.to(device)

        mot_scores_t = torch.as_tensor(mot_scores, dtype=torch.float32).to(device)
        mot_classes_t = torch.as_tensor(mot_classes, dtype=torch.int64).to(device)

        if frame_det.is_empty:
            return FrameDetections(
                frame_id=frame_det.frame_id,
                batch_index=frame_det.batch_index,
                orig_img=frame_det.orig_img,
                boxes=mot_boxes_t,
                scores=mot_scores_t,
                classes=mot_classes_t,
            )

        return FrameDetections(
            frame_id=frame_det.frame_id,
            batch_index=frame_det.batch_index,
            orig_img=frame_det.orig_img,
            boxes=torch.cat([frame_det.boxes, mot_boxes_t], dim=0),
            scores=torch.cat([frame_det.scores, mot_scores_t], dim=0),
            classes=torch.cat([frame_det.classes, mot_classes_t], dim=0),
        )

    def _stage_b_panorama_inference(self, stage_a, model, profilers, activities, stream_start, timeline_logger, batch_idx):
        """
        Panorama-mode stage B.

        Generates perspective views from every motion-passed frame, collects ALL
        active views from ALL frames into a single flat list, then runs YOLO in
        microbatches across that list (same structure as the tile path).  This
        keeps the number of TensorRT/model calls equal to ceil(total_views/BATCH_SIZE)
        regardless of how many frames are in the batch, recovering the performance
        that was lost when each frame triggered its own model call.

        Back-projection and cross-view NMS are applied per frame after inference.
        """
        im0s = stage_a["im0s"]
        original_images_bgr = stage_a["original_images_bgr"]
        display_images_bgr = stage_a.get("display_images_bgr", original_images_bgr)
        mfgs = stage_a["mfgs"]
        frame_ids = stage_a["frame_ids"]
        fg_masks = stage_a.get("fg_masks") or []
      
        if self.print_flag: 
            print(f"Original resolution {original_images_bgr[0].shape} | Cropped Resolution {im0s[0].shape}")
            self.print_flag = False

        reprojector = self.logic_module.get("PANORAMA")

        # ── Pass 1: generate all active views across the whole batch ─────────
        # all_view_items: flat list of (view_bgr, bni, v_id)
        all_view_items: List = []
        for bni, (keep_frame, panorama_bgr) in enumerate(zip(mfgs, im0s)):
            if not keep_frame or reprojector is None:
                continue
            fg_mask = fg_masks[bni] if bni < len(fg_masks) else None
            for view_img, v_id, has_motion in reprojector.get_views(panorama_bgr, fg_mask_small=fg_mask):
                if has_motion:
                    all_view_items.append((view_img, bni, v_id))

        # Per-frame accumulator: bni → {"boxes": [], "scores": [], "classes": []}
        frame_acc: dict = {}
        total_preprocess_ms = 0.0
        total_inference_ms = 0.0
        total_nms_ms = 0.0

        # ── Pass 2: microbatch inference across all views ────────────────────
        micro = BATCH_SIZE
        for mb_start in range(0, max(1, len(all_view_items)), micro):
            mb = all_view_items[mb_start: mb_start + micro]
            if not mb:
                break

            view_imgs_mb = [item[0] for item in mb]
            bni_mb = [item[1] for item in mb]
            vid_mb = [item[2] for item in mb]

            t_pre0 = time.perf_counter()
            with profilers[0]:
                view_tensors = self.preprocess(view_imgs_mb)
            total_preprocess_ms += (time.perf_counter() - t_pre0) * 1e3

            t_inf0 = time.perf_counter()
            with profilers[1]:
                raw_out = self.model(view_tensors, orig_imgs=None, debug=False)
            total_inference_ms += (time.perf_counter() - t_inf0) * 1e3

            if isinstance(raw_out, tuple) and len(raw_out) == 2 and isinstance(raw_out[0], tuple):
                (i_boxes, i_scores, i_classes), event = raw_out
            else:
                i_boxes, i_scores, i_classes = raw_out
                event = None

            if event is not None and torch.cuda.is_available():
                torch.cuda.current_stream().wait_event(event)

            for bni, v_id, boxes, scores, classes in zip(bni_mb, vid_mb, i_boxes, i_scores, i_classes):
                boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes, dtype=torch.float32)
                if boxes_t.numel() == 0:
                    continue

                scores_t = (scores if torch.is_tensor(scores) else torch.as_tensor(scores)).to(torch.float32)
                classes_t = (classes if torch.is_tensor(classes) else torch.as_tensor(classes)).to(torch.int64)
                boxes_t = boxes_t.to(torch.float32)

                road_mask = self._get_road_filter_mask(classes_t)
                if not road_mask.any():
                    continue
                boxes_t = boxes_t[road_mask]
                scores_t = scores_t[road_mask]
                classes_t = classes_t[road_mask]

                boxes_np = boxes_t.cpu().numpy().astype(np.float32)
                boxes_pano = reprojector.backproject_boxes(boxes_np, v_id)

                valid = (boxes_pano[:, 2] > boxes_pano[:, 0]) & (boxes_pano[:, 3] > boxes_pano[:, 1])
                if not valid.any():
                    continue

                entry = frame_acc.setdefault(bni, {"boxes": [], "scores": [], "classes": []})
                entry["boxes"].append(boxes_pano[valid])
                entry["scores"].append(scores_t.cpu().numpy()[valid])
                entry["classes"].append(classes_t.cpu().numpy()[valid])

        # ── Pass 3: per-frame NMS + ROI back-translation ─────────────────────
        frames: List[FrameDetections] = []
        for bni, (keep_frame, fid) in enumerate(zip(mfgs, frame_ids)):
            orig_img = display_images_bgr[bni]
            if self.args.save or self.args.show:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            fg_mask_bni = fg_masks[bni] if bni < len(fg_masks) else None

            if not keep_frame:
                frames.append(FrameDetections.empty(frame_id=fid, batch_index=bni, orig_img=orig_img))
                continue

            if bni not in frame_acc or not frame_acc[bni]["boxes"]:
                fd = FrameDetections.empty(frame_id=fid, batch_index=bni, orig_img=orig_img)
                frames.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni].shape[:2]))
                continue

            accum = frame_acc[bni]
            all_boxes = np.concatenate(accum["boxes"], axis=0).astype(np.float32)
            all_scores = np.concatenate(accum["scores"], axis=0).astype(np.float32)
            all_classes = np.concatenate(accum["classes"], axis=0).astype(np.int64)

            boxes_t_all = torch.as_tensor(all_boxes, dtype=torch.float32)
            scores_t_all = torch.as_tensor(all_scores, dtype=torch.float32)
            classes_t_all = torch.as_tensor(all_classes, dtype=torch.int64)

            t_nms0 = time.perf_counter()
            keep_idx = batched_nms(boxes_t_all, scores_t_all, classes_t_all.long(), iou_threshold=PANORAMA_NMS_IOU)
            total_nms_ms += (time.perf_counter() - t_nms0) * 1e3

            boxes_t_all = boxes_t_all[keep_idx]
            scores_t_all = scores_t_all[keep_idx]
            classes_t_all = classes_t_all[keep_idx]

            if self.use_roi and boxes_t_all.numel() > 0:
                boxes_t_all = self.logic_module["ROI"].translate_bounding_boxes(
                    results=boxes_t_all,
                    orig_img_shape=self.original_imgsz,
                )

            if boxes_t_all.numel() == 0:
                fd = FrameDetections.empty(frame_id=fid, batch_index=bni, orig_img=orig_img)
                frames.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni].shape[:2]))
                continue

            fd = FrameDetections(
                frame_id=fid,
                batch_index=bni,
                orig_img=orig_img,
                boxes=boxes_t_all,
                scores=scores_t_all,
                classes=classes_t_all,
            )
            frames.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni].shape[:2]))

        return {
            "detections": DetectionBatch(frames=frames),
            "preprocess_ms": total_preprocess_ms,
            "inference_ms": total_inference_ms,
            "nms_ms": total_nms_ms,
        }

    def _stage_b_fisheye_view_inference(self, stage_a, model, profilers, activities, stream_start, timeline_logger, batch_idx):
        """
        Fisheye tangent-view Stage B.

        Generates pinhole views from each motion-passed fisheye frame, runs YOLO
        over a flat microbatched view list, back-projects detections to the
        fisheye frame, then applies cross-view NMS per source frame.
        """
        im0s = stage_a["im0s"]
        original_images_bgr = stage_a["original_images_bgr"]
        display_images_bgr = stage_a.get("display_images_bgr", original_images_bgr)
        mfgs = stage_a["mfgs"]
        frame_ids = stage_a["frame_ids"]
        fg_masks = stage_a.get("fg_masks") or []

        fep = self.logic_module.get("FEP")

        all_view_items: List = []
        total_preprocess_ms = 0.0
        total_inference_ms = 0.0
        total_nms_ms = 0.0

        t_view0 = time.perf_counter()
        for bni, (keep_frame, fisheye_bgr) in enumerate(zip(mfgs, im0s)):
            if not keep_frame or fep is None:
                continue
            fg_mask = fg_masks[bni] if bni < len(fg_masks) else None
            for view_img, v_id, has_motion in fep.get_views(fisheye_bgr, fg_mask_small=fg_mask):
                if has_motion and view_img is not None:
                    all_view_items.append((view_img, bni, v_id))
        total_preprocess_ms += (time.perf_counter() - t_view0) * 1e3

        frame_acc: dict = {}
        micro = BATCH_SIZE
        for mb_start in range(0, max(1, len(all_view_items)), micro):
            mb = all_view_items[mb_start: mb_start + micro]
            if not mb:
                break

            view_imgs_mb = [item[0] for item in mb]
            bni_mb = [item[1] for item in mb]
            vid_mb = [item[2] for item in mb]

            t_pre0 = time.perf_counter()
            with profilers[0]:
                view_tensors = self.preprocess(view_imgs_mb)
            total_preprocess_ms += (time.perf_counter() - t_pre0) * 1e3

            t_inf0 = time.perf_counter()
            with profilers[1]:
                raw_out = self.model(view_tensors, orig_imgs=None, debug=False)
            total_inference_ms += (time.perf_counter() - t_inf0) * 1e3

            if isinstance(raw_out, tuple) and len(raw_out) == 2 and isinstance(raw_out[0], tuple):
                (i_boxes, i_scores, i_classes), event = raw_out
            else:
                i_boxes, i_scores, i_classes = raw_out
                event = None

            if event is not None and torch.cuda.is_available():
                torch.cuda.current_stream().wait_event(event)

            for bni, v_id, boxes, scores, classes in zip(bni_mb, vid_mb, i_boxes, i_scores, i_classes):
                boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes, dtype=torch.float32)
                if boxes_t.numel() == 0:
                    continue

                scores_t = (scores if torch.is_tensor(scores) else torch.as_tensor(scores)).to(torch.float32)
                classes_t = (classes if torch.is_tensor(classes) else torch.as_tensor(classes)).to(torch.int64)
                boxes_t = boxes_t.to(torch.float32)

                road_mask = self._get_road_filter_mask(classes_t)
                if not road_mask.any():
                    continue
                boxes_t = boxes_t[road_mask]
                scores_t = scores_t[road_mask]
                classes_t = classes_t[road_mask]

                boxes_np = boxes_t.cpu().numpy().astype(np.float32)
                boxes_fish = fep.backproject_view_boxes(boxes_np, v_id)

                valid = (boxes_fish[:, 2] > boxes_fish[:, 0]) & (boxes_fish[:, 3] > boxes_fish[:, 1])
                if not valid.any():
                    continue

                entry = frame_acc.setdefault(bni, {"boxes": [], "scores": [], "classes": []})
                entry["boxes"].append(boxes_fish[valid])
                entry["scores"].append(scores_t.cpu().numpy()[valid])
                entry["classes"].append(classes_t.cpu().numpy()[valid])

        frames: List[FrameDetections] = []
        for bni, (keep_frame, fid) in enumerate(zip(mfgs, frame_ids)):
            orig_img = display_images_bgr[bni]
            if self.args.save or self.args.show:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            fg_mask_bni = fg_masks[bni] if bni < len(fg_masks) else None

            if not keep_frame:
                frames.append(FrameDetections.empty(frame_id=fid, batch_index=bni, orig_img=orig_img))
                continue

            if bni not in frame_acc or not frame_acc[bni]["boxes"]:
                fd = FrameDetections.empty(frame_id=fid, batch_index=bni, orig_img=orig_img)
                frames.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni].shape[:2]))
                continue

            accum = frame_acc[bni]
            all_boxes = np.concatenate(accum["boxes"], axis=0).astype(np.float32)
            all_scores = np.concatenate(accum["scores"], axis=0).astype(np.float32)
            all_classes = np.concatenate(accum["classes"], axis=0).astype(np.int64)

            boxes_t_all = torch.as_tensor(all_boxes, dtype=torch.float32)
            scores_t_all = torch.as_tensor(all_scores, dtype=torch.float32)
            classes_t_all = torch.as_tensor(all_classes, dtype=torch.int64)

            t_nms0 = time.perf_counter()
            keep_idx = batched_nms(boxes_t_all, scores_t_all, classes_t_all.long(), iou_threshold=FISHEYE_VIEW_NMS_IOU)
            total_nms_ms += (time.perf_counter() - t_nms0) * 1e3

            boxes_t_all = boxes_t_all[keep_idx]
            scores_t_all = scores_t_all[keep_idx]
            classes_t_all = classes_t_all[keep_idx]

            if self.use_roi and boxes_t_all.numel() > 0:
                boxes_t_all = self.logic_module["ROI"].translate_bounding_boxes(
                    results=boxes_t_all,
                    orig_img_shape=self.original_imgsz,
                )

            if boxes_t_all.numel() == 0:
                fd = FrameDetections.empty(frame_id=fid, batch_index=bni, orig_img=orig_img)
                frames.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni].shape[:2]))
                continue

            fd = FrameDetections(
                frame_id=fid,
                batch_index=bni,
                orig_img=orig_img,
                boxes=boxes_t_all,
                scores=scores_t_all,
                classes=classes_t_all,
            )
            frames.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni].shape[:2]))

        return {
            "detections": DetectionBatch(frames=frames),
            "preprocess_ms": total_preprocess_ms,
            "inference_ms": total_inference_ms,
            "nms_ms": total_nms_ms,
        }

    def _stage_b_inference_and_nms(self, stage_a, model, profilers, activities, stream_start, timeline_logger, batch_idx):
        im0s = stage_a["im0s"]
        original_images_bgr = stage_a["original_images_bgr"]
        display_images_bgr = stage_a.get("display_images_bgr", original_images_bgr)
        cropped_original_images_bgr = stage_a.get("cropped_original_images_bgr", original_images_bgr)
        mfgs = stage_a["mfgs"]
        frame_ids = stage_a["frame_ids"]
        
        if self.print_flag: 
            print(f"Original resolution {original_images_bgr[0].shape} | Cropped Resolution {im0s[0].shape}")
            self.print_flag = False

        for i, keep_frame in enumerate(mfgs):
            if not keep_frame:
                im0s[i] = empty_image(im0s[i])
                original_images_bgr[i] = empty_image(original_images_bgr[i])
                display_images_bgr[i] = empty_image(display_images_bgr[i])

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
                        orig_imgs=original_images_bgr if not self.use_roi else cropped_original_images_bgr,
                        debug=self.args.verbose,
                    )
                if not os.path.exists("assets/trace_jsons"):
                    os.mkdir("assets/trace_jsons")
                model_tag = getattr(self, "model_tag", type(self.model).__name__)
                prof.export_chrome_trace(f"assets/trace_jsons/trace_{model_tag}.json")
            else:
                infer_outputs = self.model(
                    images,
                    orig_imgs=original_images_bgr if not self.use_roi else cropped_original_images_bgr,
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

        fg_masks = stage_a.get("fg_masks") or []
        frames: List[FrameDetections] = []
        nms_ms = 0.0
        for bni, fid in enumerate(frame_ids):
            orig_img = display_images_bgr[bni]
            if self.args.save or self.args.show:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            boxes = i_boxes[bni]
            scores = i_scores[bni]
            cls_ = i_classes[bni]
            fg_mask_bni = fg_masks[bni] if bni < len(fg_masks) else None

            if not mfgs[bni] or boxes is None or len(boxes) == 0:
                fd = FrameDetections.empty(frame_id=fid, batch_index=bni, orig_img=orig_img)
                if mfgs[bni]:
                    fd = self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni].shape[:2])
                frames.append(fd)
                continue

            boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes)
            scores_t = scores if torch.is_tensor(scores) else torch.as_tensor(scores)
            classes_t = cls_ if torch.is_tensor(cls_) else torch.as_tensor(cls_)
            scores_t = scores_t.to(dtype=torch.float32)
            classes_t = classes_t.to(dtype=torch.int64)
            boxes_t = boxes_t.to(dtype=torch.float32)

            road_mask = self._get_road_filter_mask(classes_t)
            boxes_t = boxes_t[road_mask]
            scores_t = scores_t[road_mask]
            classes_t = classes_t[road_mask]

            if boxes_t.numel() == 0:
                fd = FrameDetections.empty(frame_id=fid, batch_index=bni, orig_img=orig_img)
                frames.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni].shape[:2]))
                continue

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
                fd = FrameDetections.empty(frame_id=fid, batch_index=bni, orig_img=orig_img)
                frames.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni].shape[:2]))
                continue

            fd = FrameDetections(
                frame_id=fid,
                batch_index=bni,
                orig_img=orig_img,
                boxes=boxes_t,
                scores=scores_t,
                classes=classes_t,
            )
            frames.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni].shape[:2]))

        stage_a["im0s"] = im0s
        return {
            "detections": DetectionBatch(frames=frames),
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "nms_ms": nms_ms,
        }

    def _stage_c_tracking_and_hazard_logic(self, detection_batch: DetectionBatch, orig_images_bgr, profilers):
        # YOLO-gated lane calibration: pass batch indices where YOLO confirmed
        # at least one detection (score >= CONF_THR) to the subtractor accumulator.
        # Motion boxes injected by _merge_motion_boxes have confidence 0.15 <
        # CONF_THR so they are automatically excluded — only real YOLO vehicle
        # detections drive the lane calibration window.
        _subtractor = self.logic_module.get("SUBTRACTOR") if self.logic_module is not None else None
        if _subtractor is not None and hasattr(_subtractor, "notify_vehicle_detections"):
            _class_names = getattr(getattr(self, "converter", None), "class_names", [])
            _vehicle_ids = {
                i for i, n in enumerate(_class_names) if n.lower() in {v.lower() for v in VOCAB}
            }
            _yolo_confirmed = []
            for fd in detection_batch:
                if fd.is_empty or fd.scores is None or fd.classes is None:
                    continue
                scores_cpu = fd.scores.cpu()
                classes_cpu = fd.classes.cpu()
                high_conf = scores_cpu >= CONF_THR
                if not high_conf.any():
                    continue
                if _vehicle_ids:
                    is_vehicle = torch.tensor(
                        [int(c) in _vehicle_ids for c in classes_cpu.tolist()],
                        dtype=torch.bool,
                    )
                    if bool((high_conf & is_vehicle).any()):
                        _yolo_confirmed.append(fd.batch_index)
                else:
                    # No class mapping available — fall back to score-only gate
                    _yolo_confirmed.append(fd.batch_index)
            if _yolo_confirmed:
                _subtractor.notify_vehicle_detections(_yolo_confirmed)

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

        # Pre-compute scene masks ONCE for the whole batch and cache them so
        # that postprocess() per frame skips redundant get_scene_masks() calls.
        target_hw = orig_images_bgr[0].shape[:2] if orig_images_bgr else None
        batch_scene_masks = self._resolve_scene_masks(target_hw=target_hw)
        self._batch_scene_cache = self._prepare_scene_mask_cache(batch_scene_masks)
        try:
            with profilers[2]:
                postprocessed = self.postprocess_batch(tracked_results, orig_images=orig_images_bgr)
        finally:
            self._batch_scene_cache = None  # restore per-frame fallback behaviour

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


    def _apply_kalman_smoother(self, detection_batch: DetectionBatch) -> DetectionBatch:
        """
        Run TileDetectionSmoother over each frame in the batch.

        Called after cross-tile / cross-view NMS and before ByteTracker (Stage C).
        Empty frames are still passed through so the smoother can emit predicted
        boxes for confirmed tracks that YOLO missed on this frame.
        """
        smoother: TileDetectionSmoother = self._tile_kalman  # type: ignore[assignment]
        smoothed: list[FrameDetections] = []
        for frame in detection_batch:
            if frame.is_empty:
                boxes_np = np.zeros((0, 4), np.float32)
                scores_np = np.zeros((0,), np.float32)
                classes_np = np.zeros((0,), np.int64)
            else:
                boxes_np = frame.boxes.float().cpu().numpy()  # type: ignore[union-attr]
                scores_np = frame.scores.float().cpu().numpy()  # type: ignore[union-attr]
                classes_np = frame.classes.cpu().numpy()  # type: ignore[union-attr]

            boxes_out, scores_out, classes_out = smoother.update(boxes_np, scores_np, classes_np)

            if len(boxes_out) == 0:
                smoothed.append(FrameDetections.empty(
                    frame_id=frame.frame_id,
                    batch_index=frame.batch_index,
                    orig_img=frame.orig_img,
                ))
            else:
                smoothed.append(FrameDetections(
                    frame_id=frame.frame_id,
                    batch_index=frame.batch_index,
                    orig_img=frame.orig_img,
                    boxes=torch.as_tensor(boxes_out, dtype=torch.float32),
                    scores=torch.as_tensor(scores_out, dtype=torch.float32),
                    classes=torch.as_tensor(classes_out, dtype=torch.int64),
                ))
        return DetectionBatch(frames=smoothed)

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
                recalibration_interval = getattr(self.args, "lane_recalibration_interval_frames", None)
                subtractor.configure_runtime_recalibration(
                    enabled=bool(recalibration_interval and int(recalibration_interval) > 0),
                    interval_frames=recalibration_interval,
                )

            # Unnecessary now. 
            # if not getattr(self.source_type, "stream", False):
                # empty_image = f"{EMPTY_IMAGE_PATH}"
                # if not os.path.exists(empty_image):
                #     raise FileNotFoundError(f"Empty image path for background subtraction does not exist: {empty_image}")
                # empty_image = cv2.imread(empty_image)
                # self.logic_module['SUBTRACTOR'].warm_up(empty_image, trials=TRIALS)
                # self._sync_subtractor_warmup_state()
                

            # Panorama mode overrides tiling: views are generated internally.
            use_panorama = (
                bool(getattr(self.args, "panorama", False))
                and self.logic_module is not None
                and self.logic_module.get("PANORAMA") is not None
            )
            use_fisheye_views = (
                not use_panorama
                and self.logic_module is not None
                and self.logic_module.get("FEP") is not None
                and bool(getattr(self.logic_module.get("FEP"), "use_tangent_views", False))
            )

            # UI/CLI force_tiles takes highest priority; otherwise auto-detect from image size.
            # A class-level force_streaming_no_tiles=True is overridden by the explicit flag.
            force_tiles_flag = bool(getattr(self.args, "force_tiles", False))
            force_no_tiles = bool(getattr(self, "force_streaming_no_tiles", False)) and not force_tiles_flag
            auto_tile = (self.orig_width // TILE_SIZE) > TILE_THR or (self.orig_height // TILE_SIZE) >= TILE_THR
            use_tiles = (not use_panorama) and (not use_fisheye_views) and (force_tiles_flag or (auto_tile and not force_no_tiles))

            # Kalman smoother is active only when multiple tile/view detections are merged
            # per frame. The regular single-pass path relies solely on ByteTracker
            # for temporal stability.
            if use_panorama or use_fisheye_views or use_tiles:
                self._tile_kalman: Optional[TileDetectionSmoother] = TileDetectionSmoother(
                    max_age=TILE_KALMAN_MAX_AGE,
                    min_hits=TILE_KALMAN_MIN_HITS,
                    iou_threshold=TILE_KALMAN_IOU_THRESHOLD,
                )
            else:
                self._tile_kalman = None

            # Per-tile activation persistence window — tiles path only.
            # Panorama already has per-view motion gating (PANORAMA_MIN_MOTION_FRACTION).
            if use_tiles and TILE_ACTIVATION_ENABLED:
                self._tile_activation: Optional[TileActivationWindow] = TileActivationWindow(
                    persist_frames=TILE_ACTIVATION_PERSIST_FRAMES,
                    motion_min_ratio=TILE_ACTIVATION_MOTION_MIN_RATIO,
                )
            else:
                self._tile_activation = None

            if use_panorama:
                Streamer.logger.info(
                    "Run Inference in Panorama Mode (%dx%d, %d views)",
                    self.orig_width, self.orig_height,
                    self.logic_module["PANORAMA"].n_views,
                )
                return self._stream_inference_impl(
                    model=model,
                    producer_flag=producer_flag,
                    preview_queue=preview_queue,
                    profilers=profilers,
                    activities=activities,
                    start_time=start_time,
                )

            if use_fisheye_views:
                Streamer.logger.info(
                    "Run Inference in Fisheye Tangent-View Mode (%dx%d, %d views)",
                    self.orig_width, self.orig_height,
                    self.logic_module["FEP"].n_views,
                )
                return self._stream_inference_impl(
                    model=model,
                    producer_flag=producer_flag,
                    preview_queue=preview_queue,
                    profilers=profilers,
                    activities=activities,
                    start_time=start_time,
                )

            if use_tiles:
                Streamer.logger.info(
                    "Run Inference with Tiles (force=%s, auto=%s, %dx%d)",
                    force_tiles_flag, auto_tile, self.orig_width, self.orig_height,
                )
                return self._stream_inference_impl_tiles(
                    model=model,
                    producer_flag=producer_flag,
                    preview_queue=preview_queue,
                    profilers=profilers,
                    activities=activities,
                    start_time=start_time,
                )

            Streamer.logger.info("Run Inference without Tiles (%dx%d)", self.orig_width, self.orig_height)
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
        batch_queue = queue.Queue(maxsize=4)

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
            display_images_bgr = stage_a.get("display_images_bgr", original_images_bgr)
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
                if self.args.save:
                    for i, fid in enumerate(frame_ids):
                        p = Path(paths[i])
                        save_path = str(self.save_dir / p.name)
                        frame_count = int(getattr(self.dataset, "count", fid))
                        orig_rgb = cv2.cvtColor(original_images_bgr[i], cv2.COLOR_BGR2RGB)
                        self.save_queue.put(("save_frame", save_path, frame_count, orig_rgb))
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
                if getattr(self, "_tile_kalman", None) is not None:
                    _zero_b = np.zeros((0, 4), np.float32)
                    _zero_s = np.zeros((0,), np.float32)
                    _zero_c = np.zeros((0,), np.int64)
                    for _ in im0s:
                        self._tile_kalman.update(_zero_b, _zero_s, _zero_c)
                self.step_attention_state()
                continue

            _use_panorama = (
                bool(getattr(self.args, "panorama", False))
                and self.logic_module is not None
                and self.logic_module.get("PANORAMA") is not None
            )
            _use_fisheye_views = (
                not _use_panorama
                and self.logic_module is not None
                and self.logic_module.get("FEP") is not None
                and bool(getattr(self.logic_module.get("FEP"), "use_tangent_views", False))
            )
            if _use_panorama:
                stage_b = self._stage_b_panorama_inference(stage_a, model, profilers, activities, stream_start, timeline_logger, batch_idx)
            elif _use_fisheye_views:
                stage_b = self._stage_b_fisheye_view_inference(stage_a, model, profilers, activities, stream_start, timeline_logger, batch_idx)
            else:
                stage_b = self._stage_b_inference_and_nms(stage_a, model, profilers, activities, stream_start, timeline_logger, batch_idx)
            preprocess_ms = stage_b["preprocess_ms"]
            inference_ms = stage_b["inference_ms"]
            nms_ms = stage_b["nms_ms"]
            detection_batch = stage_b["detections"]

            # Kalman smoother fills tile-boundary / view-overlap gaps. The regular
            # no-tiles path relies on ByteTracker.
            if (_use_panorama or _use_fisheye_views) and getattr(self, "_tile_kalman", None) is not None:
                detection_batch = self._apply_kalman_smoother(detection_batch)

            postprocess_total_t0 = time.perf_counter()
            if self.args.plot_performance:
                _tpost = time.perf_counter()
                _tpost_rel = _tpost - stream_start
            frame_bundles, preds = self._stage_c_tracking_and_hazard_logic(detection_batch, display_images_bgr, profilers)
            frame_log_rows = self._stage_d_dispatch_optional_sinks(
                preds,
                frame_bundles,
                paths,
                display_images_bgr,
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


    @mem_profile
    def _stream_inference_impl_tiles(self, **kwargs) -> Generator[Optional[Any], None, None]:
        """
        Full tiled inference pipeline with feature parity to _stream_inference_impl.

        Stage A  – reuses _stage_a_acquire_and_gate (ROI, MOG2, FEP).
        Stage B  – splits each motion frame into tiles, runs microbatch inference,
                   reconstructs tile-local boxes to frame coordinates, global NMS per frame.
        Stages C/D – reuse _stage_c_tracking_and_hazard_logic and
                     _stage_d_dispatch_optional_sinks unchanged.
        Perf logging, MQTT, preview, benchmarking – identical to the no-tiles path.
        """
        FPS_WINDOW = 100
        fps_times = collections.deque(maxlen=FPS_WINDOW)
        fps = 0
        stream_start = time.perf_counter()
        last_fps_log = stream_start
        total_frames = 0

        perf_log_path = getattr(self.args, "perf_log", None) or "assets/perf_logs/perf_log.csv"
        perf_flush_every = int(getattr(self.args, "perf_log_flush_every", 64) or 64)
        frame_flush_every = int(getattr(self.args, "perf_frame_log_flush_every", 128) or 128)
        timeline_flush_every = int(getattr(self.args, "perf_timeline_flush_every", 128) or 128)
        perf_logger = PerfLogger(perf_log_path, flush_every=perf_flush_every)
        frame_log_path = getattr(self.args, "perf_log_frames", None) or "assets/perf_logs/perf_frames.csv"
        frame_logger = FramePerfLogger(frame_log_path, flush_every=frame_flush_every)
        timeline_path = getattr(self.args, "perf_timeline", None) or "assets/perf_logs/perf_timeline.jsonl"
        timeline_logger = TimelineLogger(timeline_path, flush_every=timeline_flush_every)

        gpu_index = int(getattr(self.args, "gpu_index", 0))
        gpu_mon = GPUMonitor(gpu_index=gpu_index)
        cpu_mon = CPUMonitor()
        infer_counter = SlidingCounter(window_s=1.0)
        batch_idx = 0

        model = kwargs["model"]
        producer_flag = kwargs["producer_flag"]
        preview_queue = kwargs["preview_queue"]
        profilers = kwargs["profilers"]
        activities = kwargs["activities"]
        start_time = kwargs["start_time"]

        if self.args.bench:
            self.mp = ModelPerf(
                class_ids=[i for i, _ in enumerate(self.converter.class_names)],
                iou_thresholds=np.arange(0.50, 0.96, 0.05),
                conf_threshold=CONF_THR,
                use_101_point_interp=True,
            )
        else:
            self.mp = None

        micro = BATCH_SIZE
        overlap_ratio = TILE_OVERLAP
        tile_size = TILE_SIZE

        self.run_callbacks("on_predict_start")
        with StepContext(name="Warmup Session", catch=(Exception, RuntimeError), verbose=self.args.verbose):
            if not self.model_warmup_done:
                self.model.warmup(micro=micro, warmup_sessions=WARM_UP_SESSIONS)
                self.model_warmup_done = True

        # Async batch-loading thread — mirrors _stream_inference_impl
        batch_queue = queue.Queue(maxsize=8)

        def _tile_producer():
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
            except Exception as exc:
                Streamer.logger.error("Error in tile-mode batch producer: %s", exc)
            finally:
                batch_queue.put(None)

        producer_thread = threading.Thread(target=_tile_producer, daemon=True)
        producer_thread.start()

        while True:
            t_batch_start = time.perf_counter()
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

            stage_a = self._stage_a_acquire_and_gate(
                batch_payload, frame_read_ms, stream_start, timeline_logger, batch_idx
            )

            if stage_a.get("skip_reason") == "warmup":
                continue

            paths = stage_a["paths"]
            im0s = stage_a["im0s"]
            frame_ids = stage_a["frame_ids"]
            original_images_bgr = stage_a["original_images_bgr"]
            display_images_bgr = stage_a.get("display_images_bgr", original_images_bgr)
            roi_ms = stage_a["roi_ms"]
            mog2_ms = stage_a["mog2_ms"]
            defish_ms = stage_a["defish_ms"]
            res_h = stage_a["res_h"]
            res_w = stage_a["res_w"]
            mfgs = stage_a["mfgs"]

            # ── No-motion batch: identical path to _stream_inference_impl ──────────
            if stage_a.get("skip_reason") == "no_motion":
                Streamer.logger.debug("No motion detected in tile-mode batch – skipping inference")
                # Advance the activation window frame counter even for skipped batches
                # so persistence ages out correctly with real elapsed time.
                _tile_win_nm = getattr(self, "_tile_activation", None)
                if _tile_win_nm is not None:
                    for _ in im0s:
                        _tile_win_nm.tick()
                empty_preds = return_no_motion_frames(im0s=im0s, batch_size=len(im0s))
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
                self._enqueue_async_sink(
                    ("mqtt_no_detection", empty_preds, frame_ids),
                    stage="mqtt_ms",
                    frame_ids=frame_ids,
                )
                self._publish_no_motion_preview(original_images_bgr, preview_queue, producer_flag)
                if self.args.save:
                    for i, fid in enumerate(frame_ids):
                        p = Path(paths[i])
                        save_path = str(self.save_dir / p.name)
                        frame_count = int(getattr(self.dataset, "count", fid))
                        orig_rgb = cv2.cvtColor(original_images_bgr[i], cv2.COLOR_BGR2RGB)
                        self.save_queue.put(("save_frame", save_path, frame_count, orig_rgb))
                if self.args.plot_performance:
                    t_now = time.perf_counter()
                    scores = getattr(self.logic_module.get("SUBTRACTOR", None), "last_motion_scores", None)
                    avg_motion_score = float(np.mean(scores)) if scores else 0.0
                    fps_sliding = fps
                    total_ms = (t_now - t_batch_start) * 1e3
                    infer_calls_per_sec = infer_counter.rate(t_now)
                    batch_stage_metrics = self._get_batch_stage_metrics()
                    perf_logger.log({
                        "t_wall": t_now, "batch_idx": batch_idx, "frames_in_batch": len(im0s),
                        "res_w": res_w, "res_h": res_h, "motion_density": 0.0,
                        "avg_motion_score": avg_motion_score, "inference_ran": 0,
                        "frames_inferred": 0, "infer_calls_per_sec": infer_calls_per_sec,
                        "gpu_util": gpu_stats.get("gpu_util", float("nan")),
                        "gpu_mem_used_mb": gpu_stats.get("mem_used_mb", float("nan")),
                        "gpu_mem_total_mb": gpu_stats.get("mem_total_mb", float("nan")),
                        "cpu_util": cpu_stats.get("cpu_util", float("nan")),
                        "frame_read_ms_per_frame": frame_read_ms / max(len(im0s), 1),
                        "roi_ms_per_frame": roi_ms / max(len(im0s), 1),
                        "mog2_ms_per_frame": mog2_ms / max(len(im0s), 1),
                        "defish_ms_per_frame": defish_ms / max(len(im0s), 1),
                        "preprocess_ms_per_frame": 0.0, "inference_ms_per_frame": 0.0,
                        "postprocess_ms_per_frame": 0.0, "nms_ms_per_frame": 0.0,
                        "tracking_ms_per_frame": 0.0, "hazard_logic_ms_per_frame": 0.0,
                        "preview_encode_ms_per_frame": batch_stage_metrics.get("preview_encode_ms", 0.0) / max(len(im0s), 1),
                        "mqtt_ms_per_frame": batch_stage_metrics.get("mqtt_ms", 0.0) / max(len(im0s), 1),
                        "event_saving_ms_per_frame": 0.0,
                        "total_ms_per_frame": total_ms / max(len(im0s), 1),
                        "fps_sliding": fps_sliding,
                        "tile_skip_rate": float("nan"),
                        "tiles_total_per_frame": 0.0,
                        "tiles_submitted_per_frame": 0.0,
                    })
                    timeline_logger.log_span(batch_idx, "batch_total", t_batch_start - stream_start, t_now - stream_start, {
                        "inference_ran": 0, "frames_in_batch": int(len(im0s)),
                    })
                    batch_idx += 1
                if getattr(self, "_tile_kalman", None) is not None:
                    _zero_b = np.zeros((0, 4), np.float32)
                    _zero_s = np.zeros((0,), np.float32)
                    _zero_c = np.zeros((0,), np.int64)
                    for _ in im0s:
                        self._tile_kalman.update(_zero_b, _zero_s, _zero_c)
                self.step_attention_state()
                continue

            # ── Blank no-motion frames before splitting into tiles ─────────────────
            for i, keep_frame in enumerate(mfgs):
                if not keep_frame:
                    im0s[i] = empty_image(im0s[i])
                    original_images_bgr[i] = empty_image(original_images_bgr[i])

            if self.args.plot_performance:
                _t0 = time.perf_counter()
                _t0_rel = _t0 - stream_start

            # ── Stage B (tiles): split → (gate) → microbatch inference → reconstruct ─
            all_tiles_with_meta: list = []
            overlap_px = max(0, int(round(tile_size * overlap_ratio)))
            fg_masks_b = stage_a.get("fg_masks") or []
            tile_win = getattr(self, "_tile_activation", None)
            per_frame_tile_metrics: dict = {}   # bni_t → {"total": int, "submitted": int}

            if tile_win is not None:
                tile_win.reset_batch_metrics()

            for bni_t, (keep_frame, img, fid) in enumerate(zip(mfgs, im0s, frame_ids)):
                if tile_win is not None:
                    tile_win.tick()   # advance once per video frame (including skipped)
                per_frame_tile_metrics[bni_t] = {"total": 0, "submitted": 0}
                if not keep_frame:
                    continue
                try:
                    fid_int = int(fid)
                except (TypeError, ValueError):
                    fid_int = hash(str(fid)) & 0x7FFFFFFF
                fg_mask_b = fg_masks_b[bni_t] if bni_t < len(fg_masks_b) else None
                frame_shape_b = img.shape[:2]
                for tile_img, meta in split_image_gen(img, fid_int, tile_size=tile_size, overlap=overlap_px):
                    tx_b, ty_b = meta["left_x"], meta["top_y"]
                    per_frame_tile_metrics[bni_t]["total"] += 1
                    if tile_win is not None and not tile_win.gate_tile(
                        fg_mask_b, frame_shape_b, tx_b, ty_b, tile_size
                    ):
                        continue    # cold tile — skip inference, Kalman fills any gap
                    per_frame_tile_metrics[bni_t]["submitted"] += 1
                    all_tiles_with_meta.append((tile_img, meta, bni_t))

            # Per-frame accumulator: keyed by the integer frame-id used in tile metas
            frame_tile_acc: dict = {}
            n_microbatches = 0

            for mb_start in range(0, max(1, len(all_tiles_with_meta)), micro):
                mb = all_tiles_with_meta[mb_start: mb_start + micro]
                if not mb:
                    break
                n_microbatches += 1

                tile_imgs_raw = [item[0] for item in mb]
                tile_metas = [item[1] for item in mb]

                t_pre0 = time.perf_counter()
                with profilers[0]:
                    tile_tensors = self.preprocess(tile_imgs_raw)
                preprocess_ms += (time.perf_counter() - t_pre0) * 1e3

                t_inf0 = time.perf_counter()
                with profilers[1]:
                    if self.seen == 0 and n_microbatches == 1 and self.args.verbose:
                        with profile(activities=activities) as prof:
                            raw_out = self.model(tile_tensors, orig_imgs=None, debug=True)
                        os.makedirs("assets/trace_jsons", exist_ok=True)
                        model_tag = getattr(self, "model_tag", type(self.model).__name__)
                        prof.export_chrome_trace(f"assets/trace_jsons/trace_{model_tag}_tiles.json")
                    else:
                        raw_out = self.model(tile_tensors, orig_imgs=None, debug=False)
                inference_ms += (time.perf_counter() - t_inf0) * 1e3

                if isinstance(raw_out, tuple) and len(raw_out) == 2 and isinstance(raw_out[0], tuple):
                    (i_boxes, i_scores, i_classes), event = raw_out
                else:
                    i_boxes, i_scores, i_classes = raw_out
                    event = None

                if event is not None and torch.cuda.is_available():
                    torch.cuda.current_stream().wait_event(event)

                for meta, boxes, scores, classes in zip(tile_metas, i_boxes, i_scores, i_classes):
                    fid_key = int(meta["frame_id"])
                    boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes, dtype=torch.float32)
                    if boxes_t.numel() == 0:
                        continue

                    scores_t = scores if torch.is_tensor(scores) else torch.as_tensor(scores, dtype=torch.float32)
                    classes_t = classes if torch.is_tensor(classes) else torch.as_tensor(classes, dtype=torch.int64)
                    scores_t = scores_t.to(dtype=torch.float32)
                    classes_t = classes_t.to(dtype=torch.int64)

                    road_mask = self._get_road_filter_mask(classes_t)
                    boxes_t = boxes_t[road_mask]
                    scores_t = scores_t[road_mask]
                    classes_t = classes_t[road_mask]
                    if boxes_t.numel() == 0:
                        continue

                    # Tile had road-scene detections → extend its persistence window
                    if tile_win is not None:
                        tile_win.mark_detection(meta["top_y"], meta["left_x"])

                    # Reconstruct tile-local boxes → frame-coordinate boxes (in-place)
                    boxes_np = boxes_t.float().cpu().numpy().copy()
                    boxes_np = reconstruct_tiles(
                        boxes_np,
                        tx=meta["left_x"],
                        ty=meta["top_y"],
                        orig_H=meta["f_wh"][0],
                        orig_W=meta["f_wh"][1],
                        gain=meta["gain"],
                        pad=(meta["pad_x"], meta["pad_y"]),
                    )

                    entry = frame_tile_acc.setdefault(fid_key, {"boxes": [], "scores": [], "classes": []})
                    entry["boxes"].append(boxes_np)
                    entry["scores"].append(
                        scores_t.float().cpu().numpy() if torch.is_tensor(scores_t) else np.asarray(scores_t, np.float32)
                    )
                    entry["classes"].append(
                        classes_t.cpu().numpy() if torch.is_tensor(classes_t) else np.asarray(classes_t, np.int64)
                    )

            if self.args.plot_performance:
                self._record_stage_time("preprocess_ms", preprocess_ms, frame_ids=frame_ids)
                self._record_stage_time("inference_ms", inference_ms, frame_ids=frame_ids)
                timeline_logger.log_span(
                    batch_idx, "preprocess+inference_tiles", _t0_rel, time.perf_counter() - stream_start
                )

            # Assemble DetectionBatch: global NMS per frame, optional ROI back-projection
            frames_dets: List[FrameDetections] = []
            for bni_a, (fid, keep_frame) in enumerate(zip(frame_ids, mfgs)):
                orig_img = display_images_bgr[bni_a]
                if self.args.save or self.args.show:
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                fg_mask_bni = fg_masks_b[bni_a] if bni_a < len(fg_masks_b) else None
                try:
                    fid_key = int(fid)
                except (TypeError, ValueError):
                    fid_key = hash(str(fid)) & 0x7FFFFFFF

                if not keep_frame:
                    frames_dets.append(FrameDetections.empty(frame_id=fid, batch_index=bni_a, orig_img=orig_img))
                    continue

                if fid_key not in frame_tile_acc:
                    fd = FrameDetections.empty(frame_id=fid, batch_index=bni_a, orig_img=orig_img)
                    frames_dets.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni_a].shape[:2]))
                    continue

                accum = frame_tile_acc[fid_key]
                if not accum["boxes"]:
                    fd = FrameDetections.empty(frame_id=fid, batch_index=bni_a, orig_img=orig_img)
                    frames_dets.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni_a].shape[:2]))
                    continue

                all_boxes_np = np.concatenate(accum["boxes"], axis=0).astype(np.float32)
                all_scores_np = np.concatenate(accum["scores"], axis=0).astype(np.float32)
                all_classes_np = np.concatenate(accum["classes"], axis=0).astype(np.int64)

                boxes_t_all = torch.as_tensor(all_boxes_np, dtype=torch.float32)
                scores_t_all = torch.as_tensor(all_scores_np, dtype=torch.float32)
                classes_t_all = torch.as_tensor(all_classes_np, dtype=torch.int64)

                t_nms0 = time.perf_counter()
                keep_idx = batched_nms(boxes_t_all, scores_t_all, classes_t_all.long(), iou_threshold=TILE_NMS_IOU)
                nms_elapsed = (time.perf_counter() - t_nms0) * 1e3
                nms_ms += nms_elapsed
                self._record_stage_time("nms_ms", nms_elapsed, frame_id=fid)

                boxes_t_all = boxes_t_all[keep_idx]
                scores_t_all = scores_t_all[keep_idx]
                classes_t_all = classes_t_all[keep_idx]

                # Map from im0s (possibly ROI-cropped) space back to original full-frame space
                if self.use_roi and boxes_t_all.numel() > 0:
                    boxes_t_all = self.logic_module["ROI"].translate_bounding_boxes(
                        results=boxes_t_all,
                        orig_img_shape=self.original_imgsz,
                    )

                if boxes_t_all.numel() == 0:
                    fd = FrameDetections.empty(frame_id=fid, batch_index=bni_a, orig_img=orig_img)
                    frames_dets.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni_a].shape[:2]))
                    continue

                fd = FrameDetections(
                    frame_id=fid,
                    batch_index=bni_a,
                    orig_img=orig_img,
                    boxes=boxes_t_all,
                    scores=scores_t_all,
                    classes=classes_t_all,
                )
                frames_dets.append(self._merge_motion_boxes(fd, fg_mask_bni, im0s[bni_a].shape[:2]))

            detection_batch = DetectionBatch(frames=frames_dets)
            # ── End Stage B (tiles) ───────────────────────────────────────────────

            # Kalman smoother: fills tile-boundary gaps before ByteTracker
            if getattr(self, "_tile_kalman", None) is not None:
                detection_batch = self._apply_kalman_smoother(detection_batch)

            postprocess_total_t0 = time.perf_counter()
            if self.args.plot_performance:
                _tpost = time.perf_counter()
                _tpost_rel = _tpost - stream_start

            frame_bundles, preds = self._stage_c_tracking_and_hazard_logic(
                detection_batch, display_images_bgr, profilers
            )
            frame_log_rows = self._stage_d_dispatch_optional_sinks(
                preds, frame_bundles, paths, display_images_bgr, preview_queue, producer_flag
            )

            per_frame_pre = preprocess_ms / max(n_microbatches, 1)
            per_frame_inf = inference_ms / max(n_microbatches, 1)

            for fb, r in zip(frame_bundles, preds):
                r.speed = {
                    "preprocess": per_frame_pre,
                    "inference": per_frame_inf,
                    "postprocess": profilers[2].dt * 1e3 / max(len(im0s), 1),
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
                            "[FPS][tiles] End-to-end: %.2f | Average: %.2f", fps, avg_fps
                        )
                        last_fps_log = now

            self.run_callbacks("on_predict_postprocess_end")

            if self.args.plot_performance:
                t_now = time.perf_counter()
                frames_in_batch = len(im0s)
                frames_inferred = int(sum(bool(x) for x in mfgs))
                motion_density = frames_inferred / frames_in_batch if frames_in_batch else 0.0
                scores_sub = getattr(self.logic_module.get("SUBTRACTOR", None), "last_motion_scores", None)
                avg_motion_score = float(np.mean(scores_sub)) if scores_sub else 0.0
                inference_ran = 1 if frames_inferred > 0 else 0
                if inference_ran:
                    infer_counter.add(t_now, 1.0)
                infer_calls_per_sec = infer_counter.rate(t_now)
                total_ms = (t_now - t_batch_start) * 1e3
                batch_stage_metrics = self._get_batch_stage_metrics()
                postprocess_ms = max(
                    0.0,
                    ((t_now - postprocess_total_t0) * 1e3)
                    - batch_stage_metrics.get("preview_encode_ms", 0.0)
                    - batch_stage_metrics.get("mqtt_ms", 0.0),
                )
                timeline_logger.log_span(
                    batch_idx, "postprocess", _tpost_rel, t_now - stream_start,
                    {"frame_in_batch": int(bni)},
                )
                per_read = frame_read_ms / max(frames_in_batch, 1)
                per_roi = roi_ms / max(frames_in_batch, 1)
                per_mog2 = mog2_ms / max(frames_in_batch, 1)
                per_defish = defish_ms / max(frames_in_batch, 1)
                per_pre = preprocess_ms / max(frames_in_batch, 1)
                per_inf = inference_ms / max(frames_in_batch, 1)
                perf_logger.log({
                    "t_wall": t_now, "batch_idx": batch_idx, "frames_in_batch": frames_in_batch,
                    "res_w": res_w, "res_h": res_h, "motion_density": motion_density,
                    "avg_motion_score": avg_motion_score, "inference_ran": inference_ran,
                    "frames_inferred": frames_inferred, "infer_calls_per_sec": infer_calls_per_sec,
                    "gpu_util": gpu_stats.get("gpu_util", float("nan")),
                    "gpu_mem_used_mb": gpu_stats.get("mem_used_mb", float("nan")),
                    "gpu_mem_total_mb": gpu_stats.get("mem_total_mb", float("nan")),
                    "cpu_util": cpu_stats.get("cpu_util", float("nan")),
                    "frame_read_ms_per_frame": per_read,
                    "roi_ms_per_frame": per_roi,
                    "mog2_ms_per_frame": per_mog2,
                    "defish_ms_per_frame": per_defish,
                    "preprocess_ms_per_frame": per_pre,
                    "inference_ms_per_frame": per_inf,
                    "postprocess_ms_per_frame": postprocess_ms / max(frames_in_batch, 1),
                    "nms_ms_per_frame": batch_stage_metrics.get("nms_ms", 0.0) / max(frames_in_batch, 1),
                    "tracking_ms_per_frame": batch_stage_metrics.get("tracking_ms", 0.0) / max(frames_in_batch, 1),
                    "hazard_logic_ms_per_frame": batch_stage_metrics.get("hazard_logic_ms", 0.0) / max(frames_in_batch, 1),
                    "preview_encode_ms_per_frame": batch_stage_metrics.get("preview_encode_ms", 0.0) / max(frames_in_batch, 1),
                    "mqtt_ms_per_frame": batch_stage_metrics.get("mqtt_ms", 0.0) / max(frames_in_batch, 1),
                    "event_saving_ms_per_frame": batch_stage_metrics.get("event_saving_ms", 0.0) / max(frames_in_batch, 1),
                    "total_ms_per_frame": total_ms / max(frames_in_batch, 1),
                    "fps_sliding": fps,
                    # ── Tile activation research metrics ──────────────────────────────
                    "tile_skip_rate": tile_win.batch_skip_rate if tile_win is not None else float("nan"),
                    "tiles_total_per_frame": (tile_win.batch_total / max(frames_in_batch, 1)) if tile_win is not None else float("nan"),
                    "tiles_submitted_per_frame": (tile_win.batch_submitted / max(frames_in_batch, 1)) if tile_win is not None else float("nan"),
                })
                # Periodic console summary so the skip rate is visible without CSV post-processing
                if tile_win is not None and (batch_idx % 100 == 0 or batch_idx == 0):
                    Streamer.logger.info(
                        "[tiles][gate] batch %d — skip %.1f%% (%d/%d tiles submitted) | "
                        "cumulative skip %.1f%%",
                        batch_idx,
                        tile_win.batch_skip_rate * 100,
                        tile_win.batch_submitted, tile_win.batch_total,
                        tile_win.cumulative_skip_rate * 100,
                    )
                scores_list = getattr(self.logic_module.get("SUBTRACTOR", None), "last_motion_scores", None) or []
                for row in frame_log_rows:
                    fid_row = row["frame_id"]
                    bni_row = row["bni"]
                    frame_stage_metrics = self._get_frame_stage_metrics(fid_row)
                    motion_score = float(scores_list[bni_row]) if bni_row < len(scores_list) else float("nan")
                    _fm = per_frame_tile_metrics.get(bni_row, {"total": 0, "submitted": 0})
                    _fm_total = _fm["total"]
                    _fm_skip = 1.0 - _fm["submitted"] / max(_fm_total, 1) if _fm_total > 0 else float("nan")
                    frame_logger.log({
                        "t_wall": t_now, "batch_idx": batch_idx,
                        "frame_id": int(fid_row) if str(fid_row).isdigit() else fid_row,
                        "res_w": res_w, "res_h": res_h,
                        "motion_passed": row["motion_passed"],
                        "motion_score": motion_score,
                        "gpu_util": gpu_stats.get("gpu_util", float("nan")),
                        "gpu_mem_used_mb": gpu_stats.get("mem_used_mb", float("nan")),
                        "cpu_util": cpu_stats.get("cpu_util", float("nan")),
                        "frame_read_ms": per_read, "roi_ms": per_roi,
                        "mog2_ms": per_mog2, "defish_ms": per_defish,
                        "preprocess_ms": per_pre, "inference_ms": per_inf,
                        "postprocess_ms": postprocess_ms / max(frames_in_batch, 1),
                        "nms_ms": frame_stage_metrics.get("nms_ms", 0.0),
                        "tracking_ms": frame_stage_metrics.get("tracking_ms", 0.0),
                        "hazard_logic_ms": frame_stage_metrics.get("hazard_logic_ms", 0.0),
                        "preview_encode_ms": frame_stage_metrics.get("preview_encode_ms", 0.0),
                        "mqtt_ms": frame_stage_metrics.get("mqtt_ms", 0.0),
                        "event_saving_ms": frame_stage_metrics.get("event_saving_ms", 0.0),
                        "total_ms": (
                            per_read + per_roi + per_mog2 + per_defish + per_pre + per_inf
                            + (postprocess_ms / max(frames_in_batch, 1))
                            + frame_stage_metrics.get("preview_encode_ms", 0.0)
                            + frame_stage_metrics.get("mqtt_ms", 0.0)
                        ),
                        # ── per-frame tile research columns ──────────────────────────
                        "tiles_total": _fm_total,
                        "tiles_submitted": _fm["submitted"],
                        "tile_skip_rate": _fm_skip,
                    })
                timeline_logger.log_span(
                    batch_idx, "batch_total", t_batch_start - stream_start, t_now - stream_start,
                    {"inference_ran": int(inference_ran), "frames_in_batch": int(frames_in_batch)},
                )
                batch_idx += 1

            self.run_callbacks("on_predict_batch_end")
            self.step_attention_state()

        if self.stop_reason != "stream_limit":
            producer_thread.join()

        if self.args.bench and self.mp is not None:
            self.mp.finalize()
            Streamer.logger.info(self.mp.results())

        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""

        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)
            Streamer.logger.info(
                "[tiles] Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, tile_size, tile_size)}" % t
            )

        if (self.args.only_FPS and not self.args.plot_performance) or self.args.plot_performance:
            total_time = time.perf_counter() - stream_start
            if total_frames > 0:
                print(f"[FPS][tiles] FINAL Average FPS: {total_frames / total_time:.2f}")

        # Tile activation cumulative summary — useful for research / benchmarking
        _tile_win_final = getattr(self, "_tile_activation", None)
        if _tile_win_final is not None and _tile_win_final.cumulative_total > 0:
            cum = _tile_win_final.get_cumulative_metrics()
            Streamer.logger.info(
                "[tiles][gate] STREAM TOTAL — %.1f%% tiles skipped  "
                "(%d submitted / %d total, %d skipped)",
                cum["cumulative_tile_skip_rate"] * 100,
                cum["cumulative_tiles_submitted"],
                cum["cumulative_tiles_total"],
                cum["cumulative_tiles_skipped"],
            )

        if self.args.plot_performance:
            try:
                perf_logger.close()
                frame_logger.close()
                timeline_logger.close()
            except Exception:
                pass

        self.release_session_resources(preview_queue=preview_queue, producer_flag=producer_flag)
        self.run_callbacks("on_predict_end")




    def _publish_mqtt_message(self, preds, mqtt_messages, frame_ids)->None: 
        super()._publish_mqtt_message(preds, mqtt_messages, frame_ids)


    def _publish_mqtt_message_no_detection(self, preds, frame_index)->None: 
        super()._publish_mqtt_message_no_detection(preds, frame_index)
