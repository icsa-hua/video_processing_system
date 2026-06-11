from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
from obs_system.utils import global_config
from obs_system.utils.global_config import (
    TRIALS,
    HISTORY,
    THR_RATIO,
    K_CONSECUTIVE,
    HOLD_FRAMES,
    MIN_OBJ_AREA,
    MIN_MOTION_COMPONENT_AREA_RATIO,
    MOTION_MORPH_KERNEL,
    ENABLE_UNSTABLE_MOTION_MAP,
    UNSTABLE_MOTION_THRESHOLD,
    UNSTABLE_MOTION_SUPPRESSION_WEIGHT,
    ENABLE_DRIVABLE_CONFIDENCE_MAP,
    DRIVABLE_STATIC_WEIGHT,
    DRIVABLE_DETECTION_WEIGHT,
    DRIVABLE_TRACK_WEIGHT,
    DRIVABLE_UNSTABLE_NEGATIVE_WEIGHT,
)
from obs_system.utils.logger import get_logger
from obs_system.utils.common import detect_static_lanes

import numpy as np
import cv2
import os

from collections import deque
from pathlib import Path
from typing import Optional, Union

logger = get_logger("obs_system"+__name__)

class Subtractor(EventExtractorInterface): 
    def __init__(self,
                 trials=TRIALS,
                 history=HISTORY,
                 threshold_ratio=THR_RATIO,
                 detect_shadows=True,
                 empty_background_image="",
                 downscale=(320,320),
                 accum_time:int=100,
                 save_path:str="assets/background_check/lane_image.jpg",
                 recalibration_interval_frames:int=0,
                 recalibration_accum_time:Optional[int]=None,
                 save_scene_overlays: bool = True,
    ):
        self.downscale = downscale
        self.threshold_ratio = float(threshold_ratio)
        self.initial_accum_time = max(int(accum_time), 0)
        self.accum_time = self.initial_accum_time
        self.save_path = save_path
        self.recalibration_interval_frames = max(int(recalibration_interval_frames), 0)
        self.recalibration_accum_time = max(
            int(recalibration_accum_time if recalibration_accum_time is not None else self.initial_accum_time),
            0,
        )
        # self.save_scene_overlays = bool(save_scene_overlays)
        self.save_scene_overlays = False

        motion_config = global_config.get_active_motion_config()
        self.var_threshold = int(motion_config["var_threshold"])
        self.motion_gate_filtered_score_threshold = float(motion_config["gate_score_threshold"])
        self.max_motion_components = int(motion_config["max_components"])
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=history, 
                    varThreshold=self.var_threshold,
                    detectShadows=detect_shadows)


        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self.kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.kernel15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))

        self.static_bg = False 

        self.acc_mask = None
        self.prev_mask = None

        #Hysteresis 
        self.hold_frames = HOLD_FRAMES # Number of allowed frames to have movement.  
        self._recent = deque(maxlen=K_CONSECUTIVE)
        self._hold = 0 

        self.__calibration_started = False
        self.__calibration_ended = False

        # Stores per-frame motion scores (foreground ratio) from the last detect() call
        self.last_motion_scores = []  # list[float], same length as batch
        self.lanes_mask: Optional[np.ndarray] = None
        self.crosswalk_mask: Optional[np.ndarray] = None
        self._last_frame_shape: Optional[tuple[int, int]] = None
        self._source_is_stream = False
        self._startup_warmup_active = False
        self._startup_frames_seen = 0
        self._ready_for_inference = True
        self._recalibration_enabled = False
        self._recalibration_active = False
        self._frames_since_last_calibration = 0
        self._saved_lane_extractions = 0
        self._saved_crosswalk_extractions = 0
        self._max_saved_scene_extractions = 2

        # Cached per-resolution constants (set on first processed frame, avoids
        # recomputing total pixels and threshold every call)
        self._cached_total_pixels: float = 0.0
        self._cached_threshold: int = 0

        # Raw MOG2 output stored for the calibration accumulator so it can reuse
        # it without running a second background subtractor
        self._last_fg_mask: Optional[np.ndarray] = None
        self._last_motion_score: float = 0.0

        # Static lane detection — primary path for sparse-traffic environments
        self._static_lanes_mask: Optional[np.ndarray] = None
        self._static_bg_sampled: bool = False
        self._static_bg_warmup_frames: int = 50   # frames to stabilise MOG2 background

        # Rolling motion-density window for hybrid branch decision
        self._motion_score_window: deque = deque(maxlen=60)
        self._frames_processed: int = 0
        self._motion_sparse_threshold: float = 0.001  # below → static mask is primary
        self._motion_dense_threshold: float = 0.006   # above → prefer motion-derived lanes
        self._static_lane_max_fill_ratio: float = 0.65

        # YOLO-gated calibration ───────────────────────────────────────────────
        # The calibration accumulator is now driven by vehicle-confirmed frames
        # (notify_vehicle_detections) rather than raw MOG2 output.  This counter
        # tracks how many such frames have contributed to the current window.
        self._vehicle_confirmed_frames: int = 0
        # Minimum vehicle-confirmed frames before the dynamic candidate is trusted.
        # Below this the window closed without enough real vehicle evidence and the
        # static baseline is kept instead.
        self._min_vehicle_frames_for_calibration: int = max(10, self.initial_accum_time // 5)
        # Last frame seen by detect(); used as the reference frame for crosswalk
        # detection when __apply_calibration is triggered from notify_vehicle_detections.
        self._last_calibration_frame: Optional[np.ndarray] = None

        # ── Step 1: motion-quality filtering ──────────────────────────────────
        # Elliptic kernel used for the motion-gate morphology pass (larger than
        # kernel3 to better connect coherent blobs before CC analysis).
        self._motion_morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MOTION_MORPH_KERNEL, MOTION_MORPH_KERNEL)
        )

        # ── Step 2: unstable-motion map ───────────────────────────────────────
        # Accumulates raw MOG2 binary output during the calibration window.
        # Finalised into _unstable_motion_map once calibration ends.
        self._unstable_motion_accumulator: Optional[np.ndarray] = None
        self._unstable_motion_frame_count: int = 0
        # float32 at downscale resolution; high value = background that moves often
        self._unstable_motion_map: Optional[np.ndarray] = None

        # ── Step 3: drivable-area confidence map ─────────────────────────────
        self._detection_heatmap: Optional[np.ndarray] = None   # vehicle bbox paint
        self._track_heatmap: Optional[np.ndarray] = None       # track trail paint
        self._drivable_confidence_map: Optional[np.ndarray] = None
        self._drivable_rebuild_counter: int = 0
        self._drivable_rebuild_interval: int = 5   # rebuild combined map every N updates
        self._dc_call_count: int = 0               # used to throttle heatmap decay
        logger.info(
            "Active motion gating: var_threshold=%d gate_score_threshold=%.6f max_components=%d",
            self.var_threshold,
            self.motion_gate_filtered_score_threshold,
            self.max_motion_components,
        )


    def configure_source_warmup(self, source_is_stream: bool) -> None:
        self._source_is_stream = bool(source_is_stream)
        self._startup_frames_seen = 0
        self._startup_warmup_active = bool(
            self._source_is_stream
            and not self.static_bg
            and not self.__calibration_ended
            and self.initial_accum_time > 0
        )
        self._ready_for_inference = not self._startup_warmup_active

        if self._startup_warmup_active:
            logger.info(
                "Subtractor startup warmup enabled for live stream: %d frames before inference",
                self.initial_accum_time,
            )


    def configure_runtime_recalibration(
        self,
        *,
        enabled: bool,
        interval_frames: Optional[int] = None,
        accum_time: Optional[int] = None,
    ) -> None:
        if interval_frames is not None:
            self.recalibration_interval_frames = max(int(interval_frames), 0)
        if accum_time is not None:
            self.recalibration_accum_time = max(int(accum_time), 0)

        self._recalibration_enabled = bool(enabled and self.recalibration_interval_frames > 0)
        self._frames_since_last_calibration = 0

        if not self._recalibration_enabled:
            self._recalibration_active = False


    def _reset_calibration_buffers(self) -> None:
        self.acc_mask = None
        self.prev_mask = None
        self.__calibration_started = False
        self.__calibration_ended = False
        self._vehicle_confirmed_frames = 0


    def _has_nonempty_mask(self, mask: Optional[np.ndarray]) -> bool:
        return bool(mask is not None and mask.size > 0 and cv2.countNonZero(mask) > 0)

    # ── Step 1 helpers ────────────────────────────────────────────────────────

    def _filter_motion_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """Remove fragmented/thin blobs (vegetation, shadow flicker) from a binary motion mask.

        Keeps only components that are:
        - Large enough (area > MIN_MOTION_COMPONENT_AREA_RATIO × frame area)
        - Sufficiently compact (fill ratio inside bounding box > 0.12)

        If the number of remaining components exceeds MAX_MOTION_COMPONENTS the
        whole mask is blanked — too many surviving blobs indicates scattered
        vegetation-type motion, not coherent object motion.
        """
        if self._cached_total_pixels == 0 or cv2.countNonZero(binary_mask) == 0:
            return binary_mask

        min_area = MIN_MOTION_COMPONENT_AREA_RATIO * self._cached_total_pixels
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        filtered = np.zeros_like(binary_mask)
        n_valid = 0

        for i in range(1, n_labels):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            bw = float(max(stats[i, cv2.CC_STAT_WIDTH], 1))
            bh = float(max(stats[i, cv2.CC_STAT_HEIGHT], 1))
            # Discard needle-thin or highly scattered blobs typical of leaf motion
            if area / (bw * bh) < 0.12:
                continue
            filtered[labels == i] = 255
            n_valid += 1

        # Many small-but-passing blobs = vegetation scatter field → suppress all
        if n_valid > self.max_motion_components:
            return np.zeros_like(binary_mask)

        return filtered

    def _compute_weighted_motion_score(self, filtered_mask: np.ndarray) -> float:
        """Compute a motion score from the CC-filtered mask, downweighting pixels
        that fall inside the unstable-motion map (Step 2).

        Returns a value in [0, 1] (fraction of frame area, after weighting).
        """
        if self._cached_total_pixels == 0:
            return 0.0

        if not ENABLE_UNSTABLE_MOTION_MAP or self._unstable_motion_map is None:
            return float(cv2.countNonZero(filtered_mask)) / self._cached_total_pixels

        unstable = self._unstable_motion_map
        if unstable.shape != filtered_mask.shape:
            unstable = cv2.resize(
                unstable,
                (filtered_mask.shape[1], filtered_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        fg = (filtered_mask > 0).astype(np.float32)
        # Stable pixels → weight 1.0; unstable pixels → weight SUPPRESSION_WEIGHT
        weight_map = 1.0 - (1.0 - UNSTABLE_MOTION_SUPPRESSION_WEIGHT) * unstable
        return float(np.sum(fg * weight_map)) / self._cached_total_pixels

    # ── Step 3 helpers ────────────────────────────────────────────────────────

    def update_drivable_confidence(
        self,
        boxes_xyxy,
        classes,
        class_names: list,
        track_points: Optional[dict] = None,
        orig_hw: Optional[tuple] = None,
    ) -> None:
        """Paint vehicle detections and track trails into heatmaps, then
        rebuild the drivable-area confidence map every _drivable_rebuild_interval
        calls.

        boxes_xyxy : Tensor or ndarray (N×4) in original-image coordinates.
        classes    : Tensor or ndarray (N,) of integer class ids.
        orig_hw    : (H, W) of the original frame, used to scale boxes to
                     downscale resolution.  Falls back to _last_frame_shape.
        """
        if not ENABLE_DRIVABLE_CONFIDENCE_MAP:
            return

        self._dc_call_count += 1
        target_h, target_w = self.downscale

        # ── initialise heatmaps on first call ────────────────────────────────
        if self._detection_heatmap is None:
            self._detection_heatmap = np.zeros((target_h, target_w), np.float32)
        if self._track_heatmap is None:
            self._track_heatmap = np.zeros((target_h, target_w), np.float32)

        # ── slow heatmap decay (once per ~16 calls ≈ one batch) ──────────────
        if self._dc_call_count % 16 == 0:
            self._detection_heatmap *= 0.97
            self._track_heatmap *= 0.97

        # ── coordinate scale from original frame to downscale ─────────────────
        frame_h, frame_w = orig_hw if orig_hw is not None else (
            self._last_frame_shape if self._last_frame_shape is not None else self.downscale
        )
        sx = target_w / max(float(frame_w), 1.0)
        sy = target_h / max(float(frame_h), 1.0)

        # ── paint vehicle detection bboxes ───────────────────────────────────
        _vehicle_names = {"car", "truck", "bus", "bike", "bicycle", "motorbike", "motorcycle"}
        if boxes_xyxy is not None:
            boxes_np = (
                boxes_xyxy.detach().cpu().numpy()
                if hasattr(boxes_xyxy, "detach")
                else np.asarray(boxes_xyxy, dtype=np.float32)
            )
            classes_np = (
                classes.detach().cpu().numpy()
                if hasattr(classes, "detach")
                else np.asarray(classes, dtype=np.int64)
            ) if classes is not None else np.zeros(len(boxes_np), dtype=np.int64)

            for box, cls_id in zip(boxes_np, classes_np):
                cls_name = class_names[int(cls_id)] if 0 <= int(cls_id) < len(class_names) else ""
                if cls_name.lower() not in _vehicle_names:
                    continue
                x1 = int(np.clip(box[0] * sx, 0, target_w - 1))
                y1 = int(np.clip(box[1] * sy, 0, target_h - 1))
                x2 = int(np.clip(box[2] * sx, 0, target_w))
                y2 = int(np.clip(box[3] * sy, 0, target_h))
                if x2 > x1 and y2 > y1:
                    self._detection_heatmap[y1:y2, x1:x2] = np.minimum(
                        self._detection_heatmap[y1:y2, x1:x2] + 0.5, 20.0
                    )

        # ── paint track centroid trails ───────────────────────────────────────
        if track_points:
            for pts in track_points.values():
                if not isinstance(pts, np.ndarray) or pts.ndim < 2:
                    continue
                for pt in pts:
                    if len(pt) >= 2:
                        px = int(np.clip(float(pt[0]) * sx, 0, target_w - 1))
                        py = int(np.clip(float(pt[1]) * sy, 0, target_h - 1))
                        self._track_heatmap[py, px] = min(
                            self._track_heatmap[py, px] + 0.3, 20.0
                        )

        # ── rebuild combined confidence map on schedule ───────────────────────
        self._drivable_rebuild_counter += 1
        if self._drivable_rebuild_counter >= self._drivable_rebuild_interval:
            self._drivable_rebuild_counter = 0
            self._rebuild_drivable_confidence()

    def _rebuild_drivable_confidence(self) -> None:
        """Combine all signals into a single float32 confidence map [0, 1]."""
        if not ENABLE_DRIVABLE_CONFIDENCE_MAP:
            return

        target_h, target_w = self.downscale
        conf = np.zeros((target_h, target_w), np.float32)

        # 1. Static lane mask (strongest prior when available)
        lane_src = self.lanes_mask if self.lanes_mask is not None else self._static_lanes_mask
        if lane_src is not None:
            lane = lane_src
            if lane.shape[:2] != (target_h, target_w):
                lane = cv2.resize(lane, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            conf += DRIVABLE_STATIC_WEIGHT * (lane > 0).astype(np.float32)

        # 2. Vehicle detection heatmap
        if self._detection_heatmap is not None:
            det = np.clip(self._detection_heatmap, 0.0, 20.0) / 20.0
            det_blurred = cv2.GaussianBlur(det, (15, 15), 0)
            conf += DRIVABLE_DETECTION_WEIGHT * det_blurred

        # 3. Track trail heatmap
        if self._track_heatmap is not None:
            trk = np.clip(self._track_heatmap, 0.0, 20.0) / 20.0
            trk_blurred = cv2.GaussianBlur(trk, (11, 11), 0)
            conf += DRIVABLE_TRACK_WEIGHT * trk_blurred

        # 4. Negative: subtract unstable-motion regions (background clutter)
        if self._unstable_motion_map is not None:
            unstable = self._unstable_motion_map
            if unstable.shape[:2] != (target_h, target_w):
                unstable = cv2.resize(
                    unstable, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )
            conf -= DRIVABLE_UNSTABLE_NEGATIVE_WEIGHT * unstable

        self._drivable_confidence_map = np.clip(conf, 0.0, 1.0)

    def get_drivable_confidence_map(
        self, target_hw: Optional[tuple] = None
    ) -> Optional[np.ndarray]:
        """Return the drivable-area confidence map at an optional target resolution.

        Returns None if the map has not yet been built (calibration not finished
        and no detections received).
        """
        if not ENABLE_DRIVABLE_CONFIDENCE_MAP or self._drivable_confidence_map is None:
            return None
        if target_hw is None:
            return self._drivable_confidence_map
        th, tw = target_hw
        if self._drivable_confidence_map.shape[:2] == (th, tw):
            return self._drivable_confidence_map
        return cv2.resize(
            self._drivable_confidence_map, (tw, th), interpolation=cv2.INTER_LINEAR
        )

    def _begin_runtime_recalibration(self) -> None:
        if not self._recalibration_enabled or self.recalibration_accum_time <= 0:
            return

        self._run_static_lane_detection()  # refresh static baseline before accumulating
        self.accum_time = self.recalibration_accum_time
        self._recalibration_active = True
        self._frames_since_last_calibration = 0
        self._reset_calibration_buffers()
        logger.info(
            "Starting runtime lane recalibration for %d frames",
            self.recalibration_accum_time,
        )


    def notify_vehicle_detections(self, confirmed_batch_indices: list) -> None:
        """YOLO-gated lane calibration accumulator.

        Called from Stage C after YOLO inference completes for a batch.
        Only the fg_masks of frames where YOLO confirmed at least one vehicle
        (score >= CONF_THR) are accumulated — MOG2 false positives from
        trees, shadows, and lighting changes that would otherwise contaminate
        the lane candidate are silently skipped.

        Calibration completes once ``accum_time`` vehicle-confirmed frames
        have been accumulated.  If fewer than
        ``_min_vehicle_frames_for_calibration`` frames were confirmed before
        the window closed, the dynamic result is discarded and the static
        baseline is kept.
        """
        if self.__calibration_ended or self.accum_time <= 0:
            return
        if not self.__calibration_started or not confirmed_batch_indices:
            return

        h, w = self._last_frame_shape if self._last_frame_shape is not None else self.downscale

        for bni in confirmed_batch_indices:
            fg_masks = getattr(self, "_last_batch_fg_masks", [])
            if bni >= len(fg_masks):
                continue
            fg_mask = fg_masks[bni]
            if fg_mask is None:
                continue

            # fg_mask is already thresholded at 127 (shadows included); apply
            # the same CC quality filter used by the motion gate so large
            # connected vegetation blobs don't survive via vehicle confirmation.
            fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel3)
            if self._cached_total_pixels > 0:
                fg_clean = self._filter_motion_components(fg_clean)

            if ENABLE_UNSTABLE_MOTION_MAP:
                fg_bin = (fg_clean > 0).astype(np.float32)
                if self._unstable_motion_accumulator is None:
                    self._unstable_motion_accumulator = np.zeros_like(fg_bin, dtype=np.float32)
                    self._unstable_motion_frame_count = 0
                self._unstable_motion_accumulator += fg_bin
                self._unstable_motion_frame_count += 1

            mask_resized = cv2.resize(fg_clean.astype(np.float32), (w, h))
            blended = cv2.addWeighted(mask_resized, 0.35, self.prev_mask, 0.65, 0)
            self.prev_mask = blended
            self.acc_mask = cv2.add(self.acc_mask, blended)

            self._vehicle_confirmed_frames += 1
            self.accum_time -= 1

            if self.accum_time <= 0:
                self.__calibration_ended = True
                break

        if not self.__calibration_ended:
            return

        # ── calibration window closed ─────────────────────────────────────────
        if self._vehicle_confirmed_frames < self._min_vehicle_frames_for_calibration:
            logger.info(
                "YOLO-gated calibration: only %d/%d vehicle-confirmed frames; "
                "keeping static baseline",
                self._vehicle_confirmed_frames,
                self._min_vehicle_frames_for_calibration,
            )
            if self.lanes_mask is None and self._static_lanes_mask is not None:
                self.lanes_mask = self._static_lanes_mask.copy()
        else:
            logger.info(
                "YOLO-gated calibration complete: %d vehicle-confirmed frames accumulated",
                self._vehicle_confirmed_frames,
            )
            if self._last_calibration_frame is not None:
                self.__apply_calibration(self._last_calibration_frame, save_img=True)

        self.accum_time = -1
        self._startup_warmup_active = False
        self._ready_for_inference = True
        self._recalibration_active = False
        self._frames_since_last_calibration = 0
        self._recent.clear()
        self._hold = 0

    def is_ready_for_inference(self) -> bool:
        return bool(self._ready_for_inference)


    def get_startup_warmup_progress(self) -> tuple[int, int]:
        return int(self._startup_frames_seen), int(self.initial_accum_time)


    def warm_up(self, empty_background_image:Optional[np.ndarray], trials:int=TRIALS):
        if empty_background_image is None:
            raise ValueError("Empty background image is required for warm_up of Subtractor.")

        if isinstance(empty_background_image, str): 
            logger.debug("Empty Background training for subtractor")
            empty_bg = cv2.imread(empty_background_image)

        elif isinstance(empty_background_image, np.ndarray):
            empty_bg = empty_background_image    

        else: 
            raise ValueError("Invalid type for empty_background_image in warm_up of Subtractor.")
        
        if empty_bg is not None: 

            if self.downscale: 
                empty_bg = cv2.resize(empty_bg, self.downscale, interpolation=cv2.INTER_AREA)

            for _ in range(trials): 
                self.bg_subtractor.apply(empty_bg, learningRate=1.0) 

            self.static_bg = True #Shows that we entered the first time. 
            self._startup_warmup_active = False
            self._ready_for_inference = True


    def detect(self, batch, save_img:bool=True): 

        if not batch: return []

        save_dir = None
        save_idx = None
        if save_img: 
            parent = os.getcwd()
            save_dir = f"{parent}/assets/jetson_background_check/"
            os.makedirs(save_dir, exist_ok=True)
            logger.debug(f"Background Images saved in {save_dir}")
            save_idx = 0 
        
        
        h, w = batch[0].shape[:2]
        self._last_frame_shape = (h, w)

        if (
            self._recalibration_enabled
            and not self._recalibration_active
            and self.__calibration_ended
            and self.recalibration_interval_frames > 0
            and self._frames_since_last_calibration >= self.recalibration_interval_frames
        ):
            self._begin_runtime_recalibration()

        if not self.__calibration_started: 
            self.acc_mask = np.zeros((h, w), np.float32) 
            self.prev_mask = np.zeros((h,w), np.float32)
            self.__calibration_started = True 

        motion_flags = []
        motion_scores = []
        self._last_batch_fg_masks: list = []  # binary downscale masks, one per frame
        lanes_final = None
        last_frame = batch[-1]
        self._last_calibration_frame = last_frame  # used by notify_vehicle_detections
        startup_skip_batch = self._startup_warmup_active and not self._ready_for_inference

        for frame in batch:

            if self.downscale: 
                frame = cv2.resize(frame, self.downscale, interpolation=cv2.INTER_AREA)

            motion_flag = self.__call_subtractor(frame, save_dir=save_dir, save_img=save_img, save_idx=save_idx)

            motion_flags.append(motion_flag)
            motion_scores.append(self._last_motion_score)
            if self._last_fg_mask is not None:
                _, _bin = cv2.threshold(self._last_fg_mask, 127, 255, cv2.THRESH_BINARY)
                self._last_batch_fg_masks.append(_bin.copy())
            else:
                self._last_batch_fg_masks.append(None)

            if save_img and save_idx is not None: 
                save_idx += 1 

            # Accumulation is YOLO-gated: only frames where YOLO confirms a
            # vehicle contribute to the calibration window, via
            # notify_vehicle_detections() called from Stage C.
            # Exception: live-stream startup warmup where YOLO has not yet run
            # still uses the old MOG2 path so the subtractor has an initial
            # baseline before inference begins.
            if startup_skip_batch and self.accum_time > 0:
                self.__cal_calibrator(frame, size=(h,w))
                self._startup_frames_seen = min(self.initial_accum_time, self._startup_frames_seen + 1)

        # Update rolling density window and trigger one-shot static detection
        # once the background model has had enough frames to stabilise
        self._motion_score_window.extend(motion_scores)
        self._frames_processed += len(batch)
        if not self._static_bg_sampled and self._frames_processed >= self._static_bg_warmup_frames:
            self._run_static_lane_detection()
            self._static_bg_sampled = True

        if self.accum_time == 0 and self.__calibration_ended :
            lanes_final = self.__apply_calibration(last_frame, save_img=save_img)
            self.accum_time = -1
            self._startup_warmup_active = False
            self._ready_for_inference = True
            self._recalibration_active = False
            self._frames_since_last_calibration = 0
            self._recent.clear()
            self._hold = 0
        elif (
            self._recalibration_enabled
            and self.__calibration_ended
            and not self._recalibration_active
            and not startup_skip_batch
        ):
            self._frames_since_last_calibration += len(batch)

        if startup_skip_batch:
            motion_flags = [False] * len(motion_flags)

        self.last_motion_scores = motion_scores
        return motion_flags, lanes_final
        

    def __call_subtractor(self, frame, save_dir=None, save_img=False, save_idx:int=0):

        lr = 0.0 if self.static_bg else -1
        raw = self.bg_subtractor.apply(frame, learningRate=lr)
        self._last_fg_mask = raw  # stored for calibration accumulator (uses lower threshold)

        _, subtractor_mask = cv2.threshold(raw, 254, 255, cv2.THRESH_BINARY)

        # Cache resolution-dependent constants on first call
        if self._cached_total_pixels == 0:
            dh, dw = subtractor_mask.shape[:2]
            self._cached_total_pixels = float(dh * dw)
            self._cached_threshold = int(self.threshold_ratio * dh * dw)

        # Raw pixel count used in the hold-active fast path (cheap, no filtering)
        raw_motion_pixels = cv2.countNonZero(subtractor_mask)
        self._last_motion_score = raw_motion_pixels / self._cached_total_pixels

        # Early exit: hysteresis hold is active — skip morph + CC work
        if self._hold > 0:
            self._hold -= 1
            self._recent.append(raw_motion_pixels > self._cached_threshold)
            if save_dir is not None and save_img and save_idx is not None:
                s = cv2.morphologyEx(subtractor_mask, cv2.MORPH_OPEN, self.kernel3)
                s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, self.kernel3)
                self.__save_subtractor(frame, s, save_dir, save_idx)
            return True

        # ── Step 1: morphology cleanup with the configurable kernel ──────────
        subtractor_mask = cv2.morphologyEx(subtractor_mask, cv2.MORPH_OPEN, self._motion_morph_kernel)
        subtractor_mask = cv2.morphologyEx(subtractor_mask, cv2.MORPH_CLOSE, self._motion_morph_kernel)

        # ── Step 1: connected-component quality filter ────────────────────────
        # Removes small/thin/fragmented blobs that resemble vegetation noise.
        filtered_mask = self._filter_motion_components(subtractor_mask)

        # ── Step 2: weighted motion score (unstable-region suppression) ───────
        filtered_score = self._compute_weighted_motion_score(filtered_mask)
        self._last_motion_score = filtered_score

        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        flag = False
        if contours:
            max_obj_area = max((cv2.contourArea(c) for c in contours), default=0)
            min_obj_area = MIN_OBJ_AREA * self._cached_total_pixels
            flag = (
                filtered_score > self.motion_gate_filtered_score_threshold
                and max_obj_area > min_obj_area
            )

        self._recent.append(flag)
        if len(self._recent) == self._recent.maxlen and all(self._recent):
            self._hold = self.hold_frames

        if save_dir is not None and save_img and save_idx is not None:
            self.__save_subtractor(frame, filtered_mask, save_dir, save_idx)

        return flag

            
    def __save_subtractor(self, frame, mask, save_dir:str, idx:int): 
        motion_cutout = cv2.bitwise_and(frame, frame, mask=mask) 
        cv2.imwrite(os.path.join(save_dir, f"{idx:06d}_motion.png"), motion_cutout) 


    def __cal_calibrator(self, frame, **kwargs):

        if self.__calibration_ended:
            return

        if self._last_fg_mask is None:
            return

        h, w = kwargs["size"]

        # Reuse the raw MOG2 output already computed in __call_subtractor.
        # Threshold at 127 to include shadow regions, which also mark vehicle paths,
        # giving denser accumulation than the motion-gate mask (threshold 254).
        _, fg = cv2.threshold(self._last_fg_mask, 127, 255, cv2.THRESH_BINARY)
        fg_clean = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel3)
        # Remove scattered/thin vegetation blobs before accumulating — same CC quality
        # filter used by the motion gate, applied here so the accumulator isn't
        # contaminated by tree motion that would otherwise inflate candidate_lanes.
        if self._cached_total_pixels > 0:
            fg_clean = self._filter_motion_components(fg_clean)

        # ── Step 2: accumulate unstable-motion map at downscale resolution ────
        # fg_clean is already at downscale size (frame was resized before this call).
        if ENABLE_UNSTABLE_MOTION_MAP:
            fg_bin = (fg_clean > 0).astype(np.float32)
            if self._unstable_motion_accumulator is None:
                self._unstable_motion_accumulator = np.zeros_like(fg_bin, dtype=np.float32)
                self._unstable_motion_frame_count = 0
            self._unstable_motion_accumulator += fg_bin
            self._unstable_motion_frame_count += 1

        mask_resized = cv2.resize(fg_clean.astype(np.float32), (w, h))
        # Conservative EWA blend: raw single-frame mask is noisier than the old
        # slow-learning fgbg, so weight the new frame less than the original 0.6
        blended = cv2.addWeighted(mask_resized, 0.35, self.prev_mask, 0.65, 0)
        self.prev_mask = blended
        self.acc_mask = cv2.add(self.acc_mask, blended)

        if self.accum_time > 0:
            self.accum_time -= 1
        if self.accum_time == 0:
            self.__calibration_ended = True


    def __apply_calibration(self, frame, **kwargs): 

        save_img = kwargs.get("save_img", False)

        if self.acc_mask is None: 
            return None

        acc_mask_norm = cv2.normalize(self.acc_mask, None, 0, 255, cv2.NORM_MINMAX)
        acc_uint8 = acc_mask_norm.astype(np.uint8)

        lanes_closed = cv2.morphologyEx(acc_uint8, cv2.MORPH_CLOSE, self.kernel15, iterations=3)

        _, labels, stats, _ = cv2.connectedComponentsWithStats(lanes_closed, connectivity=8)

        # Fraction-based threshold adapts to any input resolution
        h_acc, w_acc = self.acc_mask.shape
        min_area = max(500, int(0.005 * h_acc * w_acc))
        lanes_clean = np.zeros_like(lanes_closed)

        for i, stat in enumerate(stats):
            if i == 0:
                continue
            if stat[cv2.CC_STAT_AREA] >= min_area:
                lanes_clean[labels == i] = 255

        lanes_smooth = cv2.GaussianBlur(lanes_clean, (11, 11), 0)
        _, candidate_lanes = cv2.threshold(lanes_smooth, 50, 255, cv2.THRESH_BINARY)

        # ── Step 2: finalise unstable-motion map ─────────────────────────────
        if (
            ENABLE_UNSTABLE_MOTION_MAP
            and self._unstable_motion_accumulator is not None
            and self._unstable_motion_frame_count > 0
            and self._unstable_motion_map is None   # build only once (first calibration)
        ):
            norm = self._unstable_motion_accumulator / float(self._unstable_motion_frame_count)
            self._unstable_motion_map = (norm > UNSTABLE_MOTION_THRESHOLD).astype(np.float32)
            logger.info(
                "Unstable motion map built: %d unstable pixels (%.1f%% of downscale area)",
                int(self._unstable_motion_map.sum()),
                100.0 * float(self._unstable_motion_map.sum()) / max(float(self._unstable_motion_map.size), 1.0),
            )

        # Hybrid merge: blend motion accumulation result with static geometry baseline
        candidate_lanes = self._merge_with_static(candidate_lanes)

        candidate_crosswalk = self.__detect_crosswalks(last_frame=frame, lanes_mask=candidate_lanes)

        previous_lanes = None if self.lanes_mask is None else self.lanes_mask.copy()
        previous_crosswalk = None if self.crosswalk_mask is None else self.crosswalk_mask.copy()

        keep_candidate = self._has_nonempty_mask(candidate_lanes) or not self._has_nonempty_mask(previous_lanes)
        if keep_candidate:
            self.lanes_mask = candidate_lanes
            self.crosswalk_mask = candidate_crosswalk
            lanes_final = candidate_lanes
        else:
            lanes_final = previous_lanes
            self.crosswalk_mask = previous_crosswalk
            logger.info("Runtime lane recalibration produced an empty mask; keeping previous lane mask")

        # ── Step 3: seed drivable confidence map with static lane signal ──────
        self._rebuild_drivable_confidence()

        if save_img and self.save_path and lanes_final is not None:
            self.__save_calibration(
                frame=frame,
                lanes_final=lanes_final,
                crosswalk_final=self.crosswalk_mask,
            )
        
        return lanes_final

    def _run_static_lane_detection(self) -> None:
        bg = self.bg_subtractor.getBackgroundImage()
        if bg is None or self._last_frame_shape is None:
            return
        h, w = self._last_frame_shape
        if bg.shape[:2] != (h, w):
            bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = detect_static_lanes(bg, kernel15=self.kernel15)
        if mask is not None:
            self._static_lanes_mask = mask
            logger.info(
                "Static lane detection: %d foreground pixels (%.1f%% of frame)",
                cv2.countNonZero(mask),
                100.0 * cv2.countNonZero(mask) / max(float(mask.size), 1.0),
            )
        else:
            logger.debug("Static lane detection: no reliable road markings found")


    def _mask_fill_ratio(self, mask: Optional[np.ndarray]) -> float:
        if mask is None or mask.size == 0:
            return 0.0
        return float(cv2.countNonZero(mask)) / max(float(mask.size), 1.0)


    def _merge_with_static(self, motion_mask: np.ndarray) -> np.ndarray:
        """
        Merge the motion-accumulation lane candidate with the static geometry baseline.

        - No static mask available → return motion_mask unchanged.
        - Sparse traffic (low mean density) → static mask is primary; motion
          accumulation did not observe enough vehicles to be trustworthy.
        - Mid-density traffic → keep motion lanes and add only nearby static support.
        - Dense traffic → prefer motion-derived lanes so static geometry cannot
          flood the whole ROI.
        """
        if self._static_lanes_mask is None:
            return motion_mask

        static = self._static_lanes_mask
        if static.shape != motion_mask.shape:
            static = cv2.resize(
                static, (motion_mask.shape[1], motion_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        mean_density = (
            float(np.mean(list(self._motion_score_window)))
            if self._motion_score_window else 0.0
        )
        static_fill_ratio = self._mask_fill_ratio(static)

        if not self._has_nonempty_mask(motion_mask):
            merged = static.copy()
            logger.debug(
                "Lane mask: static primary, empty motion candidate, density=%.4f static_fill=%.3f",
                mean_density,
                static_fill_ratio,
            )
            return merged

        motion_fill_ratio = self._mask_fill_ratio(motion_mask)
        if motion_fill_ratio >= self._static_lane_max_fill_ratio:
            logger.info(
                "Lane mask: motion candidate overfills ROI (fill=%.3f); "
                "using static baseline (density=%.4f static_fill=%.3f)",
                motion_fill_ratio,
                mean_density,
                static_fill_ratio,
            )
            return static.copy()

        motion_support = cv2.dilate(motion_mask, self.kernel15, iterations=2)
        static_supported = cv2.bitwise_and(static, motion_support)

        if mean_density < self._motion_sparse_threshold:
            merged = static.copy()
            logger.debug(
                "Lane mask: static primary, sparse motion, density=%.4f static_fill=%.3f",
                mean_density,
                static_fill_ratio,
            )
        elif mean_density >= self._motion_dense_threshold:
            if static_fill_ratio >= self._static_lane_max_fill_ratio:
                merged = motion_mask.copy()
                logger.info(
                    "Lane mask: motion primary, suppressing overgrown static lane mask in dense traffic "
                    "(density=%.4f static_fill=%.3f)",
                    mean_density,
                    static_fill_ratio,
                )
            elif self._has_nonempty_mask(static_supported):
                merged = cv2.bitwise_or(motion_mask, static_supported)
                logger.debug(
                    "Lane mask: motion primary with local static support, density=%.4f static_fill=%.3f",
                    mean_density,
                    static_fill_ratio,
                )
            else:
                merged = motion_mask.copy()
                logger.debug(
                    "Lane mask: motion primary, dense traffic, density=%.4f static_fill=%.3f",
                    mean_density,
                    static_fill_ratio,
                )
        elif self._has_nonempty_mask(static_supported):
            merged = cv2.bitwise_or(motion_mask, static_supported)
            logger.debug(
                "Lane mask: constrained hybrid, density=%.4f static_fill=%.3f",
                mean_density,
                static_fill_ratio,
            )
        else:
            merged = motion_mask.copy()
            logger.debug(
                "Lane mask: motion fallback, no nearby static support, density=%.4f static_fill=%.3f",
                mean_density,
                static_fill_ratio,
            )

        return merged


    def get_scene_masks(self, expand_px: int = 0) -> dict:
        """
        Returns lane, crosswalk, and drivable-confidence masks in current frame
        coordinates.  `expand_px` dilates binary regions to increase tolerance
        (high-attention mode).

        When expand_px == 0 the stored masks are returned by reference
        (no copy). Callers must not mutate the returned arrays.
        The drivable_confidence_map is always returned by reference (float, read-only).
        """
        drivable = self._drivable_confidence_map  # Step 3: always attach, may be None

        if self.lanes_mask is None:
            if self._static_lanes_mask is not None:
                return {
                    "lane_mask": self._static_lanes_mask,
                    "crosswalk_mask": np.zeros_like(self._static_lanes_mask, dtype=np.uint8),
                    "drivable_confidence_map": drivable,
                }
            if self._last_frame_shape is None:
                return {"lane_mask": None, "crosswalk_mask": None, "drivable_confidence_map": drivable}
            h, w = self._last_frame_shape
            return {
                "lane_mask": np.zeros((h, w), dtype=np.uint8),
                "crosswalk_mask": np.zeros((h, w), dtype=np.uint8),
                "drivable_confidence_map": drivable,
            }

        if expand_px <= 0:
            cw = self.crosswalk_mask
            if cw is None:
                if not hasattr(self, "_zeros_crosswalk") or self._zeros_crosswalk.shape != self.lanes_mask.shape:
                    self._zeros_crosswalk = np.zeros_like(self.lanes_mask, dtype=np.uint8)
                cw = self._zeros_crosswalk
            return {"lane_mask": self.lanes_mask, "crosswalk_mask": cw, "drivable_confidence_map": drivable}

        # Dilation path (high-attention mode) – copies are required for binary masks
        lane = self.lanes_mask.copy()
        crosswalk = None if self.crosswalk_mask is None else self.crosswalk_mask.copy()
        k = max(3, int(expand_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        lane = cv2.dilate(lane, kernel, iterations=1)
        if crosswalk is not None:
            crosswalk = cv2.dilate(crosswalk, kernel, iterations=1)
            crosswalk = cv2.bitwise_and(crosswalk, lane)
        if crosswalk is None:
            crosswalk = np.zeros_like(lane, dtype=np.uint8)
        return {"lane_mask": lane, "crosswalk_mask": crosswalk, "drivable_confidence_map": drivable}

    def __detect_crosswalks(self, last_frame: Optional[np.ndarray], lanes_mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Non-ML crosswalk detection from lane region:
        find repeated bright stripe-like blobs and merge them into a crosswalk region.
        Runs only when lane calibration is finalized.
        """
        if last_frame is None or lanes_mask is None:
            if self._last_frame_shape is None:
                return np.zeros((1, 1), dtype=np.uint8)
            return np.zeros(self._last_frame_shape, dtype=np.uint8)

        if lanes_mask.ndim != 2:
            lanes_mask = cv2.cvtColor(lanes_mask, cv2.COLOR_BGR2GRAY)
        lanes_mask = (lanes_mask > 0).astype(np.uint8) * 255

        if cv2.countNonZero(lanes_mask) == 0:
            return np.zeros_like(lanes_mask, dtype=np.uint8)

        if last_frame.ndim == 3:
            gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = last_frame.copy()

        lane_gray = cv2.bitwise_and(gray, gray, mask=lanes_mask)
        lane_gray = cv2.GaussianBlur(lane_gray, (5, 5), 0)

        # Enhance bright zebra-like paint marks.
        top_hat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        enhanced = cv2.morphologyEx(lane_gray, cv2.MORPH_TOPHAT, top_hat_kernel)
        _, stripes = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        stripes = cv2.morphologyEx(
            stripes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
        )
        stripes = cv2.morphologyEx(
            stripes, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3)), iterations=2
        )
        stripes = cv2.bitwise_and(stripes, lanes_mask)

        contours, _ = cv2.findContours(stripes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lane_area = float(cv2.countNonZero(lanes_mask))
        min_area = max(50.0, 0.0015 * lane_area)

        stripe_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue

            aspect = float(w) / float(max(h, 1))
            if aspect < 3.5:
                continue

            extent = area / float(w * h)
            if extent < 0.45:
                continue

            stripe_candidates.append(
                {
                    "contour": cnt,
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "cx": float(x + (w * 0.5)),
                    "cy": float(y + (h * 0.5)),
                }
            )

        if len(stripe_candidates) < 4:
            return np.zeros_like(lanes_mask, dtype=np.uint8)

        stripe_candidates.sort(key=lambda item: item["cy"])

        def _x_overlap_ratio(a: dict, b: dict) -> float:
            left = max(a["x"], b["x"])
            right = min(a["x"] + a["w"], b["x"] + b["w"])
            overlap = max(0.0, float(right - left))
            denom = max(float(max(a["w"], b["w"])), 1.0)
            return overlap / denom

        def _is_sequential(prev: dict, curr: dict) -> bool:
            gap_y = curr["cy"] - prev["cy"]
            mean_h = 0.5 * float(prev["h"] + curr["h"])
            width_ratio = float(curr["w"]) / float(max(prev["w"], 1))
            height_ratio = float(curr["h"]) / float(max(prev["h"], 1))
            x_overlap = _x_overlap_ratio(prev, curr)

            return (
                gap_y >= (0.35 * mean_h)
                and gap_y <= (4.0 * max(prev["h"], curr["h"]))
                and 0.55 <= width_ratio <= 1.8
                and 0.5 <= height_ratio <= 2.0
                and x_overlap >= 0.45
            )

        stripe_runs = []
        current_run = [stripe_candidates[0]]
        for cand in stripe_candidates[1:]:
            if _is_sequential(current_run[-1], cand):
                current_run.append(cand)
            else:
                stripe_runs.append(current_run)
                current_run = [cand]
        stripe_runs.append(current_run)

        stripe_mask = np.zeros_like(lanes_mask, dtype=np.uint8)
        selected_runs = 0

        for run in stripe_runs:
            if len(run) < 4:
                continue

            gaps = np.diff([item["cy"] for item in run]).astype(np.float32)
            if gaps.size > 0:
                gap_mean = float(gaps.mean())
                if gap_mean <= 0:
                    continue
                gap_cv = float(gaps.std() / gap_mean)
                if gap_cv > 0.55:
                    continue

            run_heights = np.array([item["h"] for item in run], dtype=np.float32)
            run_span = float(run[-1]["cy"] - run[0]["cy"])
            if run_span < (2.5 * float(np.median(run_heights))):
                continue

            selected_runs += 1
            for item in run:
                cv2.drawContours(stripe_mask, [item["contour"]], -1, 255, thickness=-1)

        # Require at least one long ordered stripe run instead of isolated bright bars.
        if selected_runs == 0:
            return np.zeros_like(lanes_mask, dtype=np.uint8)

        crosswalk = cv2.morphologyEx(
            stripe_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9)),
            iterations=2,
        )
        crosswalk = cv2.dilate(
            crosswalk,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        crosswalk = cv2.bitwise_and(crosswalk, lanes_mask)

        # Remove tiny islands.
        _, labels, stats, _ = cv2.connectedComponentsWithStats(crosswalk, connectivity=8)
        min_cc_area = max(200.0, 0.0025 * lane_area)
        final_mask = np.zeros_like(crosswalk, dtype=np.uint8)
        for i, stat in enumerate(stats):
            if i == 0:
                continue
            if stat[cv2.CC_STAT_AREA] >= min_cc_area:
                final_mask[labels == i] = 255

        return final_mask



    def _build_scene_overlay(
        self,
        frame: np.ndarray,
        lane_mask: Optional[np.ndarray],
        crosswalk_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if frame is None:
            return None

        overlay = frame.copy()
        out = frame.copy()
        lane_color = (255, 200, 80)
        crosswalk_color = (0, 255, 255)

        if lane_mask is not None and lane_mask.size > 0 and cv2.countNonZero(lane_mask) > 0:
            lane_contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lane_contours = [cnt for cnt in lane_contours if cv2.contourArea(cnt) >= 500]
            if lane_contours:
                cv2.drawContours(overlay, lane_contours, -1, lane_color, thickness=-1)
                cv2.drawContours(out, lane_contours, -1, lane_color, thickness=2)

        if crosswalk_mask is not None and crosswalk_mask.size > 0 and cv2.countNonZero(crosswalk_mask) > 0:
            crosswalk_contours, _ = cv2.findContours(crosswalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            crosswalk_contours = [cnt for cnt in crosswalk_contours if cv2.contourArea(cnt) >= 100]
            if crosswalk_contours:
                cv2.drawContours(overlay, crosswalk_contours, -1, crosswalk_color, thickness=-1)
                cv2.drawContours(out, crosswalk_contours, -1, crosswalk_color, thickness=2)

        return cv2.addWeighted(overlay, 0.22, out, 0.78, 0.0)

    def _resolve_output_path(self, save_path: Optional[Union[str, Path]] = None) -> Path:
        output_path = Path(save_path or self.save_path).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def save_lane_mask_only(
        self,
        save_path: Optional[Union[str, Path]] = None,
        lane_mask: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        mask = lane_mask
        if mask is None:
            mask = self.lanes_mask if self._has_nonempty_mask(self.lanes_mask) else self._static_lanes_mask
        if not self._has_nonempty_mask(mask):
            logger.warning("Lane mask save requested, but no lane mask is available yet")
            return None

        output_path = self._resolve_output_path(save_path)
        cv2.imwrite(str(output_path), mask)
        logger.info("Saved lane-only mask to %s", output_path)
        return output_path

    def __save_calibration(
        self,
        frame: np.ndarray,
        lanes_final: np.ndarray,
        crosswalk_final: Optional[np.ndarray],
    ):
        if frame is None:
            return

        save_path = self._resolve_output_path()
        self.save_lane_mask_only(save_path=save_path, lane_mask=lanes_final)

        save_stem = save_path.stem or "lanes_final"
        save_suffix = save_path.suffix or ".png"

        if (
            self.save_scene_overlays
            and
            self._saved_lane_extractions < self._max_saved_scene_extractions
            and lanes_final is not None
            and lanes_final.size > 0
            and cv2.countNonZero(lanes_final) > 0
        ):
            lane_overlay = self._build_scene_overlay(frame, lanes_final)
            if lane_overlay is not None:
                lane_path = save_path.parent / f"{save_stem}_lanes_{self._saved_lane_extractions + 1:02d}{save_suffix}"
                cv2.imwrite(str(lane_path), lane_overlay)
                self._saved_lane_extractions += 1

        if (
            self.save_scene_overlays
            and
            self._saved_crosswalk_extractions < self._max_saved_scene_extractions
            and crosswalk_final is not None
            and crosswalk_final.size > 0
            and cv2.countNonZero(crosswalk_final) > 0
        ):
            crosswalk_overlay = self._build_scene_overlay(frame, lanes_final, crosswalk_final)
            if crosswalk_overlay is not None:
                crosswalk_path = (
                    save_path.parent
                    / f"{save_stem}_crosswalks_{self._saved_crosswalk_extractions + 1:02d}{save_suffix}"
                )
                cv2.imwrite(str(crosswalk_path), crosswalk_overlay)
                self._saved_crosswalk_extractions += 1

        # calb_contours, _ = cv2.findContours(lanes_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(last_frame, calb_contours, -1, (0,255,0), 2) 
        # cv2.imshow("Lanes Overlay", last_frame) 
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 

            
