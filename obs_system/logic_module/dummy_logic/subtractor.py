from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
from obs_system.utils.global_config import (
    TRIALS,
    HISTORY,
    VARTHRESHOLD,
    THR_RATIO,
    K_CONSECUTIVE,
    HOLD_FRAMES,
    MIN_OBJ_AREA,
)
from obs_system.utils.logger import get_logger

import numpy as np
import cv2
import os

from collections import deque
from pathlib import Path
from typing import Optional

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
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=history, 
                    varThreshold=VARTHRESHOLD,
                    detectShadows=detect_shadows)


        self.fgbg = cv2.createBackgroundSubtractorMOG2(
                    history=history, 
                    varThreshold=VARTHRESHOLD,
                    detectShadows=False)

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


    def _has_nonempty_mask(self, mask: Optional[np.ndarray]) -> bool:
        return bool(mask is not None and mask.size > 0 and cv2.countNonZero(mask) > 0)


    def _begin_runtime_recalibration(self) -> None:
        if not self._recalibration_enabled or self.recalibration_accum_time <= 0:
            return

        self.accum_time = self.recalibration_accum_time
        self._recalibration_active = True
        self._frames_since_last_calibration = 0
        self._reset_calibration_buffers()
        logger.info(
            "Starting runtime lane recalibration for %d frames",
            self.recalibration_accum_time,
        )


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


    def detect(self, batch, save_img:bool=False): 

        if not batch: return []

        save_dir = None
        save_idx = None
        if save_img: 
            parent = os.getcwd()
            save_dir = f"{parent}/assets/background_check/"
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
        lanes_final = None
        last_frame = batch[-1]
        startup_skip_batch = self._startup_warmup_active and not self._ready_for_inference

        for frame in batch:

            if self.downscale: 
                frame = cv2.resize(frame, self.downscale, interpolation=cv2.INTER_AREA)

            motion_flag = self.__call_subtractor(frame, save_dir=save_dir, save_img=save_img, save_idx=save_idx)

            motion_flags.append(motion_flag)
            motion_scores.append(getattr(self, '_last_motion_score', 0.0))

            if save_img and save_idx is not None: 
                save_idx += 1 

            should_accumulate_calibration = False
            if self.accum_time > 0:
                should_accumulate_calibration = startup_skip_batch or self._recalibration_active or motion_flag

            if should_accumulate_calibration:
                self.__cal_calibrator(frame, size=(h,w))
                if startup_skip_batch:
                    self._startup_frames_seen = min(self.initial_accum_time, self._startup_frames_seen + 1)

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
     
        # Learning rate: 0 if static pre-trained background, default otherwise
        lr = 0.0 if self.static_bg else -1 
        mask = self.bg_subtractor.apply(frame, learningRate=lr) 

        _,subtractor_mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        subtractor_mask = cv2.morphologyEx(subtractor_mask, cv2.MORPH_OPEN, self.kernel3)
        subtractor_mask = cv2.morphologyEx(subtractor_mask, cv2.MORPH_CLOSE, self.kernel3)

        # Motion score: foreground pixel ratio in [0,1]
        motion_pixels = cv2.countNonZero(subtractor_mask)
        total_pixels = float(subtractor_mask.shape[0] * subtractor_mask.shape[1])
        motion_score = (motion_pixels / total_pixels) if total_pixels > 0 else 0.0
        self._last_motion_score = motion_score
        contours, _ = cv2.findContours(subtractor_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        flag = False 
        if contours:  
            
            # Edge case single moving car will fail 
            motion_pixels = cv2.countNonZero(subtractor_mask) 
            total = subtractor_mask.shape[0] * subtractor_mask.shape[1] 

            threshold = int(self.threshold_ratio * total) 

            #Object aware threshold (largest contour area) 
            max_obj_area = max((cv2.contourArea(c) for c in contours), default=0) 
            min_obj_area = MIN_OBJ_AREA * total 
            flag = (motion_pixels > threshold) or (max_obj_area > min_obj_area) 

        # Hysteresis
        self._recent.append(flag) 

        if self._hold > 0: 
            motion_flag = True 
            self._hold -= 1 
        else: 
            motion_flag = flag 
            if len(self._recent) == self._recent.maxlen and all(self._recent): 
                self._hold = self.hold_frames 
        
        if save_dir is not None and save_img and save_idx is not None: 
            self.__save_subtractor(frame, subtractor_mask, save_dir, save_idx)

        return motion_flag 

            
    def __save_subtractor(self, frame, mask, save_dir:str, idx:int): 
        motion_cutout = cv2.bitwise_and(frame, frame, mask=mask) 
        cv2.imwrite(os.path.join(save_dir, f"{idx:06d}_motion.png"), motion_cutout) 


    def __cal_calibrator(self, frame, **kwargs): 

        if self.__calibration_ended: 
            return self.acc_mask, self.prev_mask

        h, w = kwargs["size"]

        fg_mask = self.fgbg.apply(frame, learningRate=0.01) 
        _, fgmask_threshold = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY) 

        fgmask_clean = cv2.morphologyEx(fgmask_threshold, cv2.MORPH_OPEN, self.kernel5, iterations=2) 
        fgmask_clean = cv2.morphologyEx(fgmask_clean, cv2.MORPH_CLOSE, self.kernel5, iterations=2)

        mask_resized = cv2.resize(fgmask_clean, (w,h))
        blended = cv2.addWeighted(mask_resized.astype(np.float32), 0.6, self.prev_mask, 0.4, 0)

        self.prev_mask = blended
        self.acc_mask = cv2.add(self.acc_mask, blended) 

        if not self.__calibration_ended and self.accum_time > 0: 
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

        min_area = 15000 
        lanes_clean = np.zeros_like(lanes_closed) 

        for i, stat in enumerate(stats): 
            if i == 0 : 
                continue 

            if stat[cv2.CC_STAT_AREA] >= min_area: 
                lanes_clean[labels == i] = 255

        lanes_smooth = cv2.GaussianBlur(lanes_clean, (11,11), 0) 
        _, candidate_lanes = cv2.threshold(lanes_smooth, 50, 255, cv2.THRESH_BINARY)
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
        
        if save_img and self.save_path and lanes_final is not None:
            self.__save_calibration(
                frame=frame,
                lanes_final=lanes_final,
                crosswalk_final=self.crosswalk_mask,
            )
        
        return lanes_final

    def get_scene_masks(self, expand_px: int = 0) -> dict:
        """
        Returns lane and crosswalk masks in current frame coordinates.
        `expand_px` dilates regions to increase tolerance (high-attention mode).
        """
        lane = None if self.lanes_mask is None else self.lanes_mask.copy()
        crosswalk = None if self.crosswalk_mask is None else self.crosswalk_mask.copy()

        if lane is None:
            if self._last_frame_shape is None:
                return {"lane_mask": None, "crosswalk_mask": None}
            h, w = self._last_frame_shape
            return {
                "lane_mask": np.zeros((h, w), dtype=np.uint8),
                "crosswalk_mask": np.zeros((h, w), dtype=np.uint8),
            }

        if expand_px > 0:
            k = max(3, int(expand_px) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            lane = cv2.dilate(lane, kernel, iterations=1)
            if crosswalk is not None:
                crosswalk = cv2.dilate(crosswalk, kernel, iterations=1)
                crosswalk = cv2.bitwise_and(crosswalk, lane)

        if crosswalk is None:
            crosswalk = np.zeros_like(lane, dtype=np.uint8)

        return {"lane_mask": lane, "crosswalk_mask": crosswalk}

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

    def __save_calibration(
        self,
        frame: np.ndarray,
        lanes_final: np.ndarray,
        crosswalk_final: Optional[np.ndarray],
    ):
        if frame is None:
            return

        save_path = Path(self.save_path).expanduser()
        if not save_path.is_absolute():
            save_path = Path.cwd() / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path), lanes_final)

        save_stem = save_path.stem or "lanes_final"
        save_suffix = save_path.suffix or ".png"

        if (
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

            
