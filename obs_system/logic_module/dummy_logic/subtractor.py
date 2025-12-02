from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
from obs_system.utils.global_config import TRIALS, HISTORY, VARTHRESHOLD,THR_RATIO, K_CONSECUTIVE, HOLD_FRAMES, MIN_OBJ_AREA, VARTHRESHOLD
from obs_system.utils.logger import get_logger

import numpy as np
import cv2
import os
import pdb

from collections import deque

logger = get_logger("obs_system"+__name__)

class Subtractor(EventExtractorInterface): 
    def __init__(self,
                 trials=TRIALS,
                 history=HISTORY,
                 threshold_ratio=THR_RATIO,
                 detect_shadows=True,
                 empty_background_image="",
                 downscale=(320,320),
                 accum_time:int=500,
                 save_path:str="lanes_final.png"
    ):
        self.downscale = downscale
        self.threshold_ratio = float(threshold_ratio)
        self.accum_time = accum_time
        self.save_path = save_path
        
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

        if empty_background_image: 
            logger.debug("Empty Background traing for subtractor")
            empty_bg = cv2.imread(empty_background_image)
            
            if empty_bg is not None: 

                if self.downscale: 
                    empty_bg = cv2.resize(empty_bg, self.downscale, interpolation=cv2.INTER_AREA)

                for _ in range(trials): 
                    self.bg_subtractor.apply(empty_bg, learningRate=1.0) 

                self.static_bg = True #Shows that we entered the first time. 

    
    def detect(self, batch, save_img:bool=False): 

        if not batch: return []

        save_dir = None
        if save_img: 
            parent = os.getcwd()
            save_dir = f"{parent}/assets/background_check/"
            os.makedirs(save_dir, exist_ok=True)
            logger.debug(f"Background Images saved in {save_dir}")
            save_idx = 0 
        else: 
            save_idx = None
        
        h, w = batch[0].shape[:2] 

        if not self.__calibration_started: 
            self.acc_mask = np.zeros((h, w), np.float32) 
            self.prev_mask = np.zeros((h,w), np.float32)
            self.__calibration_started = True 

        motion_flags = [] 
        last_frame = batch[-1]

        for frame in batch:

            if self.downscale: 
                frame = cv2.resize(frame, self.downscale, interpolation=cv2.INTER_AREA)

            motion_flag = self.__call_subtractor(frame, save_dir=save_dir, save_img=save_img, save_idx=save_idx)

            motion_flags.append(motion_flag)

            if save_img and save_idx is not None: 
                save_idx += 1 

            if motion_flag and self.accum_time > 0:
                self.__cal_calibrator(frame, size=(h,w))

        if self.accum_time == 0 and self.__calibration_ended :
            self.__apply_calibration(last_frame, save_img=save_img)
            self.accum_time = -1

        return motion_flags
        

    def __call_subtractor(self, frame, save_dir=None, save_img=False, save_idx:int=0): 
     
        # Learning rate: 0 if static pre-trained background, default otherwise
        lr = 0.0 if self.static_bg else -1 
        mask = self.bg_subtractor.apply(frame, learningRate=lr) 

        _,subtractor_mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        subtractor_mask = cv2.morphologyEx(subtractor_mask, cv2.MORPH_OPEN, self.kernel3)
        subtractor_mask = cv2.morphologyEx(subtractor_mask, cv2.MORPH_CLOSE, self.kernel3)
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

        if not self.__calibration_ended: 
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
        _, lanes_final = cv2.threshold(lanes_smooth, 50, 255, cv2.THRESH_BINARY) 
        
        if save_img and self.save_path: 
            self.__save_calibration(frame, lanes_final)
        
        return lanes_final



    def __save_calibration(self, last_frame, lanes_final): 
        if last_frame is None: 
            return 

        cv2.imwrite(self.save_path, lanes_final)
        # calb_contours, _ = cv2.findContours(lanes_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(last_frame, calb_contours, -1, (0,255,0), 2) 
        # cv2.imshow("Lanes Overlay", last_frame) 
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 

            

