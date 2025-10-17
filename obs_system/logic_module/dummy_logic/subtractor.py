from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
from obs_system.utils.global_config import TRIALS, HISTORY, THR_RATIO, K_CONSECUTIVE, HOLD_FRAMES, MIN_OBJ_AREA
from obs_system.utils.logger import get_logger
import numpy as np
import cv2
import os
import pdb

from collections import deque

logger = get_logger("obs_system"+__name__)

class Subtractor(EventExtractorInterface): 
    def __init__(self, trials=TRIALS, history=HISTORY, threshold_ratio=THR_RATIO, detect_shadows=True, empty_background_image="", downscale=(320,320)):
        self.downscale = downscale
        self.threshold_ratio = float(threshold_ratio)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, 
                varThreshold=100,
                detectShadows=detect_shadows)
        self.static_bg = False 

        #Hysteresis 
        self.hold_frames = HOLD_FRAMES # Number of allowed frames to have movement.  
        self._recent = deque(maxlen=K_CONSECUTIVE)
        self._hold = 0 

        if empty_background_image: 
            logger.debug("Empty Background traing for subtractor")
            empty_bg = cv2.imread(empty_background_image)
            
            if empty_bg is not None: 

                if self.downscale: 
                    empty_bg = cv2.resize(empty_bg, self.downscale, interpolation=cv2.INTER_AREA)

                for _ in range(trials): 
                    self.bg_subtractor.apply(empty_bg, learningRate=1.0) 
                self.static_bg = True

    
    def detect(self, batch, save_img=False): 

        save_dir = "" 
        if save_img: 
            parent = os.getcwd()
            save_dir = f"{parent}/assets/background_check/"
            os.makedirs(save_dir, exist_ok = True)
            logger.debug(f"Background Images saved in {save_dir}")
            self._save_idx = 0 

        motion_flags = [] 
        for frame in batch:
            if self.downscale:
                frame = cv2.resize(frame, self.downscale, interpolation=cv2.INTER_AREA)

            lr = 0.0 if self.static_bg else -1 
            mask = self.bg_subtractor.apply(frame, learningRate=lr) 
            mask = (mask==255).astype(np.uint8)* 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(mask, contours, -1, (0, 255, 0), 3)

            flag = False 
            if len(contours) > 0:
                # Edge Case single moving car will fail 
                motion_pixels = cv2.countNonZero(mask)
                total = mask.shape[0] * mask.shape[1] 

                threshold = int(self.threshold_ratio * total)
                    
                # Object-aware threshold (largest contour area) 
                max_obj_area = max((cv2.contourArea(c) for c in contours), default = 0) 

                min_obj_area = MIN_OBJ_AREA * total

                flag = (motion_pixels > threshold) or (max_obj_area > min_obj_area)
            
            self._recent.append(flag) 
            if self._hold > 0: 
                motion_flag = True 
                self._hold -= 1 
            else: 
                motion_flag = flag 

                if len(self._recent) == self._recent.maxlen and all(self._recent): 
                    self._hold = self.hold_frames 

            motion_flags.append(motion_flag)
            if save_img: 
                idx = self._save_idx 
                self._save_idx += 1 
                
                # cv2.imwrite(os.path.join(save_dir, f"{idx:06d}_mask.png"), mask)
                motion_cutout = cv2.bitwise_and(frame,frame, mask=mask) 
                cv2.imwrite(os.path.join(save_dir, f"{idx:06d}_motion.png"), motion_cutout)

        return motion_flags
        
        
