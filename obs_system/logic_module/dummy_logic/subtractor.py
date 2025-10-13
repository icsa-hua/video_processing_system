from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
import numpy as np
import cv2
import os

from collections import deque

class Subtractor(EventExtractorInterface): 
    def __init__(self, trials=10, history=500, threshold_ratio=0.02, detect_shadows=True, empty_background_image="", downscale=(320,320)):
        self.downscale = downscale
        self.threshold_ratio = float(threshold_ratio)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, 
                varThreshold=100,
                detectShadows=detect_shadows)
        self.static_bg = False 

        #Hysteresis 
        k_consecutive = 3 # Number of consecutive frames 
        self.hold_frames = 10 # Number of allowed frames to have movement.  
        self._recent = deque(maxlen=k_consecutive)
        self._hold = 0 

        if empty_background_image: 
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
                motion_pixels = cv2.countNonZero(mask)
                threshold = int(self.threshold_ratio * (mask.shape[0] * mask.shape[1]))
                flag = (motion_pixels > threshold)
            
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

                cv2.imwrite(os.path.join(save_dir, f"{idx:06d}_mask.png"), mask)
                motion_cutout = cv2.bitwise_and(frame,frame, mask=mask) 
                cv2.imwrite(os.path.join(save_dir, f"{idx:06d}_motion.png"), motion_cutout)








        return motion_flags
        
        
