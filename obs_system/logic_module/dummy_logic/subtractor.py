from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
import numpy as np
import cv2


class Subtractor(EventExtractorInterface): 
    def __init__(self, trials=10, history=500, threshold_ratio=0.02, detect_shadows=True, empty_background_image="", downscale=(640,640)):
        self.downscale = downscale
        self.threshold_ratio = float(threshold_ratio)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, 
                varThreshold=100,
                detectShadows=detect_shadows)
        self.static_bg = False 
        
        if empty_background_image: 
            empty_bg = cv2.imread(empty_background_image)

            if empty_bg is not None: 

                if self.downscale: 
                    empty_bg = cv2.resize(empty_bg, self.downscale, interpolation=cv2.INTER_AREA)

                for _ in range(trials): 
                    self.bg_subtractor.apply(empty_bg, learningRate=1.0) 
                self.static_bg = True

    
    def detect(self, batch): 
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
            if len(contours) > 0:
                motion_pixels = cv2.countNonZero(mask)
                threshold = int(self.threshold_ratio * (mask.shape[0] * mask.shape[1]))
                motion_flags.append(motion_pixels > threshold)
            else:
                motion_flags.append(False)

        return motion_flags
        
        
