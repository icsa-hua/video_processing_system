from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
import numpy as np
import cv2


class Subtractor(EventExtractorInterface): 
    def __init__(self, trials=10, history=500, threshold=100, detect_shadows=True, empty_background_image=""):
        empty_background_image = cv2.imread(empty_background_image)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=threshold, detectShadows=detect_shadows)

        for _ in range(trials): 
            self.bg_subtractor.apply(empty_background_image) 

    
    def detect(self, batch, threshold=500): 
        motion_flags = [] 
        for frame in batch:
            mask = self.bg_subtractor.apply(frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(mask, contours, -1, (0, 255, 0), 3)
            if len(contours) > 0:
                motion_pixels = cv2.countNonZero(mask)
                if motion_pixels > threshold:  # You choose the threshold based on scene
                    motion_flags.append(True)
                else:
                    motion_flags.append(False)
            else:
                motion_flags.append(False)

        return motion_flags
        
        