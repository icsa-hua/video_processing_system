from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
import numpy as np
import cv2
import time 

class BoundingBoxOverlapDetector(EventExtractorInterface):
    def detect(self, predictions: np.ndarray) -> list:
        overlaps = []

        for i in range(predictions.shape[0]):
            for j in range(i+1, predictions.shape[0]):
                if self._boxes_overlap(predictions[i][:4], predictions[j][:4]):
                    overlaps.append((predictions[i], predictions[j]))

        return overlaps

    def _boxes_overlap(self, box1: np.ndarray, box2: np.ndarray) -> bool:
        # Unpack coordinates and dimensions
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Determine the coordinates of the overlap rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        # Check for no overlap
        if x_right < x_left or y_bottom < y_top:
            return False

        return True
    
    def _draw_semitransparent_rectangle(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        overlay = image.copy()
        x, y, w, h = box

        # Define color and transparency factor (alpha)
        color = (0, 255, 0)  # Green color
        alpha = 0.5  # 50% transparency

        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)  # -1 fills the rectangle
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        return image
    
    def draw_overlaps(self, image: np.ndarray, overlaps: list) -> np.ndarray:
        print(type(image))
        print(type(overlaps))
        time.sleep(90)
        for overlap in overlaps:
            box1, box2 = overlap
            image = self._draw_semitransparent_rectangle(image, box1[:4])
            image = self._draw_semitransparent_rectangle(image, box2[:4])
            
        return image
