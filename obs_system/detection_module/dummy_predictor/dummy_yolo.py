from obs_system.detection_module.interface.predictor import ModelLoader
import numpy as np
import cv2 

class DummyPredictor(ModelLoader):
    def __init__(self,model_path: str) -> None:
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        pass
    
    def predict(self, frame: np.ndarray) -> np.ndarray:
        #result = self.model.predict(frame)
        #Dummy result
        # Each detection is represented by [x, y, width, height, class_id]
        result = np.array([
            [100, 150, 200, 250, 1],  # First dummy detection
            [400, 350, 150, 200, 2],  # Second dummy detection
            [50, 80, 100, 150, 3]   # Third dummy detection
        ])
        return result
    
    def draw_boxes(self, frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
        for detection in detections:
            x, y, width, height, class_id = detection
            # Draw a rectangle around each detection
            start_point = (int(x), int(y))
            end_point = (int(x + width), int(y + height))
            color = (0, 255, 0)  # Green color for the rectangle
            thickness = 2  # Thickness of the rectangle border
            final_frane = cv2.rectangle(frame, start_point, end_point, color, thickness)

        return final_frane


