from obs_system.camera_module.interface.consumer import Consumer
import cv2
import numpy as np
from multiprocessing import connection

class DummyConsumer(Consumer):
    def __init__(self, conn: connection, camera_id: int, dummy_path: str) -> None:
        super().__init__(conn)
        self.camera_id = camera_id
        self.dummy_path = dummy_path
        self.cap = None

    def connect(self):
        pass
    
    def get_last_frame(self, camera_id: int) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            # Restart the video if it ends
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame
    
    def close(self):
        self.cap.release()

    def transmit_frame(self, frame: np.ndarray):
        self.conn.send(frame)
    
    def run(self):
        self.connect()
        self.cap = cv2.VideoCapture(self.dummy_path)
        while True:
            frame = self.get_last_frame(0)
            
            if frame is not None and self.conn is not None:
                try:
                    self.transmit_frame(frame)
                    
                except Exception as e:
                    print("Error sending frame:", e)
                    break
