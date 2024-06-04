from obs_system.camera_module.interface.consumer import Consumer
import cv2
import numpy as np
from multiprocessing import connection
import time

class DummyConsumer(Consumer):
    def __init__(self, conn: connection, camera_id: int, dummy_path: str) -> None:
        super().__init__(conn)
        self.camera_id = camera_id
        self.dummy_path = dummy_path
        self.cap = None

    def connect(self): #multiprocessing 
        pass
    
    def get_last_frame(self, camera_id: int) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            # Restart the video if it ends
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame

 
    def close(self):
        #Close all video representatives. 
        self.cap.release()

    def transmit_frame(self, frame: np.ndarray):
        #child process sends frame to parent process. 
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        self.conn.send((frame.shape, frame.dtype, frame.tobytes()))
    
    def run(self):
        self.connect()
        self.cap = cv2.VideoCapture(self.dummy_path)
        while True:
            frame = self.get_last_frame(0)
            if frame is not None and self.conn is not None:
                #If frame pointer is not empty and processes are functional. 
                try:
                    self.transmit_frame(frame)
                    
                except Exception as e:
                    print("Error sending frame:", e)
                    break
