from abc import ABC, abstractmethod
import numpy as np
from multiprocessing import connection

class Consumer(ABC):
    def __init__(self, conn: connection):
        self.conn = conn

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def get_last_frame(self, camera_id: int) -> np.ndarray:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def transmit_frame(self, frame: np.ndarray):
        pass