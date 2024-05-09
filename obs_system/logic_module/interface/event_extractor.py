from abc import ABC, abstractmethod
import numpy as np

class EventExtractorInterface(ABC):
    @abstractmethod
    def detect(self, predictions: np.ndarray) -> list:
        pass