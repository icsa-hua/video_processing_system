from abc import ABC, abstractmethod
import numpy as np

class EventExtractorInterface(ABC):

    @abstractmethod
    def detect(self, predictions: np.ndarray, save:bool) -> list:
        pass
