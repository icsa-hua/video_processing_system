from abc import ABC, abstractmethod
import numpy as np


class ModelLoader(ABC):

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> np.ndarray:
        pass