from abc import ABC, abstractmethod
import numpy as np

class ONNXInterface(ABC):
    def __init__(self, model_path):
        self.model_path = model_path #ONNX model path
        self.session = None

    @abstractmethod
    def quantize_model(self): 
        pass

    @abstractmethod
    def load_model(self, source: str): 
        pass

    @abstractmethod
    def session_create(self):
        pass

    @abstractmethod
    def session_run(self, input_data) -> np.ndarray:
        pass


