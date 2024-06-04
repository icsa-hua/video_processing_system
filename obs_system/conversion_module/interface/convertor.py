from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Iterator, List

class CameraProcessor(ABC):
    @abstractmethod
    def __init__(self, frame, model, output_path: str, max_workers: int = 10, native_size: int = 1024, FOV: int = 90, overlap: float = 0.2):
        """
        Initialize the camera processor with the given parameters.
        """
        pass

    @abstractmethod
    def cart_coordinates(self):
        """
        Calculate Cartesian coordinates for mapping.
        """
        pass

    @abstractmethod
    def map_to_sphere(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, yaw_radian: float, pitch_radian: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map Cartesian coordinates (x, y, z) to spherical coordinates.
        """
        pass

    @abstractmethod
    def sliding_windows(self, patch_size: int, overlap: float) -> Iterator[Tuple[int, int, np.ndarray]]:
        """
        Generate sliding window patches from the image.
        """
        pass

    @abstractmethod
    def video_conversion(self, output_file: str):
        """
        Convert the spherical mapped images into a video.
        """
        pass

    @abstractmethod
    def interpolate_color(self, coords: np.ndarray, method: str = 'bilinear') -> np.ndarray:
        """
        Interpolate colors for the image based on the given coordinates.
        """
        pass

    @abstractmethod
    def compute_patches(self) -> Tuple[int, int, int]:
        """
        Compute the number of patches and the stride needed based on the image size and overlap.
        """
        pass

    @abstractmethod
    def decomposition(self) -> List[np.ndarray]:
        """
        Decompose the panorama image into overlapping patches and save them.
        """
        pass

