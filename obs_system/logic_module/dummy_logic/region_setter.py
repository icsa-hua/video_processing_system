from pathlib import Path

from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
import numpy as np
import pdb
import cv2
import torch 
from typing import Any
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import platform
from ultralytics.utils.ops import scale_boxes
from obs_system.utils import global_config

class RegionSetter(EventExtractorInterface):

    def __init__(self, capture_dir=None) -> None:
        self.regions = []
        self.x_start = 0
        self.x_end = 0
        self.y_start = 0
        self.y_end = 0
        self.current_region = None
        self._capture_dir: Path | None = Path(capture_dir) if capture_dir is not None else None
        self._save_call_count: int = 0
        self._save_captured: int = 0
        self._save_max: int = 10
        self._save_spacing: int = 50

    def detect(self, predictions):
        return super().detect(predictions)


    def set_regions(self, image) -> list: 
        if image is None or not hasattr(image, "shape"):
            raise ValueError("A valid image is required to initialize ROI regions.")
        
        # This should be modified base on the camera feed, cannot be applied to all camera feeds. 
        # NOTE: ROI in this case considers the road network exclusively. 
        original_height, original_width = image.shape[:2]

        roi_config = global_config.get_active_roi_config()
        reference_width = max(1, int(roi_config["reference_width"]))
        reference_height = max(1, int(roi_config["reference_height"]))

        aspect_ratio_width = original_width / reference_width
        aspect_ratio_height = original_height / reference_height

        x_start = int(roi_config["x1"] * aspect_ratio_width)
        x_end = int(roi_config["x2"] * aspect_ratio_width)
        y_start = int(roi_config["y1"] * aspect_ratio_height)
        y_end = int(roi_config["y2"] * aspect_ratio_height)
        
        x_start, x_end = sorted([x_start, x_end])
        y_start, y_end = sorted([y_start, y_end])

        self.x_start = max(0, min(x_start, original_width))
        self.x_end = max(0, min(x_end, original_width))
        self.y_start = max(0, min(y_start, original_height))
        self.y_end = max(0, min(y_end, original_height))

        if self.x_end <= self.x_start or self.y_end <= self.y_start:
            self.x_start, self.y_start = 0, 0
            self.x_end, self.y_end = original_width, original_height

        self.regions = [{
                    "name": "Road Polygon Region",
                    "polygon": Polygon([
                        (self.x_start, self.y_start),
                        (self.x_end, self.y_start),
                        (self.x_end, self.y_end),
                        (self.x_start, self.y_end),
                    ]),
                    "counts": 0,
                    "dragging": False,
                    "region_color": global_config.REGION_COLOR,  # BGR Value
                    "text_color": (255, 255, 255),  # Region Text Color
                },
        ]
        return self.regions
    

    def translate_bounding_boxes(self, results, orig_img_shape, input_img_shape=None, crop_img_shape=None):
        x0, y0 = self.x_start, self.y_start
        H, W = orig_img_shape

        xyxy = results.clone() if isinstance(results, torch.Tensor) else np.array(results, copy=True)

        if input_img_shape is not None and crop_img_shape is not None and len(xyxy):
            xyxy = scale_boxes(input_img_shape, xyxy, crop_img_shape)

        xyxy[:, [0, 2]] += x0
        xyxy[:, [1, 3]] += y0

        if isinstance(xyxy, torch.Tensor):
            xyxy[:, [0, 2]].clamp_(0, W)
            xyxy[:, [1, 3]].clamp_(0, H)
        else:
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, W)
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, H)

        return xyxy


    def crop_image(self, images):
        def _crop_one(image):
            if image is None:
                raise ValueError("No image to crop in crop_image method of RegionSetter.")
            h, w = image.shape[:2]
            x0 = max(0, min(self.x_start, w))
            x1 = max(0, min(self.x_end, w))
            y0 = max(0, min(self.y_start, h))
            y1 = max(0, min(self.y_end, h))
            if x1 <= x0 or y1 <= y0:
                return image
            return image[y0:y1, x0:x1]

        if isinstance(images, (list, tuple)):
            result = [_crop_one(image) for image in images]
            if images:
                self._maybe_save_crop(images[0], result[0])
            return result
        if isinstance(images, np.ndarray):
            result = _crop_one(images)
            self._maybe_save_crop(images, result)
            return result

        raise ValueError("No images to crop in crop_image method of RegionSetter.")

    def _maybe_save_crop(self, original: np.ndarray, cropped: np.ndarray) -> None:
        if self._capture_dir is None:
            return
        self._save_call_count += 1
        if not (self._save_call_count == 1 or self._save_call_count % self._save_spacing == 0):
            return
        if self._save_captured >= self._save_max:
            return
        self._save_captured += 1
        idx = self._save_captured
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self._capture_dir / f"s1_raw_{idx:03d}.png"), original)
        cv2.imwrite(str(self._capture_dir / f"s4_roi_{idx:03d}.png"), cropped)


    def count_regions(self, bbox) -> None:
        x = float(bbox[0])
        y = float(bbox[1])
        if self.x_start <= x <= self.x_end and self.y_start <= y <= self.y_end:
            for region in self.regions:
                region["counts"] += 1


    def mouse_callback(self, event:int, x:int, y:int, flags:int, param:Any)->None:
        # Mouse left button down event
        if event == cv2.EVENT_LBUTTONDOWN:
            for region in self.regions:
                if region["polygon"].contains(Point((x, y))):
                    self.current_region = region
                    self.current_region["dragging"] = True
                    self.current_region["offset_x"] = x
                    self.current_region["offset_y"] = y

        # Mouse move event
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.current_region is not None and self.current_region["dragging"]:
                dx = x - self.current_region["offset_x"]
                dy = y - self.current_region["offset_y"]
                self.current_region["polygon"] = Polygon(
                    [(p[0] + dx, p[1] + dy) for p in self.current_region["polygon"].exterior.coords]
                )
                self.current_region["offset_x"] = x
                self.current_region["offset_y"] = y

        # Mouse left button up event
        elif event == cv2.EVENT_LBUTTONUP:
            if self.current_region is not None and self.current_region["dragging"]:
                self.current_region["dragging"] = False


    def _show_regions(self, im:np.ndarray)->None:
        for region in self.regions:
            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            region_color = (0, 0, 0)
            region_label = "ROI-Inference"
            region_text_color = (255, 255, 255)
            x, y, w, h = cv2.boundingRect(polygon_coords)
            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
            )
            text_x = max(x, 0)
            text_y = max(y - 10, text_size[1] + 6)

            cv2.rectangle(
                im,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(im, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2)
            cv2.polylines(im, [polygon_coords], isClosed=True, color=region_color, thickness=2)
        
        # cv2.imshow("Regions", im)
        # cv2.waitKey(1)
