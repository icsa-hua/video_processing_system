from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
import numpy as np
import cv2
from typing import Any
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import platform 

class RegionSetter(EventExtractorInterface):

    def __init__(self) -> None:
        self.regions = []
        self.x_start = 0
        self.x_end = 0
        self.y_start = 0
        self.y_end = 0 
        self.current_region = None 

    def detect(self, predictions):
        return super().detect(predictions)


    def set_regions(self, image) -> list: 
        
        # This should be modified base on the camera feed, cannot be applied to all camera feeds. 
        # NOTE: ROI in this case considers the road network exclusively. 
        original_height, original_width = image.shape[:2]

        new_width = 640 
        new_height = 640

        aspect_ratio_width = original_width / new_width
        aspect_ratio_height = original_height / new_height

        # NOTE: Change these values based on the camera feed.
        x_start = int(0 * aspect_ratio_width)
        x_end = int(430 * aspect_ratio_width)
        y_start = int(639 * aspect_ratio_height)
        y_end = int(300 * aspect_ratio_height)

        
        self.x_start, self.x_end = sorted([x_start, x_end])
        self.y_start, self.y_end = sorted([y_start, y_end])

        return [{
                    "name": "Road Polygon Region",
                    "polygon": Polygon([(x_start, y_start), (x_end, y_start), (x_end, y_end), (x_start, y_end)]),  # Polygon points
                    "counts": 0,
                    "dragging": False,
                    "region_color": (255, 42, 4),  # BGR Value
                    "text_color": (255, 255, 255),  # Region Text Color
                },
        ]
    

    def crop_image(self, images):
        return [image[self.y_start:self.y_end, self.x_start:self.x_end] for image in images]



    def count_regions(self, bbox) -> None:
        for region in self.regions:
            if region["polygon"].contains(Point((bbox[0], bbox[1]))):
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
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]
            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)
            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2

            cv2.rectangle(im,(text_x - 5, text_y - text_size[1] - 5),(text_x + text_size[0] + 5, text_y + 5),region_color,-1,)
            cv2.putText(im, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2)
            cv2.polylines(im, [polygon_coords], isClosed=True, color=region_color, thickness=2)
        
