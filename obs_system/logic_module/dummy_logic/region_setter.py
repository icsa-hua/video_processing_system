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
from obs_system.utils.global_config import TILE_SIZE, ROI_X1, ROI_Y1, ROI_X2, ROI_Y2, REGION_COLOR

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

        new_width = TILE_SIZE
        new_height = TILE_SIZE

        aspect_ratio_width = original_width / new_width
        aspect_ratio_height = original_height / new_height

        # NOTE: Change these values based on the camera feed.
        x_start = int(ROI_X1 * aspect_ratio_width)
        x_end = int(ROI_X2 * aspect_ratio_width)
        y_start = int(ROI_Y1 * aspect_ratio_height)
        y_end = int(ROI_Y2 * aspect_ratio_height)
        
        self.x_start, self.x_end = sorted([x_start, x_end])
        self.y_start, self.y_end = sorted([y_start, y_end])

        return [{
                    "name": "Road Polygon Region",
                    "polygon": Polygon([(x_start, y_start), (x_end, y_start), (x_end, y_end), (x_start, y_end)]),  # Polygon points
                    "counts": 0,
                    "dragging": False,
                    "region_color": REGION_COLOR,  # BGR Value
                    "text_color": (255, 255, 255),  # Region Text Color
                },
        ]
    

    def translate_bounding_boxes(self, results,  # results[0]
                         orig_img_shape,      # (H, W) of original, e.g. (1080, 1920)
                         crop_shape,          # (h, w) of crop BEFORE letterbox, e.g. (572, 1290)
                         lb_shape=(640, 640)  # letterboxed image shape given to model
                         ):
        x0, y0 = self.x_start, self.y_start
        H, W = orig_img_shape

        # 1) bring boxes from 640-letterboxed coords back to crop coords (572x1290)
        xyxy = results
        # This is necessary if the detections are in the 640x640 format
        # h0, w0 = crop_shape
        # xyxy = scale_boxes(lb_shape, xyxy, (h0, w0))

        # 2) add crop offset -> original image coords
        xyxy[:, [0, 2]] += x0
        xyxy[:, [1, 3]] += y0

        # optional clamp
        xyxy[:, [0, 2]].clamp_(0, W)
        xyxy[:, [1, 3]].clamp_(0, H)

        return xyxy


    def crop_image(self, images):
        
        if len(images) > 1 and isinstance(images, list): 
            return [image[self.y_start:self.y_end, self.x_start:self.x_end] for image in images]
        elif isinstance(images, np.ndarray): 
            return images[self.y_start:self.y_end, self.x_start:self.x_end]
        else: 
            raise ValueError("No images to crop in crop_image method of RegionSetter.")


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
        
        cv2.imshow("Regions", im)
        cv2.waitKey(1)
