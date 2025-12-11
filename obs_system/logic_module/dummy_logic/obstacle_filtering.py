from obs_system.utils.global_config import VOCAB, RED
import numpy as np 
import torch
import cv2
import pdb
from typing import List
from collections import defaultdict


def get_class_name(class_id, classes): 
    return classes[class_id]

def is_non_permitted_vehicle(obj_label): 
    return True if obj_label in VOCAB else False

def location_checker(obs_bbox, lanes_final, orig_shape, overlap_threshold=0.2): 

    x_min, y_min, x_max, y_max = obs_bbox
    orig_height, orig_width, _ = orig_shape 

    w = (x_max - x_min)
    h = (y_max - y_min)
    area =  w * h

    lane_mask = np.zeros((orig_height, orig_width), dtype=np.uint8) 
    cv2.drawContours(lane_mask, [lanes_final], 0, 255, -1)

    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min)) 
    x_max = max(orig_width, int(x_max))
    y_max = max(orig_height, int(y_max))

    bbox_region_in_lane = lane_mask[y_min:y_max, x_min:x_max]
    overlap_pixel_count = cv2.countNonZero(bbox_region_in_lane)

    if area > 0:
        overlap_ratio = overlap_pixel_count / area
        return overlap_ratio >= overlap_threshold
    else:
        # Bounding box has zero area (invalid), so no overlap
        return False 


def classification_obstacles(boxes, classes, lanes_final, orig_shape, orig_classes: list): 
    labels = [] 
    basic_classes = orig_classes
    for (box, cls) in (zip(boxes, classes)): 
        if not is_non_permitted_vehicle(cls): 
            labels.append(cls)

            continue

        is_inside_lane = location_checker(box, lanes_final, orig_shape)

        if not is_inside_lane: 
            labels.append(cls)
            continue

        new_label = f"Lane_Obstacle_{cls}"
        if new_label not in basic_classes: 
            basic_classes.append(new_label)

        labels.append(new_label) 
    return torch.tensor(labels), basic_classes
       

def draw_obstacles():pass
