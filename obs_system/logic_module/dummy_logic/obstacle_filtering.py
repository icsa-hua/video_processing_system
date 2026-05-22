import cv2
import numpy as np
import torch

from typing import Any, Dict, List, Optional, Sequence, Tuple


ALLOWED_VEHICLES = {
    "car",
    "truck",
    "bus",
    "bike",
    "bicycle",
    "motorbike",
    "motorcycle",
}

ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

DEBRIS_CLASSES = {
    "backpack",
    "suitcase",
    "handbag",
    "bottle",
    "chair",
    "bench",
    "tire",
    "box",
}

NORMALIZED_ALLOWED_VEHICLES = {v.strip().lower().replace("-", "").replace("_", "").replace(" ", "") for v in ALLOWED_VEHICLES}
NORMALIZED_ANIMAL_CLASSES = {v.strip().lower().replace("-", "").replace("_", "").replace(" ", "") for v in ANIMAL_CLASSES}
NORMALIZED_DEBRIS_CLASSES = {v.strip().lower().replace("-", "").replace("_", "").replace(" ", "") for v in DEBRIS_CLASSES}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _resolve_class_name(class_value: Any, class_names: Sequence[str]) -> str:
    if isinstance(class_value, (int, np.integer)):
        idx = int(class_value)
        return class_names[idx] if 0 <= idx < len(class_names) else f"class_{idx}"
    if isinstance(class_value, (float, np.floating)):
        idx = int(class_value)
        return class_names[idx] if 0 <= idx < len(class_names) else f"class_{idx}"
    if torch.is_tensor(class_value):
        if class_value.numel() == 0:
            return "unknown"
        idx = int(class_value.item())
        return class_names[idx] if 0 <= idx < len(class_names) else f"class_{idx}"
    return str(class_value)


def _clip_box_xyxy(box: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.tolist()
    x1 = int(np.clip(np.floor(min(x1, x2)), 0, max(w - 1, 0)))
    y1 = int(np.clip(np.floor(min(y1, y2)), 0, max(h - 1, 0)))
    x2 = int(np.clip(np.ceil(max(x1, x2)), 0, w))
    y2 = int(np.clip(np.ceil(max(y1, y2)), 0, h))
    return x1, y1, x2, y2


def _box_xyxy_int(box: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.tolist()
    return int(np.floor(min(x1, x2))), int(np.floor(min(y1, y2))), int(np.ceil(max(x1, x2))), int(np.ceil(max(y1, y2)))


def _mask_integral(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None or mask.size == 0:
        return None
    binary = (mask > 0).astype(np.uint8)
    return cv2.integral(binary)


def _overlap_ratio(
    mask: Optional[np.ndarray],
    bbox_xyxy: Tuple[int, int, int, int],
    *,
    integral: Optional[np.ndarray] = None,
) -> float:
    if mask is None or mask.size == 0:
        return 0.0

    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        return 0.0

    area = float((x2 - x1) * (y2 - y1))
    if area <= 0:
        return 0.0
    if integral is not None:
        count = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        return float(count) / area
    region = mask[y1:y2, x1:x2]
    return float(cv2.countNonZero(region)) / area


def _lane_bounds(lane_mask: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    if lane_mask is None or lane_mask.size == 0:
        return None
    ys, xs = np.where(lane_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _risk_from_position(y_center: float, lane_bbox: Optional[Tuple[int, int, int, int]]) -> str:
    if lane_bbox is None:
        return "medium"
    _, y1, _, y2 = lane_bbox
    span = max(float(y2 - y1), 1.0)
    pos = float(y_center - y1) / span
    if pos >= 0.70:
        return "high"
    if pos >= 0.40:
        return "medium"
    return "low"


def _categorize_hazard(
    class_name: str,
    risk_level: str,
    box_area_ratio: float,
    near_lane_edge: bool,
) -> Tuple[str, str]:
    n = _normalize_name(class_name)
    if n == "person":
        return "pedestrian_in_lane", "urgent_mqtt|audio_alert|continuous_tracking|store_clip"
    if n in NORMALIZED_ANIMAL_CLASSES:
        return "animal_on_road", "medium_alert|tracking|store_event"
    if n in NORMALIZED_DEBRIS_CLASSES:
        if box_area_ratio < 0.01:
            return "small_debris", "low_alert|verify_persistence|log_event"
        return "debris_or_fallen_object", "medium_alert|tracking|store_event"
    if near_lane_edge and box_area_ratio < 0.003:
        return "edge_small_object", "low_alert|verify_persistence|log_event"
    if risk_level == "high":
        return "large_static_object", "high_alert|flag_obstruction|store_clip|repeat_until_cleared"
    return "unknown_obstruction", "medium_alert|tracking|store_event"


def analyze_lane_hazards(
    boxes: Any,
    classes: Any,
    class_names: Sequence[str],
    lane_mask: Optional[np.ndarray],
    crosswalk_mask: Optional[np.ndarray],
    returned_boxes: Any = None,
    lane_bbox: Optional[Tuple[int, int, int, int]] = None,
    lane_integral: Optional[np.ndarray] = None,
    crosswalk_integral: Optional[np.ndarray] = None,
    lane_nonzero: Optional[bool] = None,
    crosswalk_nonzero: Optional[bool] = None,
    lane_overlap_threshold: float = 0.20,
    crosswalk_overlap_threshold: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Uses post-NMS detections + lane/crosswalk masks to produce hazard metadata.
    Vectorised: integral-image lookups for all boxes in a single numpy op,
    Python loop only over the (typically small) in-lane subset.
    """
    if boxes is None or classes is None:
        return []
    lane_has_pixels = lane_nonzero if lane_nonzero is not None else (
        lane_mask is not None and lane_mask.size > 0 and cv2.countNonZero(lane_mask) > 0
    )
    if lane_mask is None or lane_mask.size == 0 or not lane_has_pixels:
        return []

    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.asarray(boxes, dtype=np.float32)

    if torch.is_tensor(classes):
        cls_np = classes.detach().cpu().numpy()
    else:
        cls_np = np.asarray(classes)

    if returned_boxes is None:
        returned_boxes_np = boxes_np
    elif torch.is_tensor(returned_boxes):
        returned_boxes_np = returned_boxes.detach().cpu().numpy()
    else:
        returned_boxes_np = np.asarray(returned_boxes, dtype=np.float32)

    n = min(len(boxes_np), len(cls_np))
    if n == 0 or boxes_np.size == 0 or returned_boxes_np.size == 0:
        return []

    h, w = lane_mask.shape[:2]
    raw = boxes_np[:n]

    # Vectorised box clipping – avoids per-box Python .tolist() calls
    bx1 = np.clip(np.floor(np.minimum(raw[:, 0], raw[:, 2])), 0, max(w - 1, 0)).astype(np.int32)
    by1 = np.clip(np.floor(np.minimum(raw[:, 1], raw[:, 3])), 0, max(h - 1, 0)).astype(np.int32)
    bx2 = np.clip(np.ceil(np.maximum(raw[:, 0], raw[:, 2])), 0, w).astype(np.int32)
    by2 = np.clip(np.ceil(np.maximum(raw[:, 1], raw[:, 3])), 0, h).astype(np.int32)
    valid = (bx2 > bx1) & (by2 > by1)
    areas = ((bx2 - bx1) * (by2 - by1)).astype(np.float64)

    # Vectorised lane overlap using integral image fancy indexing
    if lane_integral is not None and valid.any():
        ih, iw = lane_integral.shape
        ix1 = np.clip(bx1, 0, iw - 1)
        iy1 = np.clip(by1, 0, ih - 1)
        ix2 = np.clip(bx2, 0, iw - 1)
        iy2 = np.clip(by2, 0, ih - 1)
        counts = (
            lane_integral[iy2, ix2]
            - lane_integral[iy1, ix2]
            - lane_integral[iy2, ix1]
            + lane_integral[iy1, ix1]
        ).astype(np.float64)
        lane_overlaps = np.where(areas > 0, counts / areas, 0.0)
    else:
        lane_overlaps = np.zeros(n, dtype=np.float64)
        for i in np.where(valid)[0]:
            region = lane_mask[by1[i]:by2[i], bx1[i]:bx2[i]]
            lane_overlaps[i] = float(cv2.countNonZero(region)) / areas[i]

    in_lane_mask = valid & (lane_overlaps >= lane_overlap_threshold)
    if not in_lane_mask.any():
        return []

    # Vectorised crosswalk overlap for person candidates in-lane
    crosswalk_available = (
        crosswalk_mask is not None
        and crosswalk_mask.size > 0
        and (crosswalk_nonzero if crosswalk_nonzero is not None else cv2.countNonZero(crosswalk_mask) > 0)
    )
    cross_overlaps = np.zeros(n, dtype=np.float64)
    if crosswalk_available and crosswalk_integral is not None and in_lane_mask.any():
        ih, iw = crosswalk_integral.shape
        ix1 = np.clip(bx1, 0, iw - 1)
        iy1 = np.clip(by1, 0, ih - 1)
        ix2 = np.clip(bx2, 0, iw - 1)
        iy2 = np.clip(by2, 0, ih - 1)
        cw_counts = (
            crosswalk_integral[iy2, ix2]
            - crosswalk_integral[iy1, ix2]
            - crosswalk_integral[iy2, ix1]
            + crosswalk_integral[iy1, ix1]
        ).astype(np.float64)
        cross_overlaps = np.where(areas > 0, cw_counts / areas, 0.0)

    lane_bbox = lane_bbox if lane_bbox is not None else _lane_bounds(lane_mask)
    frame_area = float(max(w * h, 1))
    hazards: List[Dict[str, Any]] = []

    # Inner loop only over in-lane boxes (small subset after vectorised filter)
    for i in np.where(in_lane_mask)[0]:
        class_name = _resolve_class_name(cls_np[i], class_names)
        name_norm = _normalize_name(class_name)

        if name_norm in NORMALIZED_ALLOWED_VEHICLES:
            continue

        lane_overlap = float(lane_overlaps[i])
        cross_overlap = float(cross_overlaps[i])
        is_person = name_norm == "person"
        in_crosswalk = is_person and crosswalk_available and cross_overlap >= crosswalk_overlap_threshold
        if in_crosswalk:
            continue

        x1, y1, x2, y2 = int(bx1[i]), int(by1[i]), int(bx2[i]), int(by2[i])

        rb = returned_boxes_np[i]
        rx1 = int(np.floor(min(float(rb[0]), float(rb[2]))))
        ry1 = int(np.floor(min(float(rb[1]), float(rb[3]))))
        rx2 = int(np.ceil(max(float(rb[0]), float(rb[2]))))
        ry2 = int(np.ceil(max(float(rb[1]), float(rb[3]))))

        y_center = 0.5 * (y1 + y2)
        risk = _risk_from_position(y_center=y_center, lane_bbox=lane_bbox)
        lane_left, _, lane_right, _ = lane_bbox if lane_bbox is not None else (0, 0, w, h)
        x_center = 0.5 * (x1 + x2)
        lane_w = max(float(lane_right - lane_left), 1.0)
        near_edge = (x_center - lane_left) / lane_w < 0.15 or (lane_right - x_center) / lane_w < 0.15

        area_ratio = float((x2 - x1) * (y2 - y1)) / frame_area
        category, action = _categorize_hazard(class_name, risk, area_ratio, near_edge)
        kind = "Person in Lane" if is_person else "Object in Path"

        hazards.append(
            {
                "det_index": int(i),
                "class_name": class_name,
                "class_id": int(cls_np[i]) if np.isscalar(cls_np[i]) else -1,
                "bbox_xyxy": [rx1, ry1, rx2, ry2],
                "lane_overlap": lane_overlap,
                "crosswalk_overlap": cross_overlap,
                "risk": risk,
                "category": category,
                "action": action,
                "kind": kind,
                "message": f"{kind} ({risk.upper()})",
                "in_crosswalk": bool(in_crosswalk),
                "in_lane": True,
            }
        )

    return hazards


def classification_obstacles(boxes, classes, lanes_final, orig_shape, orig_classes: list):
    """
    Backward-compatible shim.
    Keeps original class IDs unchanged.
    """
    if torch.is_tensor(classes):
        return classes, list(orig_classes)
    return torch.tensor(classes), list(orig_classes)
