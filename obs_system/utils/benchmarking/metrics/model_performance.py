from obs_system.utils.benchmarking.interface.benchmark import BenchMark
from obs_system.utils.benchmarking.interface.utils import _iou_xyxy

import numpy as np 
from typing import Dict, Optional, Iterable, Tuple, List


class ModelPerf(BenchMark): 
    

    def __init__(self, class_ids:Optional[Iterable[int]] = None, 
                 iou_thresholds:Iterable[float] = (0.5, ), 
                 conf_threshold: float = 0.5, 
                 use_101_point_interp: bool = True, 
        ) -> None : 
        """
        Args:
            class_ids: list of known class ids (optional; used for per-class coverage).
            iou_thresholds: IoU thresholds for AP (e.g., (0.5,) or (0.50,0.55,...,0.95)).
            conf_threshold: score threshold for P/R/F1 tally.
            use_101_point_interp: if True, AP via 101-point interpolation (COCO style).
                                   else trapezoidal integration over precision-recall curve.
        """

        self._all_classes = set(class_ids) if class_ids is not None else set() 
        self._ious = np.array(sorted(iou_thresholds), dtype=np.float32) 
        self._conf_thr = float(conf_threshold) 
        self._interp_101 = bool(use_101_point_interp) 

        self._image_index = 0 
        self._det_by_class:Dict[int, List[Tuple[float, int, np.ndarray]]] = {} 
        self._GT_per_image_class: Dict[int, Dict[int, List[np.ndarray]]] = {} 

        self._prec_tally = 0 
        self._recall_tally = 0 
        self._f1_tally = 0 

        self._results:Dict[str, float]={} 
        self._fn_micro = 0 
        self._fp_micro = 0 
        self._tp_micro = 0


    def update(
            self, 
            boxes_xyxy:np.ndarray, 
            scores:np.ndarray, 
            classes: np.ndarray, 
            gt_boxes_xyxy: np.ndarray, 
            gt_classes: np.ndarray, 

        ) -> None : 

        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        classes = np.asarray(classes, dtype=np.int64).reshape(-1)
        gt_boxes_xyxy = np.asarray(gt_boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        gt_classes = np.asarray(gt_classes, dtype=np.int64).reshape(-1)

        # Track class universe (optional)
        self._all_classes.update(classes.tolist())
        self._all_classes.update(gt_classes.tolist())

        # Store GT by image/class
        img_gt: Dict[int, List[np.ndarray]] = {}
        for c in gt_classes:
            img_gt.setdefault(int(c), []).append(None)

        # Fill boxes in same pass
        idx_map = {i: int(c) for i, c in enumerate(gt_classes)}
        per_cls_gts: Dict[int, List[np.ndarray]] = {}
        for i in range(gt_boxes_xyxy.shape[0]):
            c = idx_map[i]
            per_cls_gts.setdefault(c, []).append(gt_boxes_xyxy[i])
        self._GT_per_image_class[self._image_index] = per_cls_gts
        
        # Store detections per class (score, img_idx, box)
        for i in range(boxes_xyxy.shape[0]):
            c = int(classes[i])
            self._det_by_class.setdefault(c, []).append((float(scores[i]), self._image_index, boxes_xyxy[i]))

        # Micro P/R/F1 at fixed threshold (fast pass)
        self._accumulate_pr_fixed_threshold(
            boxes_xyxy, scores, classes, gt_boxes_xyxy, gt_classes, self._conf_thr
        )

        self._image_index += 1


    def finalize(self)->None: 
        ap_by_iou: List[float]=[] 
        for iou_thr in self._ious:
            ap_c = []
            for cls in sorted(self._all_classes):
                ap = self._ap_for_class(cls, float(iou_thr))
                if ap is not None:
                    ap_c.append(ap)
            if len(ap_c):
                ap_by_iou.append(float(np.mean(ap_c)))
        mAP = float(np.mean(ap_by_iou)) if len(ap_by_iou) else 0.0

        # Micro PRF1
        precision = self._tp_micro / max(1, (self._tp_micro + self._fp_micro))
        recall = self._tp_micro / max(1, (self._tp_micro + self._fn_micro))
        f1 = 2 * precision * recall / max(1e-9, (precision + recall))

        # Store results
        self._results = {
            "mAP": mAP,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }

        # (Optionally) expose per-IoU mAP as well
        for i, iou_thr in enumerate(self._ious):
            key = f"mAP@{iou_thr:.2f}"
            self._results[key] = ap_by_iou[i] if i < len(ap_by_iou) else 0.0


    def results(self)->Dict[str,float]: 
        return dict(self._results) 


    def reset(self) -> None:
        self._all_classes.clear()
        self._image_index = 0
        self._det_by_class.clear()
        self._GT_per_image_class.clear()
        self._tp_micro = self._fp_micro = self._fn_micro = 0
        self._results = {}

    def _accumulate_pr_fixed_threshold(
        self,
        det_boxes: np.ndarray,
        det_scores: np.ndarray,
        det_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        conf_thr: float,
    ) -> None:
        """Computes micro TP/FP/FN at a fixed confidence threshold (IoU=0.5 default for this pass)."""
        # Filter detections by threshold
        keep = det_scores >= conf_thr
        det_boxes = det_boxes[keep]
        det_scores = det_scores[keep]
        det_classes = det_classes[keep]

        if gt_boxes.size == 0 and det_boxes.size == 0:
            return
        if gt_boxes.size == 0 and det_boxes.size > 0:
            self._fp_micro += det_boxes.shape[0]
            return
        if gt_boxes.size > 0 and det_boxes.size == 0:
            self._fn_micro += gt_boxes.shape[0]
            return

        # Match per class (IoU=0.5 for P/R/F1 fast path)
        iou_thr = 0.5
        for cls in np.unique(np.concatenate([det_classes, gt_classes], axis=0)):
            dmask = det_classes == cls
            gmask = gt_classes == cls
            d = det_boxes[dmask]
            g = gt_boxes[gmask]

            if d.size == 0:
                self._fn_micro += g.shape[0]
                continue
            if g.size == 0:
                self._fp_micro += d.shape[0]
                continue

            ious = _iou_xyxy(d, g)
            # greedy matching by score is better, but this fast path is fine:
            # count matches with IoU>=thr, ensuring each GT is matched at most once
            gt_matched = np.zeros(g.shape[0], dtype=bool)
            tp = 0
            for i in range(d.shape[0]):
                j = np.argmax(ious[i])
                if ious[i, j] >= iou_thr and not gt_matched[j]:
                    tp += 1
                    gt_matched[j] = True
                else:
                    self._fp_micro += 1
            self._tp_micro += tp
            self._fn_micro += int((~gt_matched).sum())

        
    def _ap_for_class(self, cls:int, iou_thr:float)->Optional[float]: 
        dets = self._det_by_class.get(cls, [])
        if len(dets) == 0:
            # If class exists in GT, AP=0; if not present at all, we ignore it by returning None
            # Decide based on presence in GT:
            present_in_gt = any(cls in g for g in self._GT_per_image_class.values())
            return 0.0 if present_in_gt else None

        # Sort detections by score (desc)
        dets.sort(key=lambda t: t[0], reverse=True)

        # Build GT structures: for each image, list of boxes + matched flags
        gt_by_img = self._GT_per_image_class
        gt_boxes_per_img = {}
        gt_used_per_img = {}
        any_gt = False
        for img_idx, per_cls in gt_by_img.items():
            g = np.array(per_cls.get(cls, []), dtype=np.float32).reshape(-1, 4)
            gt_boxes_per_img[img_idx] = g
            gt_used_per_img[img_idx] = np.zeros(g.shape[0], dtype=bool)
            if g.shape[0] > 0:
                any_gt = True

        if not any_gt:
            # No GT for this class anywhere → AP is undefined; ignore class
            return None

        tps = np.zeros(len(dets), dtype=np.float32)
        fps = np.zeros(len(dets), dtype=np.float32)

        # Greedy match by score
        for i, (score, img_idx, box) in enumerate(dets):
            g = gt_boxes_per_img.get(img_idx, np.zeros((0, 4), dtype=np.float32))
            if g.shape[0] == 0:
                fps[i] = 1.0
                continue
            ious = _iou_xyxy(np.expand_dims(box.astype(np.float32), 0), g).reshape(-1)
            j = int(np.argmax(ious)) if ious.size else -1
            if j >= 0 and ious[j] >= iou_thr and not gt_used_per_img[img_idx][j]:
                tps[i] = 1.0
                gt_used_per_img[img_idx][j] = True
            else:
                fps[i] = 1.0

        # Cumulate
        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)

        # Precision-Recall
        total_gt = sum(b.shape[0] for b in gt_boxes_per_img.values())
        if total_gt == 0:
            return None

        recall = cum_tp / max(1, total_gt)
        precision = cum_tp / np.maximum(1, (cum_tp + cum_fp))

        # AP
        if self._interp_101:
            # COCO-style 101-point interpolation
            ap = 0.0
            for r in np.linspace(0, 1, 101):
                p = precision[recall >= r].max() if np.any(recall >= r) else 0.0
                ap += p
            ap /= 101.0
        else:
            # Trapezoidal area under PR (with monotonic precision)
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))
            # make precision monotonically decreasing
            for i in range(mpre.size - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

        return float(ap)
















