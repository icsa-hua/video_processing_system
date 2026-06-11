"""
Kalman-filter detection smoother for tiled and panorama inference paths.

After cross-tile / cross-view NMS produces the merged detection set for a frame,
this module maintains constant-velocity Kalman tracks across frames and augments
the set with predicted boxes for confirmed tracks that YOLO missed (e.g. objects
near tile boundaries or in low-motion views).  The smoother does NOT assign
persistent IDs — that remains ByteTracker's job in Stage C.

Ported and adapted from CarDet_Dummy_EdgeAI/src/tracker.py (SORT algorithm).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Kalman geometry helpers
# ---------------------------------------------------------------------------

def _xyxy_to_z(box: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] → measurement column vector [cx, cy, area, aspect]."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w * 0.5
    cy = box[1] + h * 0.5
    s = max(float(w * h), 1.0)
    r = float(w) / max(float(h), 1e-6)
    return np.array([cx, cy, s, r], dtype=np.float32).reshape(4, 1)


def _x_to_xyxy(x: np.ndarray) -> np.ndarray:
    """State [cx,cy,s,r,...] → [x1,y1,x2,y2]."""
    cx, cy = float(x[0, 0]), float(x[1, 0])
    s = max(float(x[2, 0]), 1.0)
    r = max(float(x[3, 0]), 1e-3)
    w = float(np.sqrt(s * r))
    h = s / w
    return np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32)


def _iou_pair(a: np.ndarray, b: np.ndarray) -> float:
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter == 0.0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-6)


def _iou_matrix(det_boxes: np.ndarray, trk_boxes: np.ndarray) -> np.ndarray:
    D, T = len(det_boxes), len(trk_boxes)
    mat = np.zeros((D, T), dtype=np.float32)
    for d in range(D):
        for t in range(T):
            mat[d, t] = _iou_pair(det_boxes[d], trk_boxes[t])
    return mat


# ---------------------------------------------------------------------------
# Single-box Kalman filter (classic 7-dim SORT state)
# ---------------------------------------------------------------------------

class _KalmanBox:
    """
    7-dim constant-velocity Kalman filter for one bounding box.
    State: [cx, cy, area, aspect, vcx, vcy, v_area].
    """

    def __init__(self, box: np.ndarray, score: float, cls: int) -> None:
        ndim = 7
        self.F = np.eye(ndim, dtype=np.float32)
        for i in range(3):                   # cx, cy, area get velocity coupling
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, ndim), dtype=np.float32)
        self.H[:4, :4] = np.eye(4)

        self.P = np.eye(ndim, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 1000.0            # high initial velocity uncertainty
        self.Q = np.eye(ndim, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01
        self.R = np.eye(4, dtype=np.float32)
        self.R[2:, 2:] *= 10.0              # area/aspect measurement noisier

        self.x = np.zeros((ndim, 1), dtype=np.float32)
        self.x[:4] = _xyxy_to_z(box)

        self.time_since_update: int = 0
        self.hits: int = 1
        self.age: int = 0
        self.score: float = float(score)
        self.cls: int = int(cls)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.x[2, 0] <= 0:
            self.x[2, 0] = 1.0
        self.age += 1
        self.time_since_update += 1
        return _x_to_xyxy(self.x)

    def update(self, box: np.ndarray, score: float, cls: int) -> None:
        z = _xyxy_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0], dtype=np.float32) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.score = 0.6 * self.score + 0.4 * float(score)
        self.cls = int(cls)

    @property
    def xyxy(self) -> np.ndarray:
        return _x_to_xyxy(self.x)


# ---------------------------------------------------------------------------
# Hungarian / greedy assignment
# ---------------------------------------------------------------------------

def _associate(
    trk_boxes: np.ndarray,
    det_boxes: np.ndarray,
    iou_thr: float,
) -> tuple:
    """Returns (matches, unmatched_det_indices, unmatched_trk_indices)."""
    D, T = len(det_boxes), len(trk_boxes)
    if D == 0 or T == 0:
        return [], list(range(D)), list(range(T))

    iou = _iou_matrix(det_boxes, trk_boxes)

    if _HAS_SCIPY:
        rows, cols = linear_sum_assignment(-iou)  # type: ignore[possibly-unbound]
        cands = list(zip(rows.tolist(), cols.tolist()))
    else:
        cands = []
        used_d: set = set()
        used_t: set = set()
        for d, t in sorted(
            ((d, t) for d in range(D) for t in range(T)),
            key=lambda p: -iou[p[0], p[1]],
        ):
            if d in used_d or t in used_t:
                continue
            used_d.add(d)
            used_t.add(t)
            cands.append((d, t))

    matches, matched_d, matched_t = [], set(), set()
    for d, t in cands:
        if iou[d, t] >= iou_thr:
            matches.append((d, t))
            matched_d.add(d)
            matched_t.add(t)

    return (
        matches,
        [d for d in range(D) if d not in matched_d],
        [t for t in range(T) if t not in matched_t],
    )


# ---------------------------------------------------------------------------
# Public smoother class
# ---------------------------------------------------------------------------

class TileDetectionSmoother:
    """
    Kalman-filter smoother for merged tile / panorama-view detections.

    Call ``update()`` once per frame with the NMS-merged detection arrays.
    The smoother returns those same detections augmented with Kalman-predicted
    boxes for any confirmed track that had no matching detection this frame,
    preventing objects from disappearing when they happen to fall near a tile
    boundary or in a momentarily low-confidence view.

    Parameters
    ----------
    max_age : int
        Maximum frames a track survives without a matching detection before it
        is deleted.  Keep small (2–4) so stale predictions don't linger.
    min_hits : int
        Minimum detection hits before a track contributes predicted boxes.
        Prevents one-frame false-positives from generating phantom predictions.
    iou_threshold : float
        Minimum IoU required to associate a detection with a track prediction.
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 2,
        iou_threshold: float = 0.35,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: list[_KalmanBox] = []

    def reset(self) -> None:
        self._tracks = []

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        boxes   : (N, 4) float32, xyxy frame-coordinate boxes (may be empty)
        scores  : (N,)   float32 confidence values
        classes : (N,)   int64   class ids

        Returns
        -------
        Tuple (boxes, scores, classes) — same dtypes, shape >= input.
        Added rows are Kalman-predicted boxes for confirmed tracks with no match.
        """
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).ravel()
        classes = np.asarray(classes, dtype=np.int64).ravel()

        # ── 1. Advance all existing tracks ──────────────────────────────────
        trk_predicted: list[np.ndarray] = [t.predict() for t in self._tracks]

        # Drop any track whose prediction drifted to NaN / infinity
        valid = [np.all(np.isfinite(p)) for p in trk_predicted]
        self._tracks = [t for t, ok in zip(self._tracks, valid) if ok]
        trk_predicted = [p for p, ok in zip(trk_predicted, valid) if ok]

        # ── 2. Associate detections with track predictions ───────────────────
        trk_arr = np.stack(trk_predicted) if trk_predicted else np.zeros((0, 4), np.float32)
        matches, unmatched_d, _ = _associate(trk_arr, boxes, self.iou_threshold)

        # ── 3. Update matched tracks ─────────────────────────────────────────
        for d, t in matches:
            self._tracks[t].update(boxes[d], float(scores[d]), int(classes[d]))

        # ── 4. Spawn new tracks from unmatched detections ────────────────────
        for d in unmatched_d:
            self._tracks.append(_KalmanBox(boxes[d], float(scores[d]), int(classes[d])))

        # ── 5. Cull expired tracks ────────────────────────────────────────────
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        # ── 6. Collect predicted boxes for confirmed-but-missed tracks ────────
        matched_t_set = {t for _, t in matches}
        extra_boxes: list[np.ndarray] = []
        extra_scores: list[float] = []
        extra_classes: list[int] = []

        for ti, track in enumerate(self._tracks):
            if ti in matched_t_set:
                continue                         # already in the original set
            if track.hits < self.min_hits:
                continue                         # not yet confirmed
            if track.time_since_update == 0:
                continue                         # freshly updated (guard)
            pred = track.xyxy
            if not np.all(np.isfinite(pred)):
                continue
            extra_boxes.append(pred)
            extra_scores.append(track.score)
            extra_classes.append(track.cls)

        if not extra_boxes:
            return boxes, scores, classes

        extra_boxes_np = np.stack(extra_boxes).astype(np.float32)
        extra_scores_np = np.array(extra_scores, dtype=np.float32)
        extra_classes_np = np.array(extra_classes, dtype=np.int64)

        if len(boxes) == 0:
            return extra_boxes_np, extra_scores_np, extra_classes_np

        return (
            np.concatenate([boxes, extra_boxes_np], axis=0),
            np.concatenate([scores, extra_scores_np]),
            np.concatenate([classes, extra_classes_np]),
        )
