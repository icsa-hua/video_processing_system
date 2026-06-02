"""
Perspective reprojection from equirectangular panorama images.

Converts a cropped equirectangular panorama (e.g. 1920×600 road-region crop)
into 2–4 overlapping pinhole/tangent views sized for YOLO detection, then
back-projects bounding boxes from view space to panorama pixel space.

Why perspective reprojection instead of traditional defish:
  The equirectangular images fed to this pipeline are already unwrapped from a
  fisheye camera. The distortion problem is purely geometric — a wide-angle
  panorama projects straight lines as curves, causing YOLO (trained on pinhole
  images) to misclassify elongated or curved shapes.  Reprojecting local regions
  into proper pinhole views corrects this without needing a distortion model.

Coordinate conventions:
  Panorama pixel (px, py):
    px in [0, W),  left→right maps to lon ∈ [-hfov/2, +hfov/2]
    py in [0, H),  top→bottom maps to lat ∈ [+vfov/2, -vfov/2]
  Spherical angles (lon, lat) in radians:
    lon: horizontal angle around the vertical axis, 0 = image centre
    lat: elevation angle, 0 = image centre, positive = up
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Tuple

from obs_system.utils.logger import get_logger
from obs_system.utils.global_config import (
    PANORAMA_N_VIEWS,
    PANORAMA_VIEW_SIZE,
    PANORAMA_VIEW_FOV_DEG,
    PANORAMA_PITCH_DEG,
    PANORAMA_HFOV_DEG,
    PANORAMA_VFOV_DEG,
    PANORAMA_MIN_MOTION_FRACTION,
)

logger = get_logger("obs_system." + __name__)


class PanoramaReprojector:
    """
    Generates overlapping perspective (pinhole) views from an equirectangular
    panorama crop and back-projects bounding boxes to panorama pixel space.

    Args:
        n_views:               Number of perspective views (2–4 recommended).
        view_size:             (width, height) of each output view in pixels.
        view_fov_deg:          Horizontal FOV of each perspective view in degrees.
        pitch_deg:             Pitch offset applied to all views in degrees
                               (negative → tilt downward toward the road).
        panorama_hfov_deg:     Total horizontal angular span of the panorama image.
        panorama_vfov_deg:     Total vertical angular span of the panorama image.
        min_motion_fraction:   Fraction of a view's horizontal strip that must have
                               motion (from the fg mask) to activate that view.
    """

    def __init__(
        self,
        n_views: int = PANORAMA_N_VIEWS,
        view_size: Tuple[int, int] = PANORAMA_VIEW_SIZE,
        view_fov_deg: float = PANORAMA_VIEW_FOV_DEG,
        pitch_deg: float = PANORAMA_PITCH_DEG,
        panorama_hfov_deg: float = PANORAMA_HFOV_DEG,
        panorama_vfov_deg: float = PANORAMA_VFOV_DEG,
        min_motion_fraction: float = PANORAMA_MIN_MOTION_FRACTION,
    ) -> None:
        self.n_views = max(1, int(n_views))
        self.view_w, self.view_h = int(view_size[0]), int(view_size[1])
        self.view_fov_deg = float(view_fov_deg)
        self.pitch_deg = float(pitch_deg)
        self.panorama_hfov_deg = float(panorama_hfov_deg)
        self.panorama_vfov_deg = float(panorama_vfov_deg)
        self.min_motion_fraction = float(min_motion_fraction)

        # Lazily built when the first frame arrives (need panorama W, H)
        self._pano_w: int = 0
        self._pano_h: int = 0
        self._map_x: List[np.ndarray] = []   # [n_views] × (view_h, view_w) float32
        self._map_y: List[np.ndarray] = []
        self._yaw_rads: List[float] = []      # view centre yaw in radians

    # ── Rotation helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _rot_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
        """3×3 rotation matrix: view frame → world frame."""
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        R_yaw = np.array([[ cy, 0, sy],
                          [  0, 1,  0],
                          [-sy, 0, cy]], dtype=np.float64)
        R_pitch = np.array([[1,  0,   0],
                            [0, cp, -sp],
                            [0, sp,  cp]], dtype=np.float64)
        return R_yaw @ R_pitch

    # ── Map construction ──────────────────────────────────────────────────────

    def _build_maps(self, pano_w: int, pano_h: int) -> None:
        """
        Precompute (map_x, map_y) remap arrays for each perspective view.

        For each output pixel (u, v) in the perspective view:
          1. Compute the 3D ray via the pinhole model.
          2. Rotate into world frame.
          3. Convert to spherical (lon, lat).
          4. Map to panorama pixel coordinates.
        """
        self._pano_w = pano_w
        self._pano_h = pano_h
        self._map_x.clear()
        self._map_y.clear()
        self._yaw_rads.clear()

        hfov_rad = np.deg2rad(self.panorama_hfov_deg)
        pitch_rad = np.deg2rad(self.pitch_deg)
        vfov_rad = np.deg2rad(self.panorama_vfov_deg)

        # Centre each edge view so its FOV just reaches the panorama boundary.
        # margin = half-view-FOV keeps the outermost view fully inside the panorama.
        if self.n_views == 1:
            yaw_centers = [0.0]
        else:
            view_hfov_rad = np.deg2rad(self.view_fov_deg)
            span_rad = max(hfov_rad - view_hfov_rad, 0.0)
            yaw_centers = [
                -span_rad / 2.0 + i * span_rad / (self.n_views - 1)
                for i in range(self.n_views)
            ]

        # Pinhole intrinsics for the output view
        fx = fy = (self.view_w / 2.0) / np.tan(np.deg2rad(self.view_fov_deg) / 2.0)
        cx_v, cy_v = self.view_w / 2.0, self.view_h / 2.0

        # Pixel-centre grid for the output view (float64 for accuracy)
        uu, vv = np.meshgrid(
            np.arange(self.view_w, dtype=np.float64),
            np.arange(self.view_h, dtype=np.float64),
        )

        for yaw in yaw_centers:
            self._yaw_rads.append(yaw)
            R = self._rot_yaw_pitch(yaw, pitch_rad)  # view → world

            # Unit rays in view (camera) frame
            rx = (uu - cx_v) / fx
            ry = (vv - cy_v) / fy
            rz = np.ones_like(rx)

            # Rotate to world frame: stack into (H*W, 3), apply R^T
            rays_flat = np.stack([rx.ravel(), ry.ravel(), rz.ravel()], axis=0)  # (3, N)
            world_flat = R.T @ rays_flat                                          # (3, N)

            wx = world_flat[0].reshape(self.view_h, self.view_w)
            wy = world_flat[1].reshape(self.view_h, self.view_w)
            wz = world_flat[2].reshape(self.view_h, self.view_w)

            # 3D ray → spherical angles
            lon = np.arctan2(wx, wz)
            lat = np.arctan2(wy, np.sqrt(wx ** 2 + wz ** 2))

            # Spherical → panorama pixel
            px = (lon / hfov_rad + 0.5) * pano_w
            py = (0.5 - lat / vfov_rad) * pano_h

            self._map_x.append(px.astype(np.float32))
            self._map_y.append(py.astype(np.float32))

        logger.info(
            "PanoramaReprojector: %d views built (%dx%d px, FOV=%.0f°, pitch=%.1f°, yaws=%s)",
            self.n_views,
            self.view_w, self.view_h,
            self.view_fov_deg,
            self.pitch_deg,
            [f"{np.rad2deg(y):.1f}°" for y in self._yaw_rads],
        )

    def _ensure_maps(self, pano_w: int, pano_h: int) -> None:
        if self._pano_w != pano_w or self._pano_h != pano_h:
            self._build_maps(pano_w, pano_h)

    # ── Motion-driven view selection ──────────────────────────────────────────

    def _view_has_motion(
        self, fg_mask_small: np.ndarray, view_id: int
    ) -> bool:
        """
        Fast spatial motion check: does this view's horizontal panorama strip
        contain enough foreground pixels?

        fg_mask_small is the binary foreground mask at downscale resolution
        (e.g. 320×320 from the background subtractor).  We check the horizontal
        columns that correspond to the view's angular coverage.
        """
        W = fg_mask_small.shape[1]
        yaw = self._yaw_rads[view_id]
        half_fov = np.deg2rad(self.view_fov_deg / 2.0)
        hfov_half = np.deg2rad(self.panorama_hfov_deg / 2.0)

        x_min = int((yaw - half_fov) / (2.0 * hfov_half) * W + W / 2.0)
        x_max = int((yaw + half_fov) / (2.0 * hfov_half) * W + W / 2.0)
        x_min = max(0, min(x_min, W - 1))
        x_max = max(x_min + 1, min(x_max, W))

        strip = fg_mask_small[:, x_min:x_max]
        total = strip.size
        if total == 0:
            return False
        return float(cv2.countNonZero(strip)) / float(total) >= self.min_motion_fraction

    # ── View generation ───────────────────────────────────────────────────────

    def get_views(
        self,
        panorama: np.ndarray,
        fg_mask_small: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, int, bool]]:
        """
        Remap a panorama frame into perspective views.

        Args:
            panorama:      Equirectangular BGR image (H×W×3).
            fg_mask_small: Optional binary foreground mask at small resolution
                           (from background subtractor).  Controls which views are
                           marked as having motion.  None → all views are active.

        Returns:
            List of (view_bgr, view_id, has_motion) for each of the n_views views.
        """
        h, w = panorama.shape[:2]
        self._ensure_maps(w, h)

        result: List[Tuple[np.ndarray, int, bool]] = []
        for v_id in range(self.n_views):
            view = cv2.remap(
                panorama,
                self._map_x[v_id],
                self._map_y[v_id],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            if fg_mask_small is not None:
                has_motion = self._view_has_motion(fg_mask_small, v_id)
            else:
                has_motion = True

            result.append((view, v_id, has_motion))

        return result

    # ── Back-projection ───────────────────────────────────────────────────────

    @staticmethod
    def _bilinear_sample(
        map2d: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """
        Sample a 2D float32 map at fractional (u, v) coordinates using
        bilinear interpolation.
        """
        H, W = map2d.shape
        u0 = np.clip(np.floor(u).astype(np.int32), 0, W - 2)
        v0 = np.clip(np.floor(v).astype(np.int32), 0, H - 2)
        u1 = u0 + 1
        v1 = v0 + 1
        wu = (u - u0).astype(np.float32)
        wv = (v - v0).astype(np.float32)
        return (
            (1.0 - wu) * (1.0 - wv) * map2d[v0, u0]
            + wu * (1.0 - wv) * map2d[v0, u1]
            + (1.0 - wu) * wv * map2d[v1, u0]
            + wu * wv * map2d[v1, u1]
        )

    def backproject_boxes(
        self,
        boxes_xyxy: np.ndarray,
        view_id: int,
    ) -> np.ndarray:
        """
        Back-project bounding boxes from a perspective view to panorama pixel space.

        Samples the precomputed remap arrays at each box corner position using
        bilinear interpolation, then takes the axis-aligned bounding box of the
        four back-projected corners.

        Args:
            boxes_xyxy: (N, 4) float32 array of [x1, y1, x2, y2] in view pixels.
            view_id:    Index of the view (0..n_views-1).

        Returns:
            (N, 4) float32 array of [x1, y1, x2, y2] in panorama pixels,
            clamped to image bounds.  Rows with zero-area boxes remain in the
            output; the caller should filter with a valid-box check.
        """
        if boxes_xyxy.size == 0 or not (0 <= view_id < self.n_views):
            return np.zeros((0, 4), dtype=np.float32)

        N = len(boxes_xyxy)
        x1, y1, x2, y2 = (
            boxes_xyxy[:, 0], boxes_xyxy[:, 1],
            boxes_xyxy[:, 2], boxes_xyxy[:, 3],
        )

        # Four corners per box: TL, TR, BR, BL  →  (4N,) flat arrays
        cu = np.concatenate([x1, x2, x2, x1])  # (4N,)
        cv_ = np.concatenate([y1, y1, y2, y2])

        cu = np.clip(cu, 0.0, self.view_w - 1).astype(np.float64)
        cv_ = np.clip(cv_, 0.0, self.view_h - 1).astype(np.float64)

        pano_x = self._bilinear_sample(self._map_x[view_id], cu, cv_)
        pano_y = self._bilinear_sample(self._map_y[view_id], cu, cv_)

        # Reshape to (4, N) for min/max
        pano_x = pano_x.reshape(4, N)
        pano_y = pano_y.reshape(4, N)

        out_x1 = np.clip(pano_x.min(axis=0), 0.0, self._pano_w - 1)
        out_y1 = np.clip(pano_y.min(axis=0), 0.0, self._pano_h - 1)
        out_x2 = np.clip(pano_x.max(axis=0), 0.0, self._pano_w - 1)
        out_y2 = np.clip(pano_y.max(axis=0), 0.0, self._pano_h - 1)

        return np.stack([out_x1, out_y1, out_x2, out_y2], axis=1).astype(np.float32)
