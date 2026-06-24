from pathlib import Path

from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
import cv2
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from obs_system.utils.global_config import (
    DEFISH_ALPHA, DEFISH_BETA, DISTORTION_STRENGTH,
    DEFISH_MODEL, DEFISH_FISHEYE_FOV_DEG, DEFISH_OUTPUT_FOV_DEG,
    FISHEYE_N_VIEWS, FISHEYE_VIEW_SIZE, FISHEYE_VIEW_FOV_DEG,
    FISHEYE_VIEW_TILT_DEG, FISHEYE_VIEW_YAWS_DEG,
    FISHEYE_VIEW_MIN_MOTION_FRACTION,
    DEFAULT_FISHEYE_PROFILE, FISHEYE_PROFILES,
)


class FishEyeProjection(EventExtractorInterface): 
    """
    Geometry-aware fisheye handler:
      - Precompute undistort/rectify maps for multiple 'views' (yaw/pitch rotations).
      - Remap input frames to each rectified view (run detector there).
      - Back-project per-view boxes to original fisheye pixel coordinates.

    Theory (high level):
      - Undistort/rectify uses (K, D, R_view, K_new) to map distorted fisheye -> rectified pinhole view.
      - To back-project a rectified pixel u_rect:
            x_rect = K_new^{-1} * [u_rect, 1]
            x_cam  = R_view^T * x_rect
            (x_norm = x_cam[:2] / x_cam[2])
            u_fish  = distort_fisheye(x_norm; K, D)
    """

    def __init__(
        self,
        fog_deg: float = 90.0,
        crop=0.1,
        cx=None,
        cy=None,
        use_tangent_views: bool = True,
        n_views: int = FISHEYE_N_VIEWS,
        view_size: Tuple[int, int] = FISHEYE_VIEW_SIZE,
        view_fov_deg: float = FISHEYE_VIEW_FOV_DEG,
        view_tilt_deg: float = FISHEYE_VIEW_TILT_DEG,
        view_yaws_deg: Optional[List[float]] = None,
        min_motion_fraction: float = FISHEYE_VIEW_MIN_MOTION_FRACTION,
        profile_name: str = DEFAULT_FISHEYE_PROFILE,
        capture_dir=None,
    )->None:
        self._capture_dir: Path | None = Path(capture_dir) if capture_dir is not None else None
        self._save_call_count: int = 0
        self._save_captured: int = 0
        self._save_max: int = 10
        self._save_spacing: int = 50
        if profile_name not in FISHEYE_PROFILES:
            choices = ", ".join(sorted(FISHEYE_PROFILES))
            raise ValueError(f"Unknown fisheye profile {profile_name!r}. Choose one of: {choices}")

        profile = FISHEYE_PROFILES[profile_name]
        self.__fog_def = fog_deg
        self.__crop=crop 
        self.__cx = cx 
        self.__cy = cy 
        self.__map_cache:Dict[Tuple[int, int, float, float, float, float], Tuple[np.ndarray, np.ndarray, int, int,]] = {}
        self.__equidistant_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self.use_tangent_views = bool(use_tangent_views)
        self.profile_name = str(profile_name)

        if self.profile_name == DEFAULT_FISHEYE_PROFILE:
            configured_n_views = n_views
            configured_view_size = view_size
            configured_view_fov = view_fov_deg
            configured_yaws = FISHEYE_VIEW_YAWS_DEG if view_yaws_deg is None else view_yaws_deg
            configured_tilts = [view_tilt_deg] * int(configured_n_views)
            configured_motion_fraction = min_motion_fraction
        else:
            configured_n_views = profile["n_views"]
            configured_view_size = profile["view_size"]
            configured_view_fov = profile["view_fov_deg"]
            configured_yaws = profile["view_yaws_deg"]
            configured_tilts = profile["view_tilts_deg"]
            configured_motion_fraction = profile["min_motion_fraction"]

        self.n_views = max(1, int(configured_n_views))
        self.view_w, self.view_h = int(configured_view_size[0]), int(configured_view_size[1])
        self.view_fov_deg = float(configured_view_fov)
        self.view_tilt_deg = float(view_tilt_deg)
        yaws = configured_yaws
        if len(yaws) < self.n_views:
            raise ValueError("view_yaws_deg must contain at least n_views entries")
        self.view_yaws_deg = [float(y) for y in yaws[:self.n_views]]
        if len(configured_tilts) < self.n_views:
            raise ValueError("view_tilts_deg must contain at least n_views entries")
        self.view_tilts_deg = [float(t) for t in configured_tilts[:self.n_views]]
        self.min_motion_fraction = float(configured_motion_fraction)

        self.center_x_ratio = float(profile.get("center_x_ratio", 0.5))
        self.center_y_ratio = float(profile.get("center_y_ratio", 0.5))
        self.radius_ratio = float(profile.get("radius_ratio", 0.5))
        self.full_refresh_interval_frames = max(
            0, int(profile.get("full_refresh_interval_frames", 0))
        )
        self.view_persist_frames = max(0, int(profile.get("view_persist_frames", 0)))
        self.backproject_edge_samples = max(
            1, int(profile.get("backproject_edge_samples", 1))
        )
        self.allowed_class_names = frozenset(
            str(name).strip().lower()
            for name in profile.get("allowed_class_names", [])
            if str(name).strip()
        )
        self._last_active_frame = np.full((self.n_views,), -10**9, dtype=np.int64)
        self._fallback_frame_index = -1
        self._view_metrics = {
            "frames": 0,
            "refresh_frames": 0,
            "views_considered": 0,
            "views_submitted": 0,
            "views_motion_activated": 0,
            "views_persisted": 0,
            "views_detection_hits": 0,
        }
        self._fish_w: int = 0
        self._fish_h: int = 0
        self._tangent_map_x: List[np.ndarray] = []
        self._tangent_map_y: List[np.ndarray] = []
        self._tangent_map1_int: List[np.ndarray] = []
        self._tangent_map2_int: List[np.ndarray] = []


    def _build(self, 
                 K: np.ndarray, #3x3 fishey intrinsics 
                 D: np.ndarray, #(4, ) fishey distortion (open cv fisheye) 
                 src_size: tuple[int, int], #(W_src, H_src) original fisheye image size  
                 rect_size: tuple[int, int], #(W-rect, H_rect) per-view rectified output size 
                 views_yaw_pitch:np.ndarray, #shape (V, 2) in radians: [[yaw0, pitch0], [...],...] 
                 fov_deg: float = 90.0, 
                 alpha: float = 0.0

        )->None: 

        self.K = np.asarray(K, dtype=np.float64) 
        self.D = np.asarray(D, dtype=np.float64).reshape(-1) 
        self.W_src, self.H_src = int(src_size[0]), int(src_size[1])
        self.Wr, self.Hr = int(rect_size[0]), int(rect_size[1])

        assert self.K.shape == (3, 3), "K must be 3x3"

        self._V = int(views_yaw_pitch.shape[0])

        # Create the rectified intrinsics simple pinhole targeting desired FOV 
        # fx = fy = (Wr/2) / tran(FOV)/2; principal point = center 
        f = (self.Wr * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)

        K_new = np.array([[f, 0, self.Wr * 0.5],
                          [0, f, self.Hr * 0.5],
                          [0, 0, 1.0]], dtype=np.float64)

        # Precompute per-view rotation matrices and remap grids.
        self._R_list = np.empty((self._V, 3, 3), dtype=np.float64)
        self._Knew_list = np.empty((self._V, 3, 3), dtype=np.float64)
        self._map1 = []
        self._map2 = []

        for i in range(self._V):
            yaw, pitch = float(views_yaw_pitch[i, 0]), float(views_yaw_pitch[i, 1])
            R = self._rot_yaw_pitch(yaw, pitch)  # world->camera rotation for this tangent view
            self._R_list[i] = R
            self._Knew_list[i] = K_new.copy()
            # Remap fisheye -> rectified view i
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, R, K_new, (self.Wr, self.Hr), cv2.CV_32FC1
            )
            self._map1.append(m1)
            self._map2.append(m2)

        # lightweight cache for inverse K_new
        self._Knew_inv = np.linalg.inv(self._Knew_list)

    @staticmethod
    def _rot_yaw_pitch(yaw:float, pitch:float) -> np.ndarray: 
        
        """
        Build a rotation matrix from yaw (around +Y) and pitch (around +X).
        This orients the rectified view's optical axis on the unit sphere.
        """
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        R_yaw = np.array([[ cy, 0, sy],
                          [  0, 1,  0],
                          [-sy, 0, cy]], dtype=np.float64)
        R_pitch = np.array([[1,  0,   0],
                            [0, cp, -sp],
                            [0, sp,  cp]], dtype=np.float64)
        return R_yaw @ R_pitch


    def remap_view(self, frame_bgr: np.ndarray, view_id: int) -> np.ndarray:
        """
        Apply precomputed remap to get rectified view (BGR in -> BGR out).
        """
        assert 0 <= view_id < self._V
        return cv2.remap(frame_bgr, self._map1[view_id], self._map2[view_id],
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


    @staticmethod
    def _bilinear_sample(map2d: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        h, w = map2d.shape
        u0 = np.clip(np.floor(u).astype(np.int32), 0, w - 2)
        v0 = np.clip(np.floor(v).astype(np.int32), 0, h - 2)
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


    def _build_tangent_maps(self, fish_w: int, fish_h: int) -> None:
        """
        Build pinhole/tangent view maps for a circular equidistant fisheye.

        Source convention:
          - source image centre is the fisheye optical axis
          - yaw 0° points right; positive yaw rotates counter-clockwise
          - image y points down, so positive spherical y projects upward
        """
        self._fish_w = int(fish_w)
        self._fish_h = int(fish_h)
        self._tangent_map_x.clear()
        self._tangent_map_y.clear()
        self._tangent_map1_int.clear()
        self._tangent_map2_int.clear()

        cx = self._fish_w * self.center_x_ratio if self.__cx is None else float(self.__cx)
        cy = self._fish_h * self.center_y_ratio if self.__cy is None else float(self.__cy)
        theta_max = np.deg2rad(DEFISH_FISHEYE_FOV_DEG / 2.0)
        fisheye_radius = min(self._fish_w, self._fish_h) * self.radius_ratio
        f_fish = fisheye_radius / max(theta_max, 1e-9)

        f_view = (self.view_w / 2.0) / np.tan(np.deg2rad(self.view_fov_deg) / 2.0)
        cx_v, cy_v = self.view_w / 2.0, self.view_h / 2.0
        uu, vv = np.meshgrid(
            np.arange(self.view_w, dtype=np.float64),
            np.arange(self.view_h, dtype=np.float64),
        )
        local_x = (uu - cx_v) / f_view
        local_y = (vv - cy_v) / f_view

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        for yaw_deg, tilt_deg in zip(self.view_yaws_deg, self.view_tilts_deg):
            yaw = np.deg2rad(yaw_deg)
            tilt = np.deg2rad(tilt_deg)
            view_forward = np.array(
                [
                    np.cos(yaw) * np.sin(tilt),
                    np.sin(yaw) * np.sin(tilt),
                    np.cos(tilt),
                ],
                dtype=np.float64,
            )
            view_forward /= np.linalg.norm(view_forward)

            view_right = np.cross(world_up, view_forward)
            norm_right = np.linalg.norm(view_right)
            if norm_right < 1e-9:
                view_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                view_right /= norm_right

            view_down = -np.cross(view_forward, view_right)
            view_down /= np.linalg.norm(view_down)

            rays_x = view_forward[0] + local_x * view_right[0] + local_y * view_down[0]
            rays_y = view_forward[1] + local_x * view_right[1] + local_y * view_down[1]
            rays_z = view_forward[2] + local_x * view_right[2] + local_y * view_down[2]
            ray_norm = np.sqrt(rays_x * rays_x + rays_y * rays_y + rays_z * rays_z)
            rays_x /= ray_norm
            rays_y /= ray_norm
            rays_z /= ray_norm

            theta = np.arccos(np.clip(rays_z, -1.0, 1.0))
            phi = np.arctan2(rays_y, rays_x)
            radius = f_fish * theta

            map_x = (cx + radius * np.cos(phi)).astype(np.float32)
            map_y = (cy - radius * np.sin(phi)).astype(np.float32)

            outside = theta > theta_max
            if np.any(outside):
                map_x[outside] = -1.0
                map_y[outside] = -1.0

            self._tangent_map_x.append(map_x)
            self._tangent_map_y.append(map_y)
            map1, map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
            self._tangent_map1_int.append(map1)
            self._tangent_map2_int.append(map2)


    def _ensure_tangent_maps(self, fish_w: int, fish_h: int) -> None:
        if self._fish_w != int(fish_w) or self._fish_h != int(fish_h):
            self._build_tangent_maps(fish_w, fish_h)


    def _resolve_frame_number(self, frame_id: Optional[int]) -> int:
        if isinstance(frame_id, (int, np.integer)):
            return int(frame_id)
        self._fallback_frame_index += 1
        return self._fallback_frame_index


    def is_refresh_frame(self, frame_id: Optional[int]) -> bool:
        if self.full_refresh_interval_frames <= 0:
            return False
        if not isinstance(frame_id, (int, np.integer)):
            return False
        return int(frame_id) % self.full_refresh_interval_frames == 0


    def has_persisted_views(self, frame_id: Optional[int]) -> bool:
        if self.view_persist_frames <= 0 or not isinstance(frame_id, (int, np.integer)):
            return False
        age = int(frame_id) - self._last_active_frame
        return bool(np.any((age >= 0) & (age <= self.view_persist_frames)))


    def should_force_frame(self, frame_id: Optional[int]) -> bool:
        """Allow the frame through the whole-frame motion gate when required."""
        return self.is_refresh_frame(frame_id) or self.has_persisted_views(frame_id)


    def mark_view_detection(self, view_id: int, frame_id: Optional[int]) -> None:
        if not (0 <= int(view_id) < self.n_views):
            return
        frame_number = self._resolve_frame_number(frame_id)
        self._last_active_frame[int(view_id)] = frame_number
        self._view_metrics["views_detection_hits"] += 1


    def reset_runtime_state(self) -> None:
        self._last_active_frame.fill(-10**9)
        self._fallback_frame_index = -1
        for key in self._view_metrics:
            self._view_metrics[key] = 0


    def get_runtime_metrics(self) -> Dict[str, float | int | str]:
        frames = max(int(self._view_metrics["frames"]), 1)
        considered = max(int(self._view_metrics["views_considered"]), 1)
        return {
            "fisheye_profile": self.profile_name,
            **{
                f"fisheye_{key}": value
                for key, value in self._view_metrics.items()
            },
            "fisheye_views_submitted_per_frame": (
                float(self._view_metrics["views_submitted"]) / float(frames)
            ),
            "fisheye_view_submission_ratio": (
                float(self._view_metrics["views_submitted"]) / float(considered)
            ),
        }


    def _view_has_motion(self, fg_mask_small: np.ndarray, view_id: int) -> bool:
        if fg_mask_small is None or fg_mask_small.size == 0:
            return True

        mask_h, mask_w = fg_mask_small.shape[:2]
        sx = float(mask_w) / max(float(self._fish_w), 1.0)
        sy = float(mask_h) / max(float(self._fish_h), 1.0)
        map_x = (self._tangent_map_x[view_id] * sx).astype(np.float32)
        map_y = (self._tangent_map_y[view_id] * sy).astype(np.float32)
        view_mask = cv2.remap(
            fg_mask_small,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        total = int(view_mask.size)
        if total == 0:
            return False
        return float(cv2.countNonZero(view_mask)) / float(total) >= self.min_motion_fraction


    def get_views(
        self,
        fisheye_bgr: np.ndarray,
        fg_mask_small: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None,
        force_refresh: Optional[bool] = None,
    ) -> List[Tuple[Optional[np.ndarray], int, bool]]:
        """
        Return tangent/pinhole views for YOLO.

        Inactive views are returned with ``view_bgr=None`` so callers can avoid
        paying image remap cost when the sector has no motion, persistence, or
        scheduled refresh.  The tuple boolean means "submit this view", not
        necessarily that motion fired in the current frame.
        """
        h, w = fisheye_bgr.shape[:2]
        self._ensure_tangent_maps(w, h)
        frame_number = self._resolve_frame_number(frame_id)
        refresh = self.is_refresh_frame(frame_id) if force_refresh is None else bool(force_refresh)
        self._view_metrics["frames"] += 1
        self._view_metrics["views_considered"] += self.n_views
        if refresh:
            self._view_metrics["refresh_frames"] += 1

        result: List[Tuple[Optional[np.ndarray], int, bool]] = []
        for v_id in range(self.n_views):
            mask_available = fg_mask_small is not None
            has_motion = (
                self._view_has_motion(fg_mask_small, v_id)
                if mask_available
                else False
            )
            if has_motion:
                self._last_active_frame[v_id] = frame_number
                self._view_metrics["views_motion_activated"] += 1

            age = frame_number - int(self._last_active_frame[v_id])
            persisted = (
                self.view_persist_frames > 0
                and age >= 0
                and age <= self.view_persist_frames
            )
            ungated = not mask_available and not refresh
            active = refresh or ungated or has_motion or persisted
            if not active:
                result.append((None, v_id, False))
                continue
            if persisted and not has_motion and not refresh:
                self._view_metrics["views_persisted"] += 1

            view = cv2.remap(
                fisheye_bgr,
                self._tangent_map1_int[v_id],
                self._tangent_map2_int[v_id],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            self._view_metrics["views_submitted"] += 1
            result.append((view, v_id, True))

        if self._capture_dir is not None:
            self._save_call_count += 1
            if (
                self._save_captured < self._save_max
                and (self._save_call_count == 1 or self._save_call_count % self._save_spacing == 0)
            ):
                self._save_captured += 1
                idx = self._save_captured
                self._capture_dir.mkdir(parents=True, exist_ok=True)
                _view_labels = ("top", "left", "right", "persp")
                for view_bgr, v_id, submitted in result:
                    if view_bgr is None:
                        continue
                    label = _view_labels[v_id] if v_id < len(_view_labels) else f"v{v_id}"
                    cv2.imwrite(str(self._capture_dir / f"s3_fep_{label}_{idx:03d}.png"), view_bgr)

        return result


    def backproject_view_boxes(self, boxes_xyxy: np.ndarray, view_id: int) -> np.ndarray:
        if boxes_xyxy.size == 0 or not (0 <= view_id < self.n_views):
            return np.zeros((0, 4), dtype=np.float32)

        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        n = len(boxes_xyxy)
        x1, y1, x2, y2 = (
            boxes_xyxy[:, 0], boxes_xyxy[:, 1],
            boxes_xyxy[:, 2], boxes_xyxy[:, 3],
        )

        if self.backproject_edge_samples <= 1:
            # Preserve the historical four-corner mapping for the default profile.
            cu = np.concatenate([x1, x2, x2, x1])
            cv_ = np.concatenate([y1, y1, y2, y2])
            point_count = 4
        else:
            t = np.linspace(
                0.0,
                1.0,
                self.backproject_edge_samples,
                dtype=np.float32,
            )[:, None]
            top_x = x1[None, :] + (x2 - x1)[None, :] * t
            side_y = y1[None, :] + (y2 - y1)[None, :] * t
            cu = np.concatenate(
                [
                    top_x,
                    np.broadcast_to(x2[None, :], top_x.shape),
                    top_x[::-1],
                    np.broadcast_to(x1[None, :], top_x.shape),
                ],
                axis=0,
            ).reshape(-1)
            cv_ = np.concatenate(
                [
                    np.broadcast_to(y1[None, :], side_y.shape),
                    side_y,
                    np.broadcast_to(y2[None, :], side_y.shape),
                    side_y[::-1],
                ],
                axis=0,
            ).reshape(-1)
            point_count = self.backproject_edge_samples * 4

        cu = np.clip(cu, 0.0, self.view_w - 1).astype(np.float64)
        cv_ = np.clip(cv_, 0.0, self.view_h - 1).astype(np.float64)

        fish_x = self._bilinear_sample(self._tangent_map_x[view_id], cu, cv_)
        fish_y = self._bilinear_sample(self._tangent_map_y[view_id], cu, cv_)
        fish_x = fish_x.reshape(point_count, n)
        fish_y = fish_y.reshape(point_count, n)

        if self.backproject_edge_samples <= 1:
            out_x1 = np.clip(fish_x.min(axis=0), 0.0, self._fish_w - 1)
            out_y1 = np.clip(fish_y.min(axis=0), 0.0, self._fish_h - 1)
            out_x2 = np.clip(fish_x.max(axis=0), 0.0, self._fish_w - 1)
            out_y2 = np.clip(fish_y.max(axis=0), 0.0, self._fish_h - 1)
        else:
            valid = (
                (fish_x >= 0.0)
                & (fish_x < self._fish_w)
                & (fish_y >= 0.0)
                & (fish_y < self._fish_h)
            )
            safe_x = np.where(valid, fish_x, np.nan)
            safe_y = np.where(valid, fish_y, np.nan)
            with np.errstate(all="ignore"):
                out_x1 = np.nanmin(safe_x, axis=0)
                out_y1 = np.nanmin(safe_y, axis=0)
                out_x2 = np.nanmax(safe_x, axis=0)
                out_y2 = np.nanmax(safe_y, axis=0)
            out_x1 = np.nan_to_num(out_x1, nan=0.0)
            out_y1 = np.nan_to_num(out_y1, nan=0.0)
            out_x2 = np.nan_to_num(out_x2, nan=0.0)
            out_y2 = np.nan_to_num(out_y2, nan=0.0)
            out_x1 = np.clip(out_x1, 0.0, self._fish_w - 1)
            out_y1 = np.clip(out_y1, 0.0, self._fish_h - 1)
            out_x2 = np.clip(out_x2, 0.0, self._fish_w - 1)
            out_y2 = np.clip(out_y2, 0.0, self._fish_h - 1)
        return np.stack([out_x1, out_y1, out_x2, out_y2], axis=1).astype(np.float32)
    

    def backproject_boxes(
        self,
        dets_rect: np.ndarray
    ) -> np.ndarray:
        """
        Back-project rectified detections to original fisheye pixel space.

        Input (Nx7 float/int):
            dets_rect[:,0] = view_id in [0..V-1]
            dets_rect[:,1:5] = [x1,y1,x2,y2] in rectified pixels (view's Wr x Hr)
            dets_rect[:,5] = score
            dets_rect[:,6] = class_id
        Output (M x 6 float/int):
            [x1_f,y1_f,x2_f,y2_f, score, class_id] in original fisheye pixels (W_src x H_src)
            Boxes are clamped to image bounds; degenerate boxes are dropped automatically.
        """
        if dets_rect.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        dets_rect = np.asarray(dets_rect)
        assert dets_rect.shape[1] == 7, "dets_rect must be (N,7): [view_id,x1,y1,x2,y2,score,class]"

        # Gather corners for all boxes (vectorized).
        view_id = dets_rect[:, 0].astype(np.int32)
        x1, y1, x2, y2 = [dets_rect[:, i].astype(np.float64) for i in (1, 2, 3, 4)]

        # 4 corners per box in rectified pixels
        # shape (N,4,2): (x1,y1), (x2,y1), (x2,y2), (x1,y2)
        corners = np.stack([
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ], axis=1)  # float64 for stability

        # Normalize to rectified camera coords: x_rect = K_new^{-1} * u_rect
        # Prepare homogeneous coords (N*4, 3)
        N = corners.shape[0]
        pts_rect = corners.reshape(-1, 2)
        ones = np.ones((pts_rect.shape[0], 1), dtype=np.float64)
        hom = np.concatenate([pts_rect, ones], axis=1)  # (N*4, 3)

        # Apply per-sample K_new^{-1}: do it in chunks by view_id to stay vectorized.
        pts_fisheye_pix = np.empty_like(pts_rect, dtype=np.float64)
        for v in range(self._V):
            mask = (view_id == v)
            if not np.any(mask):
                continue
            idx = np.nonzero(mask)[0] 
            sel = np.repeat(idx * 4, 4) + np.tile(np.arange(4), idx.size)  # indices into (N*4)
            hom_v = hom[sel]  

            x_rect = (self._Knew_inv[v] @ hom_v.T).T  # (M,3)
            x_cam = (self._R_list[v].T @ x_rect.T).T
            z = np.clip(x_cam[:, 2:3], 1e-9, None)
            x_norm = x_cam[:, 0:2] / z  # (M,2) normalized (undistorted) rays

            # Distort to fisheye pixels
            x_norm = x_norm.reshape(-1, 1, 2)  # OpenCV expects (M,1,2)
            u_dist = cv2.fisheye.distortPoints(x_norm, self.K, self.D)  # (M,1,2)
            pts_fisheye_pix[sel] = u_dist.reshape(-1, 2)

        # Recompose per-box corners in fisheye, clamp to image, and make boxes
        corners_f = pts_fisheye_pix.reshape(N, 4, 2)
        x_min = np.clip(corners_f[:, :, 0].min(axis=1), 0, self.W_src - 1)
        y_min = np.clip(corners_f[:, :, 1].min(axis=1), 0, self.H_src - 1)
        x_max = np.clip(corners_f[:, :, 0].max(axis=1), 0, self.W_src - 1)
        y_max = np.clip(corners_f[:, :, 1].max(axis=1), 0, self.H_src - 1)

        # Filter degenerate boxes
        good = (x_max > x_min) & (y_max > y_min)
        if not np.any(good):
            return np.zeros((0, 6), dtype=np.float32)

        out = np.stack([
            x_min[good], y_min[good], x_max[good], y_max[good],
            dets_rect[good, 5].astype(np.float32),       # score
            dets_rect[good, 6].astype(np.float32)        # class_id (kept float for uniform dtype)
        ], axis=1).astype(np.float32)

        return out


    def detect(self, predictions: np.ndarray, save: bool) -> list:
        """
        Adapter to your interface: accepts (N,7) predictions in rectified view space and
        returns a single list with one numpy array (M,6) in fisheye space.

        predictions: np.ndarray with columns:
            [view_id, x1, y1, x2, y2, score, class_id]
        """
        boxes_fisheye = self.backproject_boxes(predictions)
        # 'save' flag left for your upstream use (e.g., dump debug images).
        return [boxes_fisheye]



    def __build_maps(self, w:int, h:int, k:float, cx:float, cy:float):
        sx = max(cx, 1.0)
        sy = max(cy, 1.0)
        x = (np.arange(w, dtype=np.float32)-cx) / sx
        y = (np.arange(h, dtype=np.float32)-cy) / sy
        xv, yv = np.meshgrid(x, y, copy=False)

        r2 = xv * xv + yv * yv
        scale = 1.0 + k * r2
        np.maximum(scale, 1e-6, out=scale)

        src_x = (xv * scale) * sx + cx
        src_y = (yv * scale) * sy + cy
        return src_x.astype(np.float32), src_y.astype(np.float32)


    def __get_cropped_maps(self, w:int, h:int, k:float, cx:float, cy:float):
        key = (w, h, float(k), float(cx), float(cy), float(self.__crop))
        cached = self.__map_cache.get(key)
        if cached is not None:
            return cached

        map_x_full, map_y_full = self.__build_maps(w, h, k, cx, cy)

        if self.__crop > 0.0:
            ch = int(round(h*self.__crop))
            cw = int(round(w*self.__crop))
            y0, y1 = ch, h-ch
            x0, x1 = cw, w-cw
            map_x = map_x_full[y0:y1, x0:x1]
            map_y = map_y_full[y0:y1, x0:x1]
            out_w, out_h = (x1-x0), (y1-y0)
        else:
            map_x, map_y = map_x_full, map_y_full
            out_w, out_h = w, h

        self.__map_cache[key] = (map_x, map_y, out_w, out_h)
        return self.__map_cache[key]


    def __build_equidistant_maps(
        self,
        w: int,
        h: int,
        cx: float,
        cy: float,
        fisheye_fov_deg: float,
        output_fov_deg: float,
    ) -> tuple:
        """
        Build remap arrays for equidistant circular fisheye undistortion.

        For a circular fisheye with equidistant projection r = f_fish * theta:
          - f_fish is derived from the fisheye circle radius and the total lens FOV.
          - Each output pixel is treated as a rectilinear ray; the angle theta from
            the optical axis is computed, then mapped back to the source radius.

        The output has the same (w, h) as the input — no cropping — so downstream
        coordinate systems (ROI translation, motion-gate tile mapping) are unaffected.
        Pixels outside the fisheye circle sample from beyond the lens boundary and
        appear black (BORDER_CONSTANT = 0).
        """
        # Fisheye: r_src = f_fish * theta, theta in [0, fisheye_fov / 2]
        theta_max = np.deg2rad(fisheye_fov_deg / 2.0)
        fisheye_radius = min(w, h) / 2.0
        f_fish = fisheye_radius / theta_max  # px per radian

        # Output pinhole focal length from desired output FOV
        output_half_fov = np.deg2rad(output_fov_deg / 2.0)
        f_out = (min(w, h) / 2.0) / np.tan(output_half_fov)

        # Per-pixel direction in rectilinear output space
        uu = np.arange(w, dtype=np.float64) - cx
        vv = np.arange(h, dtype=np.float64) - cy
        uu_grid, vv_grid = np.meshgrid(uu, vv)

        x_n = uu_grid / f_out
        y_n = vv_grid / f_out
        r_n = np.sqrt(x_n ** 2 + y_n ** 2)

        # Angle from optical axis (rectilinear pinhole)
        theta = np.arctan(r_n)

        # Azimuth angle
        phi = np.arctan2(y_n, x_n)

        # Equidistant source radius
        r_src = f_fish * theta

        src_x = (cx + r_src * np.cos(phi)).astype(np.float32)
        src_y = (cy + r_src * np.sin(phi)).astype(np.float32)
        return src_x, src_y


    def __get_equidistant_maps(
        self,
        w: int,
        h: int,
        cx: float,
        cy: float,
        fisheye_fov_deg: float,
        output_fov_deg: float,
    ) -> tuple:
        key = (w, h, cx, cy, fisheye_fov_deg, output_fov_deg)
        cached = self.__equidistant_cache.get(key)
        if cached is not None:
            return cached
        maps = self.__build_equidistant_maps(w, h, cx, cy, fisheye_fov_deg, output_fov_deg)
        self.__equidistant_cache[key] = maps
        return maps


    def _defish(self, imgs, k=DISTORTION_STRENGTH, border=cv2.BORDER_CONSTANT, apply_gain:bool=True):
        """
        Undistort fisheye images.

        When DEFISH_MODEL == "equidistant" (default), uses a proper equidistant
        circular-fisheye model parameterised by DEFISH_FISHEYE_FOV_DEG and
        DEFISH_OUTPUT_FOV_DEG. This is correct for cameras that produce a circular
        image with a fisheye projection (r = f * theta).

        When DEFISH_MODEL == "barrel", falls back to the polynomial radial model
        (scale = 1 + k * r²) which was the original behaviour.
        """
        out = []
        if not imgs:
            return out

        for img in imgs:
            h, w = img.shape[:2]
            cx = w / 2.0 if self.__cx is None else float(self.__cx)
            cy = h / 2.0 if self.__cy is None else float(self.__cy)

            if DEFISH_MODEL == "equidistant":
                map_x, map_y = self.__get_equidistant_maps(
                    w, h, cx, cy, DEFISH_FISHEYE_FOV_DEG, DEFISH_OUTPUT_FOV_DEG
                )
                undist = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=border)
            else:
                map_x, map_y, _, _ = self.__get_cropped_maps(w, h, k, cx, cy)
                undist = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=border)
                if apply_gain and (DEFISH_ALPHA != 1.0 and DEFISH_BETA != 0.0):
                    undist = cv2.convertScaleAbs(undist, alpha=DEFISH_ALPHA, beta=DEFISH_BETA)

            out.append(undist)

        return out
