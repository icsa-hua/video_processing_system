from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
import cv2 
import numpy as np 

from abc import ABC, abstractmethod
from typing import Dict, Tuple
from obs_system.utils.global_config import DEFISH_ALPHA, DEFISH_BETA, DISTORTION_STRENGTH 


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

    def __init__(self, fog_deg:float=90.0, crop=0.1, cx=None, cy=None)->None: 
        self.__fog_def = fog_deg
        self.__crop=crop 
        self.__cx = cx 
        self.__cy = cy 
        self.__map_cache:Dict[Tuple[int, int, float, float, float, float], Tuple[np.ndarray, np.ndarray, int, int,]] = {}


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

        map_x_full, map_y_full = self.__build_maps(w,h,k,cx,cy) 
    
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



    def _defish(self, imgs, k=DISTORTION_STRENGTH, border=cv2.BORDER_CONSTANT, apply_gain:bool=True):
        """Simple fisheye correction without calibration.
        Args:
            img: input BGR image
            k: distortion strength (-0.2 to -0.6 typical for fisheye)
        """
    
        out = [] 
        if not imgs: 
            return out 

        for img in imgs: 
            h, w = img.shape[:2] 
            cx = w / 2.0 if self.__cx is None else float(self.__cx) 
            cy = h / 2.0 if self.__cy is None else float(self.__cy) 

            map_x, map_y, out_w, out_y = self.__get_cropped_maps(w,h,k,cx,cy) 

            undist = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode= border)

            if apply_gain and (DEFISH_ALPHA != 1.0 and DEFISH_BETA != 0.0): 
                undist = cv2.convertScaleAbs(undist, alpha=DEFISH_ALPHA, beta=DEFISH_BETA)

            out.append(undist)

        return out 



