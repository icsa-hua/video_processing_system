"""
Tile activation persistence window for motion-gated tiled inference.

Problem: running YOLO inference on every tile of every frame is expensive and
mostly redundant — the vast majority of tiles in a road-traffic scene are static
at any given moment.  A pure motion gate solves throughput but drops objects
that stop (pedestrian waiting at a crosswalk, stationary vehicle).

Solution: two-stage activation.
  1. Motion trigger  — if the FG mask (MOG2 output, possibly downscaled) shows
     enough foreground pixels inside the tile, submit it this frame and reset
     the tile's persistence counter.
  2. Persistence hold — once a tile was triggered (by motion OR by a detection
     on a previous frame), keep submitting it for ``persist_frames`` frames
     even if motion goes cold.  This prevents stopped-object drop-out.

Research metrics: the class tracks total vs. submitted tile counts per batch
and cumulatively, making it straightforward to measure skip rate, compare
traffic densities, and quantify the throughput benefit.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _tile_has_motion(
    fg_mask: np.ndarray,
    frame_h: int,
    frame_w: int,
    tx: int,
    ty: int,
    tile_size: int,
    min_ratio: float,
) -> bool:
    """
    Return True if the tile's region of *fg_mask* contains at least
    *min_ratio* foreground pixels.

    *fg_mask* may be at a different (usually smaller) resolution than the
    original frame — the subtractor downscales before running MOG2.
    Coordinates are mapped proportionally.
    """
    mask_h, mask_w = fg_mask.shape[:2]
    sx = mask_w / max(frame_w, 1)
    sy = mask_h / max(frame_h, 1)

    x0 = max(0, int(tx * sx))
    y0 = max(0, int(ty * sy))
    x1 = min(mask_w, int((tx + tile_size) * sx))
    y1 = min(mask_h, int((ty + tile_size) * sy))

    if x1 <= x0 or y1 <= y0:
        return False

    region = fg_mask[y0:y1, x0:x1]
    return int(np.count_nonzero(region)) / region.size >= min_ratio


class TileActivationWindow:
    """
    Per-tile activation state with persistence.

    Usage pattern inside the tile loop
    -----------------------------------
    ::

        window.reset_batch_metrics()

        for bni, (keep, img, fid) in enumerate(zip(mfgs, im0s, frame_ids)):
            window.tick()          # advance one video frame
            if not keep:
                continue
            fg_mask  = fg_masks[bni] if bni < len(fg_masks) else None
            fshape   = img.shape[:2]

            for tile_img, meta in split_image_gen(img, ...):
                tx, ty = meta["left_x"], meta["top_y"]
                if not window.gate_tile(fg_mask, fshape, tx, ty, tile_size):
                    continue       # skip — cold tile, no recent detections
                submit_for_inference(tile_img, meta)

        # after inference: for every tile that had detections
        window.mark_detection(meta["top_y"], meta["left_x"])

    Parameters
    ----------
    persist_frames : int
        Frames a tile stays active after its last trigger (motion or detection).
        5–10 frames works well for 15–25 fps sources.
    motion_min_ratio : float
        Minimum fraction of a tile's region (in the FG mask) that must be
        foreground to trigger the tile.  0.005 (0.5 %) is a conservative
        default that avoids shadow noise.
    """

    def __init__(
        self,
        persist_frames: int = 5,
        motion_min_ratio: float = 0.005,
    ) -> None:
        self.persist_frames = persist_frames
        self.motion_min_ratio = motion_min_ratio

        # (ty, tx) → global frame index when tile was last triggered
        self._last_active: dict[Tuple[int, int], int] = {}
        self._frame_idx: int = 0

        # Guard against camera resolution changes (new source, resize)
        self._last_frame_shape: Tuple[int, int] = (0, 0)

        # ── Research / diagnostic counters ──────────────────────────────────
        # Per-batch (reset at the start of every batch)
        self.batch_total: int = 0
        self.batch_submitted: int = 0
        # Cumulative (lifetime of the window)
        self.cumulative_total: int = 0
        self.cumulative_submitted: int = 0

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance one video frame.  Call once per frame in the batch loop,
        for *every* frame (including frames that fail the motion gate)."""
        self._frame_idx += 1

    def reset_batch_metrics(self) -> None:
        """Reset per-batch counters.  Call once at the start of each batch."""
        self.batch_total = 0
        self.batch_submitted = 0

    def reset(self) -> None:
        """Full reset — clears all track state and counters."""
        self._last_active.clear()
        self._frame_idx = 0
        self._last_frame_shape = (0, 0)
        self.batch_total = 0
        self.batch_submitted = 0
        self.cumulative_total = 0
        self.cumulative_submitted = 0

    # ------------------------------------------------------------------
    # Per-tile API
    # ------------------------------------------------------------------

    def is_active(self, ty: int, tx: int) -> bool:
        """True if the tile is within the persistence window."""
        last = self._last_active.get((ty, tx), -(self.persist_frames + 1))
        return (self._frame_idx - last) <= self.persist_frames

    def trigger(self, ty: int, tx: int) -> None:
        """Mark tile (ty, tx) as active now, resetting its persistence counter."""
        self._last_active[(ty, tx)] = self._frame_idx

    def mark_detection(self, ty: int, tx: int) -> None:
        """Call when inference found ≥1 detection in this tile.
        Extends the persistence window so the tile stays active next frame."""
        self.trigger(ty, tx)

    def gate_tile(
        self,
        fg_mask: Optional[np.ndarray],
        frame_shape: Tuple[int, int],
        tx: int,
        ty: int,
        tile_size: int,
    ) -> bool:
        """
        Decide whether to submit this tile for inference.

        Updates internal counters and triggers the tile if motion is found.

        Returns
        -------
        bool
            True  → submit tile to YOLO inference.
            False → skip tile (no motion, not in persistence window).
        """
        frame_h, frame_w = frame_shape

        # If the frame resolution changed, stale (ty,tx) keys are meaningless
        if frame_shape != self._last_frame_shape and frame_h > 0:
            self._last_active.clear()
            self._last_frame_shape = frame_shape

        # ── Check 1: FG mask motion ──────────────────────────────────────
        has_motion = False
        if fg_mask is None:
            # No mask available — fall back to always-active to avoid blind spots
            has_motion = True
        else:
            has_motion = _tile_has_motion(
                fg_mask, frame_h, frame_w, tx, ty, tile_size, self.motion_min_ratio
            )

        if has_motion:
            self.trigger(ty, tx)

        # ── Check 2: persistence window ──────────────────────────────────
        active = has_motion or self.is_active(ty, tx)

        # ── Counters ─────────────────────────────────────────────────────
        self.batch_total += 1
        self.cumulative_total += 1
        if active:
            self.batch_submitted += 1
            self.cumulative_submitted += 1

        return active

    # ------------------------------------------------------------------
    # Research metrics
    # ------------------------------------------------------------------

    @property
    def batch_skip_rate(self) -> float:
        """Fraction of tiles skipped in the current batch (0–1)."""
        return 1.0 - self.batch_submitted / max(self.batch_total, 1)

    @property
    def cumulative_skip_rate(self) -> float:
        """Fraction of tiles skipped since construction (0–1)."""
        return 1.0 - self.cumulative_submitted / max(self.cumulative_total, 1)

    def get_batch_metrics(self) -> dict:
        skipped = self.batch_total - self.batch_submitted
        return {
            "tiles_total": self.batch_total,
            "tiles_submitted": self.batch_submitted,
            "tiles_skipped": skipped,
            "tile_skip_rate": self.batch_skip_rate,
        }

    def get_cumulative_metrics(self) -> dict:
        skipped = self.cumulative_total - self.cumulative_submitted
        return {
            "cumulative_tiles_total": self.cumulative_total,
            "cumulative_tiles_submitted": self.cumulative_submitted,
            "cumulative_tiles_skipped": skipped,
            "cumulative_tile_skip_rate": self.cumulative_skip_rate,
        }
