from __future__ import annotations

import re
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

import cv2
import numpy as np

from obs_system.application_module.dummy_application.dummy_app import Application
from obs_system.application_module.dummy_application.intermediary import gui_connector
from obs_system.application_module.dummy_application.pipeline_config import (
    PipelineConfig,
    build_arg_parser,
)
from obs_system.utils.logger import get_logger, remove_logger

logger = get_logger(name="obs_system." + __name__)
remove_logger("matplotlib")
remove_logger("matplotlib.font_manager")

# ---------------------------------------------------------------------------
# Lightweight helpers (mirrors of experiment_ablation_study counterparts)
# ---------------------------------------------------------------------------

def _copy_img(arr: Any) -> np.ndarray | None:
    if arr is None:
        return None
    try:
        a = np.asarray(arr)
        return a.copy() if a.size > 0 else None
    except Exception:
        return None


def _copy_msk(arr: Any) -> np.ndarray | None:
    return _copy_img(arr)


@contextmanager
def _temp_method(obj: Any, name: str, fn: Callable[..., Any]) -> Iterator[None]:
    original = getattr(obj, name)
    setattr(obj, name, MethodType(fn, obj))
    try:
        yield
    finally:
        setattr(obj, name, original)


# ---------------------------------------------------------------------------
# Module-capture state
# ---------------------------------------------------------------------------

_FEP_VIEW_LABELS = ("top", "left", "right", "persp")


@dataclass
class ModuleCaptureState:
    enabled: bool = False
    output_dir: Path | None = None
    max_captures: int = 10
    capture_spacing: int = 50       # min frames between routine captures
    captured_count: int = 0
    last_captured_frame_id: int = -9999
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-frame stage image writer
# ---------------------------------------------------------------------------

def _write_img(path: Path, img: np.ndarray | None) -> None:
    if img is not None and img.size > 0:
        cv2.imwrite(str(path), img)


def _save_stage_images(
    output_dir: Path | None,
    frame_idx: int,
    artifacts: dict[str, Any],
    preds: Any,
    config: PipelineConfig,
) -> None:
    """Write one PNG per active pipeline stage for a single captured frame."""
    if output_dir is None:
        return
    pfx = str(output_dir / f"frame_{frame_idx:03d}")
    raw = artifacts.get("original_bgr")

    # --- S1: raw frame -------------------------------------------------------
    _write_img(Path(f"{pfx}_s1_raw.png"), raw)

    # --- S2: MOG2 motion mask + green overlay + lane mask --------------------
    fg = artifacts.get("fg_mask")
    if fg is not None:
        disp = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR) if fg.ndim == 2 else fg.copy()
        _write_img(Path(f"{pfx}_s2_motion_mask.png"), disp)
        if raw is not None:
            ov = raw.copy()
            fg_up = cv2.resize(fg, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
            ov[fg_up > 0] = (0, 220, 80)
            _write_img(Path(f"{pfx}_s2_motion_overlay.png"), ov)

    lane = (artifacts.get("scene_masks") or {}).get("lane_mask")
    if lane is not None:
        ld = cv2.cvtColor(lane, cv2.COLOR_GRAY2BGR) if lane.ndim == 2 else lane.copy()
        _write_img(Path(f"{pfx}_s2_lane_mask.png"), ld)

    # --- S3: FEP tangent views (one file per view) ---------------------------
    # --- S4: tiling (one file per tile) --------------------------------------
    views = artifacts.get("processed_views", [])
    kind = artifacts.get("views_kind", "single")
    if kind == "fep" and config.fep:
        for vi, v in enumerate(views):
            if v is None:
                continue
            label = _FEP_VIEW_LABELS[vi] if vi < len(_FEP_VIEW_LABELS) else f"v{vi}"
            _write_img(Path(f"{pfx}_s3_fep_{label}.png"), v)
    elif kind == "tiles" and config.force_tiles:
        for ti, t in enumerate(views):
            if t is not None:
                _write_img(Path(f"{pfx}_s4_tile_{ti}.png"), t)

    # --- S4: ROI-cropped frame -----------------------------------------------
    _write_img(Path(f"{pfx}_s4_roi.png"), artifacts.get("roi_bgr"))

    # --- S5/S6/S7: annotated detections + tracking (preds.plot()) -----------
    # Separation of raw inference (S5), class-filtered+NMS (S6), and tracker
    # (S7) would require deeper streamer hooks; preds.plot() reflects the
    # final post-tracking state which covers all three stages visually.
    try:
        boxes = getattr(getattr(preds, "boxes", None), "xyxy", None)
        if boxes is not None and boxes.numel() > 0:
            rgb = preds.plot(conf=True, line_width=2)
            if rgb is not None:
                _write_img(Path(f"{pfx}_s5_detections.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception:
        pass

    # --- S8: hazard event frame ----------------------------------------------
    if getattr(preds, "hazard_events", None):
        try:
            rgb = preds.plot(conf=True, line_width=2)
            if rgb is not None:
                hz = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                h, w = hz.shape[:2]
                cv2.rectangle(hz, (0, 0), (w, h), (0, 0, 255), 6)
                cv2.putText(hz, "HAZARD EVENT", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
                _write_img(Path(f"{pfx}_s8_hazard.png"), hz)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Capture hook installer
# ---------------------------------------------------------------------------

def _install_module_capture(
    exit_stack: ExitStack,
    *,
    state: ModuleCaptureState,
    streamer: Any,
    config: PipelineConfig,
) -> None:
    """Monkey-patch _stage_a_acquire_and_gate and postprocess to capture stage images."""
    if not state.enabled or streamer is None or state.output_dir is None:
        return

    original_stage_a = streamer._stage_a_acquire_and_gate
    original_postprocess = streamer.postprocess

    def stage_a_with_capture(
        self, batch_payload, frame_read_ms, stream_start, timeline_logger, batch_idx
    ):
        stage = original_stage_a(batch_payload, frame_read_ms, stream_start, timeline_logger, batch_idx)
        if stage.get("skip_reason") == "warmup":
            return stage

        frame_ids = stage.get("frame_ids", [])
        originals = stage.get("original_images_bgr", [])
        cropped_rgb = stage.get("cropped_original_images", [])
        processed = stage.get("im0s", [])
        fg_masks = stage.get("fg_masks", []) or []

        # n_per_orig > 1 when FEP or tiling expands one original frame into
        # multiple inference inputs.  Store all of them under the first view's
        # frame_id so postprocess sees them together.
        n_orig = max(len(originals), 1)
        n_proc = len(processed)
        n_per_orig = max(n_proc // n_orig, 1)
        views_kind = (
            "fep" if config.fep and n_per_orig > 1
            else "tiles" if n_per_orig > 1
            else "single"
        )

        for orig_idx in range(n_orig):
            proc_start = orig_idx * n_per_orig
            first_fid_idx = min(proc_start, len(frame_ids) - 1)
            fid = frame_ids[first_fid_idx] if first_fid_idx < len(frame_ids) else orig_idx
            views = [
                _copy_img(processed[v])
                for v in range(proc_start, min(proc_start + n_per_orig, n_proc))
            ]
            state.pending[str(fid)] = {
                "frame_id": fid,
                "original_bgr": _copy_img(originals[orig_idx]) if orig_idx < len(originals) else None,
                "roi_bgr": (
                    cv2.cvtColor(cropped_rgb[orig_idx], cv2.COLOR_RGB2BGR)
                    if orig_idx < len(cropped_rgb) else None
                ),
                "processed_views": views,
                "views_kind": views_kind,
                "fg_mask": _copy_msk(fg_masks[orig_idx]) if orig_idx < len(fg_masks) else None,
            }
        return stage

    def postprocess_with_capture(self, preds, orig_image):
        out = original_postprocess(preds, orig_image)

        # Secondary FEP/tile views share the first view's pending entry; their
        # frame_ids won't be found here, which is intentional.
        frame_id = str(self._extract_frame_id(out))
        artifacts = state.pending.pop(frame_id, None)
        if artifacts is None:
            return out

        scene_masks: dict[str, Any] = {}
        lsm = getattr(self, "last_scene_masks", None)
        if lsm:
            for key in ("lane_mask", "crosswalk_mask", "drivable_confidence_map"):
                scene_masks[key] = _copy_msk(lsm.get(key))
        artifacts["scene_masks"] = scene_masks

        try:
            frame_id_int = int(frame_id)
        except (ValueError, TypeError):
            frame_id_int = -1

        has_hazard = bool(getattr(out, "hazard_events", None))
        frames_since_last = frame_id_int - state.last_captured_frame_id
        should_capture = state.captured_count < state.max_captures and (
            state.captured_count == 0          # always take the very first passing frame
            or has_hazard                       # always take hazard frames regardless of spacing
            or frames_since_last >= state.capture_spacing
        )

        if should_capture:
            state.captured_count += 1
            _save_stage_images(state.output_dir, state.captured_count, artifacts, out, config)
            state.last_captured_frame_id = frame_id_int

        return out

    exit_stack.enter_context(_temp_method(streamer, "_stage_a_acquire_and_gate", stage_a_with_capture))
    exit_stack.enter_context(_temp_method(streamer, "postprocess", postprocess_with_capture))


# ---------------------------------------------------------------------------
# Output-directory helper
# ---------------------------------------------------------------------------

def _resolve_modules_dir(dir_arg: str, video_source: str) -> Path:
    if dir_arg:
        return Path(dir_arg)
    source = (video_source or "").strip()
    parsed = urlparse(source)
    if parsed.scheme and parsed.netloc:
        slug = (parsed.hostname or "stream").replace(".", "_")
    else:
        raw = Path(source).stem or "capture"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-") or "capture"
    return Path("runs") / "module_captures" / f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.debug("--- Initializing Application ---")
    argparser = build_arg_parser()
    argparser.add_argument(
        "--save-all-modules",
        action="store_true",
        default=False,
        dest="save_all_modules",
        help=(
            "Capture per-stage images for up to 10 frames spread across the run "
            "(first frame, every ~50 frames, and all hazard frames). "
            "Active stages are determined by the flags you pass: "
            "--roi (ROI crop), --fep (one PNG per tangent view), "
            "--force-tiles (one PNG per tile), background subtraction and lane "
            "mask are always captured, and detections+tracking+hazard are "
            "captured when present."
        ),
    )
    argparser.add_argument(
        "--save-modules-dir",
        type=str,
        default="",
        dest="save_modules_dir",
        help=(
            "Output directory for --save-all-modules images. "
            "Defaults to runs/module_captures/<video_stem>_<timestamp>/"
        ),
    )

    args = argparser.parse_args()
    config = PipelineConfig.from_namespace(args)

    logger.warning(
        "If you change the input video source, adjust the background subtractor image. "
        "Otherwise, it will classify all frames without movement"
    )

    if config.gui:
        gui_connector(config.host_address, config.port_address)
        return

    config = config.validate()

    app = Application()

    save_all = bool(getattr(args, "save_all_modules", False))
    capture_state: ModuleCaptureState | None = None
    modules_dir: Path | None = None

    if save_all:
        modules_dir = _resolve_modules_dir(
            str(getattr(args, "save_modules_dir", "") or ""),
            config.video_source,
        )
        modules_dir.mkdir(parents=True, exist_ok=True)
        capture_state = ModuleCaptureState(enabled=True, output_dir=modules_dir)
        logger.info("Module captures will be saved to: %s", modules_dir)

        # Patch app.run_app as an instance attribute so run_application finds it
        # during self.run_app(...).  At that point the streamer is already set up
        # by setup_model, so we can inject the capture hooks safely.
        _orig_run_app = app.run_app

        def _run_app_with_capture(producer_flag=None, preview_queue=None):
            if app.streamer is not None:
                with ExitStack() as _es:
                    _install_module_capture(
                        _es,
                        state=capture_state,
                        streamer=app.streamer,
                        config=config,
                    )
                    _orig_run_app(producer_flag=producer_flag, preview_queue=preview_queue)
            else:
                _orig_run_app(producer_flag=producer_flag, preview_queue=preview_queue)

        app.run_app = _run_app_with_capture

    app.run_application(config)

    if save_all and capture_state is not None and modules_dir is not None:
        logger.info(
            "Module capture complete: %d frame(s) saved to %s",
            capture_state.captured_count,
            modules_dir,
        )


if __name__ == "__main__":
    main()
