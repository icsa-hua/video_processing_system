# Global configuration file for Thresholds and configuration variables.

import json
import logging
from pathlib import Path

#---------- Detection Thresholds ------------
CONF_THR = 0.50
NMS_IOU = 0.45 
CLASS_AGNOSTIC = True 


#---------- Image Tiling parameters ---------
TILE_SIZE = 640
TILE_OVERLAP = 0.30
TILE_NMS_IOU = 0.50
TILE_THR = 3

#--------- Defish Parameters ---------
DEFISH_K = 0.35
DEFISH_CROP = 0.05
DISTORTION_STRENGTH = -0.20
DEFISH_ALPHA = 0.5
DEFISH_BETA = 10
# Equidistant circular fisheye model ("barrel" = old polynomial, "equidistant" = circular fisheye)
DEFISH_MODEL = "equidistant"
DEFISH_FISHEYE_FOV_DEG = 180.0   # total angular span of the fisheye lens
DEFISH_OUTPUT_FOV_DEG = 90.0     # legacy single-view rectilinear output FOV

#--------- Fisheye Tangent-View Reprojection ----------
FISHEYE_N_VIEWS = 6
FISHEYE_VIEW_SIZE = (640, 640)
FISHEYE_VIEW_FOV_DEG = 70.0
FISHEYE_VIEW_TILT_DEG = 50.0
FISHEYE_VIEW_YAWS_DEG = [0, 60, 120, 180, 240, 300]
FISHEYE_VIEW_MIN_MOTION_FRACTION = 0.005
FISHEYE_VIEW_NMS_IOU = 0.2
FISHEYE_LOWER_VIEW_YAWS_DEG = [0, 240, 300]
FISHEYE_LOWER_VIEW_CONF_THR = 0.55

#--------- Motion Gating ----------
TRIALS = 10 
HISTORY = 300 
VARTHRESHOLD = 32
THR_RATIO = 0.2 
K_CONSECUTIVE = 3 
HOLD_FRAMES = 10
MIN_OBJ_AREA = 0.003
EMPTY_IMAGE_PATH = "samples/highway_rescaled.png" 

#--------- Frame Batching ---------
BATCH_SIZE = 16
WARM_UP_SESSIONS = 8
FIXED_WIDTH = 1245
FIXED_HEIGHT = 1088


#-------- ROI COORDINATE TO CROP INITIAL IMAGE --------
ROI_X1 = 0 
ROI_Y1 = 639 
ROI_X2 = 430 
ROI_Y2 = 300
REGION_COLOR = (255, 42, 4) 
ROI_REFERENCE_WIDTH = TILE_SIZE
ROI_REFERENCE_HEIGHT = TILE_SIZE
ROI_PROFILES_PATH = Path(__file__).with_name("roi_profiles.json")


#-------- False Positives -------
MIN_WH = 16 
MULTIPLIER = 0.8


#-------- Object Vocabulary -----
VOCAB = ['car', 'bus', 'bike',
         'motorbike', 'motorcycle',
         'truck', 'cyclist', 'motorcyclist']

RED = (0, 0, 255)

#--------- Motion-Derived Detection (allow_spawn) ----------
# Confidence assigned to MOG2 foreground blobs merged alongside YOLO detections.
# Must stay BELOW sv.ByteTrack.track_activation_threshold (default 0.25) so the blobs
# enter only the second-round matching pass — reinforcing existing tracks but never
# spawning new ones. This is the allow_spawn=False behaviour from CarDet_Dummy_EdgeAI.
MOTION_BOX_CONFIDENCE = 0.25
# Blob area as fraction of the fg_mask (downscale) total pixels
MOTION_BOX_MIN_AREA_RATIO = 0.003
MOTION_BOX_MAX_AREA_RATIO = 0.40
# Bounding-box w/h aspect bounds — rejects needle-thin vegetation streaks
MOTION_BOX_MIN_ASPECT = 0.4
MOTION_BOX_MAX_ASPECT = 7.0

#--------- Motion Quality Filtering (Step 1) ----------
# Fraction of total frame area; blobs smaller than this are discarded as noise
MIN_MOTION_COMPONENT_AREA_RATIO = 0.0002
# If the number of passing blobs exceeds this, assume vegetation scatter → suppress
MAX_MOTION_COMPONENTS = 25
# Morphological kernel size (pixels) used for the motion-gate cleanup pass
MOTION_MORPH_KERNEL = 5
# Minimum filtered-and-weighted motion score to pass the gate
MOTION_GATE_FILTERED_SCORE_THRESHOLD = 0.002

#--------- Unstable Motion Map (Step 2) ----------
ENABLE_UNSTABLE_MOTION_MAP = True
# Fraction of warmup frames in which a pixel must fire to be labelled "unstable"
UNSTABLE_MOTION_THRESHOLD = 0.40
# Contribution weight of an unstable pixel toward the motion score (0 = fully suppressed)
UNSTABLE_MOTION_SUPPRESSION_WEIGHT = 0.20

#--------- Drivable-Area Confidence Map (Step 3) ----------
ENABLE_DRIVABLE_CONFIDENCE_MAP = False
# Per-source contribution weights (must sum <= 1.0; remainder left to runtime learning)
DRIVABLE_STATIC_WEIGHT = 0.50
DRIVABLE_DETECTION_WEIGHT = 0.25
DRIVABLE_TRACK_WEIGHT = 0.15
DRIVABLE_UNSTABLE_NEGATIVE_WEIGHT = 0.10
# Pixels below this confidence are treated as "not reliably drivable"
DRIVABLE_CONFIDENCE_THRESHOLD = 0.35


#--------- Panorama Perspective Reprojection ----------
# Number of tangent/pinhole views to generate from an equirectangular panorama
PANORAMA_N_VIEWS = 4
# Output size (width, height) for each perspective view fed to YOLO
PANORAMA_VIEW_SIZE = (640, 640)
# Horizontal FOV in degrees for each perspective view (pinhole tangent-plane view)
PANORAMA_VIEW_FOV_DEG = 75.0
# Pitch offset for all views in degrees (negative = tilt down toward road)
PANORAMA_PITCH_DEG = 10.0
# Assumed total horizontal angular span of the panorama image in degrees
PANORAMA_HFOV_DEG = 180.0
# Assumed total vertical angular span of the panorama image in degrees
PANORAMA_VFOV_DEG = 70.0
# Fraction of a view's panorama coverage that must contain motion to activate the view
PANORAMA_MIN_MOTION_FRACTION = 0.001
# Cross-view NMS IoU threshold (applied after back-projecting all view detections to panorama space)
PANORAMA_NMS_IOU = 0.45

#--------- Tile Activation Persistence Window ----------
# Set to False to disable per-tile motion gating and fall back to whole-frame inference.
TILE_ACTIVATION_ENABLED = True
# Frames a tile stays active after its last motion trigger or detection hit.
TILE_ACTIVATION_PERSIST_FRAMES = 5
# Minimum fraction of a tile's FG-mask region that must be foreground to trigger it.
TILE_ACTIVATION_MOTION_MIN_RATIO = 0.025

#--------- Tile / Panorama Kalman Detection Smoother ----------
# Maximum frames a Kalman track survives without a matching detection.
TILE_KALMAN_MAX_AGE = 5
# Minimum detections before a track contributes Kalman-predicted boxes.
TILE_KALMAN_MIN_HITS = 1
# Minimum IoU to associate a detection with a Kalman-predicted track position.
TILE_KALMAN_IOU_THRESHOLD = 0.45

#--------- Road-Scene Class Filter ----------
# Classes that cannot appear in a road traffic scene — removed from all detections.
# Applied in both panorama and standard inference paths to reduce impossible-class FPs.
ROAD_IMPOSSIBLE_CLASSES = frozenset({
    "boat", "ship", "surfboard", "snowboard", "skis",
    "train", "airplane", "aeroplane", "helicopter",
    "submarine", "kite", "bird",
})


logger = logging.getLogger("obs_system." + __name__)

DEFAULT_ROI_CONFIG = {
    "x1": ROI_X1,
    "y1": ROI_Y1,
    "x2": ROI_X2,
    "y2": ROI_Y2,
    "reference_width": ROI_REFERENCE_WIDTH,
    "reference_height": ROI_REFERENCE_HEIGHT,
}
DEFAULT_MOTION_CONFIG = {
    "var_threshold": VARTHRESHOLD,
    "gate_score_threshold": MOTION_GATE_FILTERED_SCORE_THRESHOLD,
    "max_components": MAX_MOTION_COMPONENTS,
}
ACTIVE_ROI_CONFIG = dict(DEFAULT_ROI_CONFIG)
ACTIVE_MOTION_CONFIG = dict(DEFAULT_MOTION_CONFIG)
ACTIVE_ROI_PROFILE = "default"


def _normalize_roi_profile(raw_profile, profile_name: str) -> dict[str, int]:
    if not isinstance(raw_profile, dict):
        raise ValueError(f"ROI profile '{profile_name}' must be a JSON object.")

    try:
        normalized = {
            "x1": int(raw_profile["x1"]),
            "y1": int(raw_profile["y1"]),
            "x2": int(raw_profile["x2"]),
            "y2": int(raw_profile["y2"]),
            "reference_width": int(raw_profile.get("reference_width", ROI_REFERENCE_WIDTH)),
            "reference_height": int(raw_profile.get("reference_height", ROI_REFERENCE_HEIGHT)),
        }
    except KeyError as exc:
        raise ValueError(f"ROI profile '{profile_name}' is missing field {exc.args[0]!r}.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ROI profile '{profile_name}' contains non-integer values.") from exc

    if normalized["reference_width"] <= 0 or normalized["reference_height"] <= 0:
        raise ValueError(f"ROI profile '{profile_name}' must define positive reference dimensions.")

    return normalized


def _normalize_motion_profile(raw_motion, profile_name: str) -> dict[str, float | int]:
    if raw_motion is None:
        return dict(DEFAULT_MOTION_CONFIG)

    if not isinstance(raw_motion, dict):
        raise ValueError(f"Motion profile for '{profile_name}' must be a JSON object.")

    try:
        normalized = {
            "var_threshold": int(raw_motion.get("var_threshold", VARTHRESHOLD)),
            "gate_score_threshold": float(
                raw_motion.get("gate_score_threshold", MOTION_GATE_FILTERED_SCORE_THRESHOLD)
            ),
            "max_components": int(raw_motion.get("max_components", MAX_MOTION_COMPONENTS)),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Motion profile for '{profile_name}' contains invalid values.") from exc

    if normalized["var_threshold"] <= 0:
        raise ValueError(f"Motion profile for '{profile_name}' must define a positive var_threshold.")
    if normalized["gate_score_threshold"] < 0.0:
        raise ValueError(f"Motion profile for '{profile_name}' must define a non-negative gate_score_threshold.")
    if normalized["max_components"] <= 0:
        raise ValueError(f"Motion profile for '{profile_name}' must define a positive max_components.")

    return normalized


def _normalize_video_profile(raw_profile, profile_name: str) -> dict[str, object]:
    roi_config = _normalize_roi_profile(raw_profile, profile_name)
    motion_config = _normalize_motion_profile(raw_profile.get("motion"), profile_name)
    return {
        **roi_config,
        "motion": motion_config,
    }


def load_roi_profiles() -> dict[str, dict[str, object]]:
    profiles = {
        "default": {
            **dict(DEFAULT_ROI_CONFIG),
            "motion": dict(DEFAULT_MOTION_CONFIG),
        }
    }

    if not ROI_PROFILES_PATH.exists():
        return profiles

    try:
        payload = json.loads(ROI_PROFILES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load ROI profile file %s: %s", ROI_PROFILES_PATH, exc)
        return profiles

    if not isinstance(payload, dict):
        logger.warning("ROI profile file %s must contain a JSON object at the top level.", ROI_PROFILES_PATH)
        return profiles

    for profile_name, raw_profile in payload.items():
        try:
            profiles[str(profile_name)] = _normalize_video_profile(raw_profile, str(profile_name))
        except ValueError as exc:
            logger.warning("%s", exc)

    return profiles


def _candidate_roi_profile_keys(video_source: str = "", roi_profile: str = "") -> list[str]:
    keys: list[str] = []

    def _append(value: object) -> None:
        key = str(value or "").strip()
        if key and key not in keys:
            keys.append(key)

    _append(roi_profile)
    _append(video_source)

    source = str(video_source or "").strip()
    if not source or "://" in source:
        return keys

    source_path = Path(source)
    _append(source_path.as_posix())
    _append(source_path.name)
    _append(source_path.stem)

    try:
        resolved = source_path.resolve(strict=False)
    except OSError:
        resolved = None

    if resolved is not None:
        _append(resolved.as_posix())
        _append(resolved.name)
        _append(resolved.stem)

    return keys


def resolve_roi_profile(video_source: str = "", roi_profile: str = "") -> tuple[str, dict[str, object]]:
    profiles = load_roi_profiles()
    lowercase_profiles = {name.lower(): name for name in profiles}

    for candidate in _candidate_roi_profile_keys(video_source=video_source, roi_profile=roi_profile):
        exact = profiles.get(candidate)
        if exact is not None:
            return candidate, dict(exact)

        lowered = lowercase_profiles.get(candidate.lower())
        if lowered is not None:
            return lowered, dict(profiles[lowered])

    return "default", dict(profiles["default"])


def set_active_roi(video_source: str = "", roi_profile: str = "") -> dict[str, int]:
    global ACTIVE_ROI_CONFIG, ACTIVE_MOTION_CONFIG, ACTIVE_ROI_PROFILE

    ACTIVE_ROI_PROFILE, profile = resolve_roi_profile(
        video_source=video_source,
        roi_profile=roi_profile,
    )
    ACTIVE_ROI_CONFIG = {
        "x1": int(profile["x1"]),
        "y1": int(profile["y1"]),
        "x2": int(profile["x2"]),
        "y2": int(profile["y2"]),
        "reference_width": int(profile["reference_width"]),
        "reference_height": int(profile["reference_height"]),
    }
    ACTIVE_MOTION_CONFIG = dict(profile.get("motion", DEFAULT_MOTION_CONFIG))
    return dict(ACTIVE_ROI_CONFIG)


def get_active_roi_config() -> dict[str, int]:
    return dict(ACTIVE_ROI_CONFIG)


def get_active_roi() -> tuple[int, int, int, int]:
    return (
        ACTIVE_ROI_CONFIG["x1"],
        ACTIVE_ROI_CONFIG["y1"],
        ACTIVE_ROI_CONFIG["x2"],
        ACTIVE_ROI_CONFIG["y2"],
    )


def get_active_roi_profile() -> str:
    return ACTIVE_ROI_PROFILE


def get_active_motion_config() -> dict[str, float | int]:
    return dict(ACTIVE_MOTION_CONFIG)
