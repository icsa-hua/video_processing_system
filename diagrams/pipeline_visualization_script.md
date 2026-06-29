# Pipeline Visualization Script
### What each stage shows as a frame traverses the OBS pipeline
### Based on: `Final_Video_EdgeAI.mp4` — 31 s · 3840×2124 · 60 fps

---

## S0 — WebUI Config
**Show:** Streamlit UI titled "Real-Time Intersection Intelligence" with the configuration panel open.

**What's visible:** Model set to `yolov8s.engine`, source `rtsp / .mp4`, ROI enabled, FEP disabled, `conf 0.35 / iou 0.45`, and the "Launch Pipeline" button. Below the config form, a live fisheye thumbnail (Camera 01, overhead intersection) confirms the stream is connected.

**Key detail:** This is the only human-in-the-loop step. Everything after "Launch" is fully automated.

---

## S1 — Video Source
**Show:** Full fisheye frame from Camera 01 — overhead intersection, daytime, suburban campus setting.

**What's visible:** Circular fisheye image, heavy barrel distortion, green vegetation at edges, road surface and parking area in center. Timestamp `02-21-210x · 14:47:xx` in the top-left corner confirms live RTSP metadata is embedded.

**Key detail:** The image is `H×W×3 BGR` straight from `cv2.VideoCapture`. No processing has occurred — this is pixel-level ground truth.

---

## S2 — ROI Crop
**Show:** The same fisheye frame with the trapezoid ROI region highlighted, plus the cropped sub-frame in the card miniature.

**What's visible:** Full circular fisheye on the left, card thumbnail shows the before/after split — full frame vs. the bbox crop that isolates the road surface. Sky and peripheral vegetation are excluded.

**Key detail:** The crop is profile-specific (`cfg: ROI profile (trapezoid)`). Everything outside the polygon is discarded before motion analysis, keeping the MOG2 background model stable.

---

## S3 — Motion Gate
**Show:** Fisheye frame with a bright green blob overlaid on a moving object, and the MOG2 score badge `fg_score: 4.3% YES` in the card.

**What's visible:** The frame passes the gate (score above threshold → `YES`). The green highlight marks the connected-component foreground region. The card miniature shows the binary mask (white blobs on black) alongside the YOLO-gated calibration indicator. Frames below the threshold are silently dropped here — they never reach inference.

**Key detail:** MOG2 + connected-component filter runs at 320×320 on the cropped sub-frame. This is the primary CPU-side efficiency gate.

---

## S4 — FEP Projection · Tiling
**Show:** Four rectilinear tiles arranged in a 2×2 grid — the fisheye dewarped into four perspective views.

**What's visible:** Top-left tile shows road curb and pavement; top-right shows the road with lane markings; bottom-left shows the building edge and road surface; bottom-right shows road detail. All four tiles have straight horizon lines, confirming successful fisheye un-distortion. The card miniature shows the FEP pipeline (fisheye circle → tangent crops) alongside the tiling grid schematic.

**Key detail:** FEP (`--fep`) and Tiling (`--force-tiles`) are independent flags. Together they feed a 2×N tile batch to the detector instead of one distorted fisheye image, improving detection accuracy at the edges of the scene.

---

## S5 — Backend Inference
**Show:** Fisheye frame with the ROI circle drawn in blue and a raw YOLO bounding box: `id:2 car 0.77`.

**What's visible:** The blue circle traces the active ROI. One vehicle box is drawn with track ID and confidence. The card miniature shows the model format badges (`.pt`, `.onnx`, `.engine`) and the letterbox tensor label `[B, 3, 640, 640] FP32/FP16 · YOLOv8s`.

**Key detail:** Input is LetterBox-padded to preserve aspect ratio before the `[1, 3, 640, 640]` tensor is passed to the engine. On Jetson, TensorRT FP16 reduces latency from ~40 ms → ~8 ms per batch.

---

## S6 — Lane Detection
**Show:** Large white silhouette blob on a black background — the accumulated MOG2 lane mask after 847 vehicle-confirmed frames.

**What's visible:** The blob shape matches a car footprint, representing where vehicles consistently appear in the calibration window. The card miniature shows the road scene with two green lane lines and the status overlay: `acc. frames: 847 · confidence: HIGH`.

**Key detail:** Only frames where YOLO confirmed a vehicle (via `notify_vehicle_detections`) feed the accumulator. Raw MOG2 frames contaminated by tree motion, rain, or shadows are excluded — this is the YOLO-gated calibration feedback loop (S7 → S3 → S6).

---

## S7 — Post-Process
**Show:** Fisheye frame with multiple raw YOLO boxes from the full scene, class labels and confidence scores visible.

**What's visible:** Five detections simultaneously: `id:124 car 0.91`, `id:95 car 0.80`, `id:116 cor 0.66`, `id:157 truck 0.60`, `id:162 cor 0.51`. Boxes are colored by class (cyan for truck, white/yellow for cars). The card miniature shows the NMS diagram with a red-crossed `person` box being rejected while `car` and `truck` remain.

**Key detail:** Class whitelist keeps only vehicle and obstacle classes. Cross-view NMS suppresses duplicate boxes that appear across overlapping tile boundaries.

---

## S8 — Tracker
**Show:** Rectilinear (FEP-corrected) overhead view of the intersection with ByteTrack ID labels and blue centroid trails.

**What's visible:** `id:120 car 0.75` and `id:124 car` visible from above with persistent blue trail lines showing the direction and path of travel across the frame. The view is noticeably less distorted than the fisheye frames — this is the FEP tile being used for tracking. The card miniature shows T#3 and T#7 with fading dot trails.

**Key detail:** Track IDs are stable across frames. Kalman-predicted positions fill missed detections during brief occlusion. Trail history is 30 frames.

---

## S9 — Hazard Detection
**Show:** Rectilinear intersection view with a live hazard event: `Person in Lane! HIGH` alert overlaid on the frame.

**What's visible:** `id:211 car 0.62` tracked safely outside the lane polygon. A pedestrian (tracked separately) has entered the drivable zone — highlighted with a blue hazard polygon and the `Person in Lane! HIGH` text label. The card miniature shows the red hazard zone between the green lane lines with the warning triangle (⚠) and `HAZ T#7` badge.

**Key detail:** Dwell counter increments each frame a track's bbox overlaps the lane polygon. Threshold crossing here emits the `obstacle_in_lane` or `slow_vehicle` HazardEvent that triggers the MQTT publish.

---

## Video Comparison — Sped-Up Pipeline Demo (~23 s clip)
**Show:** Side-by-side synchronized comparison: left panel (A — Input/Raw) vs right panel (B — Pipeline Output).

**Left panel (A — Input/Raw):**
Fisheye Camera 01 overhead view running at speed, showing the full detection overlay in real-time: multiple cars, a truck, and a pedestrian tracked simultaneously across the frame. All raw YOLO boxes and track IDs are visible on the distorted fisheye image, demonstrating the volume and density of detections the pipeline handles per second.

**Right panel (B — Pipeline Output):**
FEP-corrected rectilinear view of the same scene — a road intersection with proper perspective, lane markings visible. Detection boxes rendered on the corrected geometry: `id:1 stop sign` (persistent across most frames), `car` (multiple), and `id:52 person` entering the lane zone (conf 0.63–0.81). The blue lane polygon and hazard region are visible as a semi-transparent overlay on the road surface. Confidence scores update frame-to-frame, demonstrating tracker stability.

**What the comparison proves:** The left panel shows the raw, distorted world the pipeline ingests; the right panel shows the geometrically corrected, annotated output a downstream system would consume. The stop sign remains stably tracked throughout, cars maintain consistent IDs as they transit the intersection, and the pedestrian triggers the lane-intrusion overlay at the moment of entry.

**Key detail:** Both panels share a single seek bar (0:00–0:23), confirming frame-accurate synchronization. The clip is sped up relative to real-time to fit within the demo window while still covering multiple full vehicle transits and one hazard event.

---

## Feedback Loop — YOLO-Gated Lane Calibration
**Show:** The dashed purple arrow in the pipeline diagram connecting S7 back to S3.

**What to explain:** After each batch where YOLO confirms at least one vehicle above threshold, the confirmed frame indices are fed back to `Subtractor.notify_vehicle_detections()`. The subtractor only adds those frames to its lane calibration window — ignoring tree motion, rain, and shadow frames that would corrupt the lane mask. The accumulated result is the clean white-silhouette mask shown in S6.

---

*OBS System · Roadside Obstacle Detection · Fisheye / FEP · Jetson / x86 deployment*
