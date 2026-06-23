# Pipeline Visualization Script
### What to show at each stage as an image traverses the OBS pipeline

---

## S1 — Video Source
**Show:** Full-resolution raw frame from the roadside camera (BGR, unprocessed).

**Best image:** Any frame with a vehicle visible — ideally mid-road, clear daylight.
The frame should show sky, asphalt, lane markings, and at least one moving vehicle.

**Key detail to highlight:** Resolution (e.g. 1920×1080) and the fact nothing has been
done yet — this is the pixel-level ground truth before any algorithmic decision.

---

## S2 — Motion Gate (MOG2 Background Subtraction)
**Show:** The binary foreground mask at downscaled resolution (320×320).

**Best image:** Black background with sharp white blobs where vehicles are detected.
Optionally overlay the mask on the original frame (white cutout on the road).

**Key detail:** Point out that frames without sufficient foreground area (fg_score below
threshold) are dropped here and never reach inference — this is the primary CPU-side
efficiency gate. Also show the lane mask that accumulates over the calibration window
(yellow/amber overlay on the road region).

---

## S3 — Projection (FEP / Panorama) `[optional]`
**Show:** Tangent view sub-frames extracted from a fisheye or panoramic input.

**Best image:** Side-by-side comparison of the circular fisheye image and the 4
rectilinear perspective crops (TOP, LEFT, RIGHT, PERSPECTIVE). Each crop should
show recognisable road geometry with straight horizon lines, proving the un-distortion
worked correctly.

**Key detail:** The fisheye circle loses roughly 25% of pixels at the edges due to
the projection. The tangent views reduce distortion so the detector operates on
near-rectilinear geometry.

> Skip this slide entirely if running without `--fep` or `--panorama`.

---

## S4 — ROI Masking / Tiling
**Show:** The frame with the active ROI polygon drawn as a dashed green overlay.
If tiling is enabled, show the tile grid superimposed and highlight the active tile.

**Best image:** A trapezoid ROI that covers the road surface from horizon to near-field,
masking out the sky and irrelevant background. For the tiled view, show the 2×2 or 3×3
grid with the tile that contains the vehicle highlighted in a brighter border.

**Key detail:** The ROI is profile-specific (different for each camera installation).
Anything outside the polygon is zeroed before entering the model, reducing false
positives from trees and sky.

---

## S5 — Backend Inference `[LARGEST STAGE]`
**Show:** The full scene with raw, unfiltered bounding boxes drawn by YOLO.

**Best image:** All detections visible — include noise and low-confidence detections
(person, bird, shadow blobs) shown in dashed or yellow borders alongside confident
vehicle detections in solid green. Include confidence scores (0.0–1.0) next to each
box. Show the input tensor dimension in a corner badge: `[1, 3, 640, 640]`.

**Key detail to call out:**
- Model format in use: `.pt` (PyTorch), `.onnx` (ONNX Runtime), or `.engine` (TensorRT).
- LetterBox preprocessing preserves aspect ratio with grey padding.
- On Jetson: TensorRT FP16 reduces this from ~40 ms → ~8 ms per batch.

---

## S6 — Post-Processing (Class Filter + Cross-View NMS)
**Show:** The same scene as S5 but with rejected detections visually struck through
(red X) and only valid vehicle/obstacle classes remaining in clean green boxes.

**Best image:** Before/after side by side: S5 with 6 raw boxes vs S6 with 3 clean boxes.
Show the lane mask as a semi-transparent amber overlay confirming which region is the
drivable surface.

**Key detail:** Class whitelist = `{car, truck, bus, bike, bicycle, motorbike, motorcycle,
animal classes, debris classes}`. The cross-view NMS suppresses duplicate boxes that
arise when the same vehicle appears in two overlapping tile regions.

---

## S7 — Tracker (ByteTrack / SORT)
**Show:** The scene with colored bounding boxes, each labeled with a persistent track ID
(e.g. `T#3`, `T#7`). Draw the centroid trail as a series of fading dots going back 10–30
frames.

**Best image:** Two vehicles, each with a different color, clearly separated track IDs,
and visible trail history showing the direction of travel (entering or exiting the ROI).

**Key detail:** Track IDs are stable across frames — a vehicle entering the frame gets
one ID and keeps it until it exits. Kalman-predicted positions fill in missed detections
during brief occlusion (tile boundary, shadow, overlap).

---

## S8 — Hazard Logic
**Show:** The scene with a red semi-transparent hazard zone polygon overlaid on the
drivable area. Any tracked vehicle whose bounding box intersects the zone should have
a red outline and a warning indicator (⚠).

**Best image:** One vehicle outside the zone (yellow box, labelled "safe") and one
inside (red box, labelled "⚠ HAZARD"). Show the lane mask and crosswalk region
as faint green overlays underneath.

**Key detail:** The drivable confidence map (float32, `[0.0–1.0]`) combines:
1. Static lane mask (geometry prior)
2. Vehicle detection heatmap (where cars have driven)
3. Track trail heatmap (historical paths)
4. Subtracted unstable-motion regions (trees, flags)

Events generated here are what gets published to MQTT.

---

## S9 — Output
**Show:** Three simultaneous outputs side by side.

| Channel | What to show |
|---------|-------------|
| **MQTT** | A JSON/CBOR payload snippet: `{event: "obstacle_in_lane", track_id: 7, bbox: [...], ts: ...}` |
| **Preview** | A browser window running the Streamlit/FastAPI stream at ~10–15 FPS MJPEG |
| **Save** | File explorer showing `runs/obs_pipeline/` with saved `.mp4` and annotation `.jpg` files |

**Key detail:** The MQTT publisher uses CBOR binary encoding (not JSON) to minimise
wire size. The crop image (vehicle cutout) is JPEG-compressed before publish.
The preview stream throttles to a configurable FPS cap to avoid saturating the network.

---

## Feedback Loop — YOLO-Gated Lane Calibration
**Show:** An annotated diagram of the feedback arrow from S6 back to S2.

**What to explain:** After every batch where YOLO confirms at least one vehicle
(confidence ≥ threshold), the vehicle-confirmed frame indices are sent back to the
`Subtractor.notify_vehicle_detections()` method. The subtractor accumulates only those
frames into its lane calibration window — ignoring frames with tree motion, rain, or
shadow that would contaminate the lane mask.

**Best visual:** Show the final lane mask (white blobs on black) overlaid in amber on the
road, and contrast it with what the mask would look like if built from raw MOG2 output
without YOLO gating (noisier, with vegetation and sky artifacts included).

---

*Generated for the OBS System pipeline · roadside obstacle detection · Jetson / x86 deployment*
