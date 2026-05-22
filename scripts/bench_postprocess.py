#!/usr/bin/env python3
"""
Micro-benchmark for the postprocess hot-path components.

Measures:
  1. analyze_lane_hazards  – vectorised vs old-style per-box loop
  2. get_scene_masks       – reference (expand_px=0) vs copy path
  3. _prepare_scene_mask_cache – integral pre-computation

Run from the repo root:
    python scripts/bench_postprocess.py

Optional args (env vars):
    BENCH_N=500   number of iterations per test (default 500)
    BENCH_H=1088  frame height
    BENCH_W=1245  frame width
    BENCH_BOXES=30  detections per frame
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

N = int(os.environ.get("BENCH_N", 500))
H = int(os.environ.get("BENCH_H", 1088))
W = int(os.environ.get("BENCH_W", 1245))
N_BOXES = int(os.environ.get("BENCH_BOXES", 30))
BATCH_SIZE = int(os.environ.get("BENCH_BATCH", 16))

_SEP = "-" * 60


def _make_lane_mask(h=H, w=W):
    m = np.zeros((h, w), dtype=np.uint8)
    m[h // 3 : h - h // 6, w // 6 : w - w // 6] = 255
    return m


def _make_boxes(n=N_BOXES, h=H, w=W):
    x1 = np.random.uniform(0, w - 10, n).astype(np.float32)
    y1 = np.random.uniform(0, h - 10, n).astype(np.float32)
    x2 = np.clip(x1 + np.random.uniform(20, 200, n), 0, w).astype(np.float32)
    y2 = np.clip(y1 + np.random.uniform(20, 200, n), 0, h).astype(np.float32)
    return np.stack([x1, y1, x2, y2], axis=1)


def _run(fn, n_iters, label):
    # warm up
    for _ in range(max(1, n_iters // 10)):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    elapsed = (time.perf_counter() - t0) * 1e3
    avg = elapsed / n_iters
    print(f"  {label:<50s}  {avg:7.3f} ms/call  ({elapsed:.1f} ms total / {n_iters} iters)")
    return avg


# ---------------------------------------------------------------------------
# 1. analyze_lane_hazards
# ---------------------------------------------------------------------------
def bench_hazard_analysis():
    print(_SEP)
    print("analyze_lane_hazards")
    print(_SEP)
    from obs_system.logic_module.dummy_logic.obstacle_filtering import (
        analyze_lane_hazards,
        _mask_integral,
        _lane_bounds,
    )

    lane = _make_lane_mask()
    crosswalk = np.zeros_like(lane)
    crosswalk[H // 2 : H // 2 + 60, W // 3 : W // 3 + 200] = 255

    boxes = _make_boxes()
    classes = np.random.randint(0, 80, N_BOXES)
    class_names = [f"class_{i}" for i in range(80)]
    class_names[0] = "person"
    class_names[1] = "car"

    lane_integral = _mask_integral(lane)
    crosswalk_integral = _mask_integral(crosswalk)
    lane_bbox = _lane_bounds(lane)

    _run(
        lambda: analyze_lane_hazards(
            boxes=boxes,
            classes=classes,
            class_names=class_names,
            lane_mask=lane,
            crosswalk_mask=crosswalk,
            lane_bbox=lane_bbox,
            lane_integral=lane_integral,
            crosswalk_integral=crosswalk_integral,
            lane_nonzero=True,
            crosswalk_nonzero=True,
        ),
        N,
        f"analyze_lane_hazards ({N_BOXES} boxes, integral)",
    )

    _run(
        lambda: analyze_lane_hazards(
            boxes=boxes,
            classes=classes,
            class_names=class_names,
            lane_mask=lane,
            crosswalk_mask=crosswalk,
            lane_bbox=lane_bbox,
            lane_integral=None,   # no integral – fallback path
            crosswalk_integral=None,
            lane_nonzero=True,
            crosswalk_nonzero=True,
        ),
        N,
        f"analyze_lane_hazards ({N_BOXES} boxes, no integral / fallback)",
    )

    # Torch tensor input (common in real pipeline)
    boxes_t = torch.from_numpy(boxes)
    classes_t = torch.from_numpy(classes.astype(np.int64))
    _run(
        lambda: analyze_lane_hazards(
            boxes=boxes_t,
            classes=classes_t,
            class_names=class_names,
            lane_mask=lane,
            crosswalk_mask=crosswalk,
            lane_bbox=lane_bbox,
            lane_integral=lane_integral,
            crosswalk_integral=crosswalk_integral,
            lane_nonzero=True,
            crosswalk_nonzero=True,
        ),
        N,
        f"analyze_lane_hazards ({N_BOXES} boxes, torch input, integral)",
    )


# ---------------------------------------------------------------------------
# 2. get_scene_masks  – copy vs reference
# ---------------------------------------------------------------------------
def bench_scene_masks():
    print()
    print(_SEP)
    print("get_scene_masks  (Subtractor)")
    print(_SEP)
    from obs_system.logic_module.dummy_logic.subtractor import Subtractor

    sub = Subtractor()
    sub.lanes_mask = _make_lane_mask()
    sub.crosswalk_mask = np.zeros_like(sub.lanes_mask)
    sub._last_frame_shape = (H, W)

    _run(lambda: sub.get_scene_masks(expand_px=0), N, "get_scene_masks(expand_px=0)  [reference, no copy]")
    _run(lambda: sub.get_scene_masks(expand_px=14), N, "get_scene_masks(expand_px=14) [copy + dilation]")

    # Simulate what _resolve_scene_masks does per frame in a batch
    def _resolve_like_old():
        masks = sub.get_scene_masks(expand_px=0)
        _ = masks.get("lane_mask")
        _ = masks.get("crosswalk_mask")

    _run(_resolve_like_old, N, f"_resolve_scene_masks equiv × 1 frame")

    n_batch = BATCH_SIZE
    def _resolve_batch_old():
        for _ in range(n_batch):
            masks = sub.get_scene_masks(expand_px=0)
            _ = masks.get("lane_mask")
    def _resolve_batch_new():
        masks = sub.get_scene_masks(expand_px=0)  # once
        _ = masks.get("lane_mask")

    _run(_resolve_batch_old, N // n_batch, f"get_scene_masks × {n_batch} (old – per frame in batch)")
    _run(_resolve_batch_new, N // n_batch, f"get_scene_masks × 1  (new – once per batch)")


# ---------------------------------------------------------------------------
# 3. _prepare_scene_mask_cache
# ---------------------------------------------------------------------------
def bench_scene_cache():
    print()
    print(_SEP)
    print("_prepare_scene_mask_cache  (cache hit vs cold)")
    print(_SEP)
    from obs_system.logic_module.dummy_logic.obstacle_filtering import _mask_integral, _lane_bounds

    lane = _make_lane_mask()
    crosswalk = np.zeros_like(lane)
    scene_masks = {"lane_mask": lane, "crosswalk_mask": crosswalk}

    # Simulate the cache structure
    lane_integral = _mask_integral(lane)
    lane_bbox = _lane_bounds(lane)
    crosswalk_integral = _mask_integral(crosswalk)

    def cold_prep():
        _ = cv2.countNonZero(lane)
        _ = cv2.countNonZero(crosswalk)
        _ = _mask_integral(lane)
        _ = _mask_integral(crosswalk)
        _ = _lane_bounds(lane)

    def warm_prep():
        # Cache hit – just the id() checks + dict lookup
        _ = id(lane)
        _ = id(crosswalk)

    _run(cold_prep, N, "cache COLD (countNonZero + integral + bounds)")
    _run(warm_prep, N, "cache HIT  (id() check only)")


# ---------------------------------------------------------------------------
# 4. End-to-end per-frame postprocess simulation
# ---------------------------------------------------------------------------
def bench_postprocess_simulation():
    print()
    print(_SEP)
    print("Simulated postprocess() per frame")
    print(_SEP)
    from obs_system.logic_module.dummy_logic.obstacle_filtering import (
        analyze_lane_hazards, _mask_integral, _lane_bounds,
    )
    from obs_system.logic_module.dummy_logic.subtractor import Subtractor

    sub = Subtractor()
    sub.lanes_mask = _make_lane_mask()
    sub.crosswalk_mask = np.zeros_like(sub.lanes_mask)
    sub._last_frame_shape = (H, W)

    lane_integral = _mask_integral(sub.lanes_mask)
    lane_bbox = _lane_bounds(sub.lanes_mask)
    crosswalk_integral = _mask_integral(sub.crosswalk_mask)

    boxes = _make_boxes()
    classes_np = np.random.randint(0, 80, N_BOXES)
    class_names = [f"class_{i}" for i in range(80)]
    class_names[0] = "person"

    scene_cache = {
        "lane_mask_proc": sub.lanes_mask,
        "crosswalk_mask_proc": sub.crosswalk_mask,
        "lane_integral": lane_integral,
        "crosswalk_integral": crosswalk_integral,
        "lane_bbox": lane_bbox,
        "lane_nonzero": True,
        "crosswalk_nonzero": False,
        "hazard_scale": 1.0,
    }

    def per_frame_old():
        # Old: get_scene_masks copies + prepare_cache per frame
        masks = sub.get_scene_masks(expand_px=0)
        _ = cv2.countNonZero(masks["lane_mask"])   # simulates prepare_cache cold
        analyze_lane_hazards(
            boxes=boxes, classes=classes_np, class_names=class_names,
            lane_mask=masks["lane_mask"], crosswalk_mask=masks["crosswalk_mask"],
            lane_integral=None, lane_nonzero=True, crosswalk_nonzero=False,
        )

    def per_frame_new():
        # New: shared cache, no copy, vectorised hazard analysis
        analyze_lane_hazards(
            boxes=boxes, classes=classes_np, class_names=class_names,
            lane_mask=scene_cache["lane_mask_proc"],
            crosswalk_mask=scene_cache["crosswalk_mask_proc"],
            lane_bbox=lane_bbox,
            lane_integral=lane_integral,
            crosswalk_integral=crosswalk_integral,
            lane_nonzero=True,
            crosswalk_nonzero=False,
        )

    avg_old = _run(per_frame_old, N, "OLD  per-frame (copy + no integral)")
    avg_new = _run(per_frame_new, N, "NEW  per-frame (shared cache + integral)")
    if avg_old > 0:
        print(f"\n  Speedup per frame:  {avg_old / avg_new:.2f}×")
        batch = BATCH_SIZE
        saved = (avg_old - avg_new) * batch
        print(f"  Estimated saving over batch of {batch}: {saved:.1f} ms")


if __name__ == "__main__":
    print(f"\nBenchmark config: N={N}, frame={W}×{H}, boxes={N_BOXES}, batch={BATCH_SIZE}\n")
    bench_hazard_analysis()
    bench_scene_masks()
    bench_scene_cache()
    bench_postprocess_simulation()
    print()
    print("Done.")
