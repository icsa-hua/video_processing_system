"""Plot extended performance diagnostics from an instrumented streaming pipeline.

Pandas-free (uses csv/json + numpy + matplotlib) to avoid NumPy/Pandas ABI issues.

Figures written to --outdir:
- latency_distribution.png
- gpu_util_vs_time.png
- fps_vs_resolution.png
- pipeline_occupancy.png
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_csv_as_columns(path: str) -> Dict[str, np.ndarray]:
    """Read a CSV into column arrays (float where possible)."""
    with open(path, 'r', newline='') as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    if not rows:
        return {}

    cols: Dict[str, List[float]] = {k: [] for k in rows[0].keys()}
    raw_cols: Dict[str, List[str]] = {k: [] for k in rows[0].keys()}

    for r in rows:
        for k, v in r.items():
            raw_cols[k].append(v)

    out: Dict[str, np.ndarray] = {}
    for k, vs in raw_cols.items():
        # try numeric conversion
        num: List[float] = []
        ok = True
        for v in vs:
            try:
                num.append(float(v))
            except Exception:
                ok = False
                break
        if ok:
            out[k] = np.asarray(num, dtype=float)
        else:
            out[k] = np.asarray(vs, dtype=object)

    # add relative time if present
    if 't_wall' in out and out['t_wall'].size:
        out['t_rel_s'] = out['t_wall'] - float(out['t_wall'][0])
    return out


def plot_latency_distribution(frames_cols: Dict[str, np.ndarray], outdir: Path):
    if not frames_cols or 'total_ms' not in frames_cols:
        return
    x = frames_cols['total_ms']
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    plt.figure()
    parts = plt.violinplot([x], showmeans=True, showmedians=True)
    plt.boxplot([x], positions=[1], widths=0.15)
    plt.xticks([1], ['total_ms'])
    plt.ylabel('Latency (ms/frame)')
    plt.title('Per-Frame Latency Distribution')
    plt.savefig(outdir / 'latency_distribution.png', dpi=200, bbox_inches='tight')


def plot_gpu_util_vs_time(log_cols: Dict[str, np.ndarray], outdir: Path):
    if not log_cols or 't_rel_s' not in log_cols or 'gpu_util' not in log_cols:
        return
    t = log_cols['t_rel_s']
    util = log_cols.get('gpu_util')
    mem = log_cols.get('gpu_mem_used_mb')

    plt.figure()
    plt.plot(t, util, label='GPU util (%)')
    if mem is not None:
        plt.plot(t, mem, label='GPU mem used (MB)')
    plt.xlabel('Time (s)')
    plt.title('GPU Utilization vs Time')
    plt.legend()
    plt.savefig(outdir / 'gpu_util_vs_time.png', dpi=200, bbox_inches='tight')


def plot_fps_vs_resolution(log_cols: Dict[str, np.ndarray], outdir: Path):
    if not log_cols or 'fps_sliding' not in log_cols or 'res_w' not in log_cols or 'res_h' not in log_cols:
        return

    fps = log_cols['fps_sliding']
    w = log_cols['res_w'].astype(int)
    h = log_cols['res_h'].astype(int)
    # group by (w,h)
    pairs = list(zip(w, h))
    uniq = sorted(set(pairs))
    means = []
    labels = []
    for p in uniq:
        m = np.array([pp == p for pp in pairs], dtype=bool)
        if m.any():
            means.append(float(np.nanmean(fps[m])))
            labels.append(f'{p[0]}x{p[1]}')

    if not means:
        return

    plt.figure()
    x = np.arange(len(means))
    plt.bar(x, means)
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel('Mean sliding FPS')
    plt.title('FPS vs Input Resolution')
    plt.savefig(outdir / 'fps_vs_resolution.png', dpi=200, bbox_inches='tight')


def read_timeline(jsonl_path: str) -> List[dict]:
    spans = []
    with open(jsonl_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                spans.append(json.loads(line))
            except Exception:
                continue
    return spans


def plot_pipeline_occupancy(timeline_path: str, outdir: Path, max_batches: int = 30):
    p = Path(timeline_path)
    if not p.exists():
        return
    spans = read_timeline(str(p))
    if not spans:
        return

    # filter first N batches
    spans = [s for s in spans if int(s.get('batch_idx', 0)) < max_batches]

    # stage ordering
    stage_order = ['roi', 'mog2', 'preprocess', 'inference', 'postprocess', 'postprocess_frame', 'batch_total']
    stages = sorted(set(s.get('stage', '') for s in spans), key=lambda x: stage_order.index(x) if x in stage_order else 999)

    y_positions = {st: i for i, st in enumerate(stages)}

    plt.figure(figsize=(10, max(3, 0.45 * len(stages))))
    for st in stages:
        segs = []
        for s in spans:
            if s.get('stage') != st:
                continue
            t0 = float(s.get('t0', 0.0))
            t1 = float(s.get('t1', 0.0))
            if t1 <= t0:
                continue
            segs.append((t0, t1 - t0))
        if segs:
            y = y_positions[st]
            plt.broken_barh(segs, (y - 0.4, 0.8))

    plt.yticks(list(y_positions.values()), list(y_positions.keys()))
    plt.xlabel('Time since start (s)')
    plt.title('CPU–GPU Pipeline Occupancy (Timeline)')
    plt.tight_layout()
    plt.savefig(outdir / 'pipeline_occupancy.png', dpi=200, bbox_inches='tight')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='assets/perf_logs/perf_log.csv', help='Batch-level perf log CSV.')
    ap.add_argument('--frames', default='assets/perf_logs/perf_frames.csv', help='Per-frame perf log CSV.')
    ap.add_argument('--timeline', default='assets/perf_logs/perf_timeline.jsonl', help='Timeline JSONL for occupancy.')
    ap.add_argument('--outdir', default='assets/perf_plots', help='Output directory.')
    ap.add_argument('--max_batches', type=int, default=30, help='Max batches to show in occupancy diagram.')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_cols = read_csv_as_columns(args.log) if Path(args.log).exists() else {}
    frames_cols = read_csv_as_columns(args.frames) if Path(args.frames).exists() else {}

    plot_latency_distribution(frames_cols, outdir)
    plot_gpu_util_vs_time(log_cols, outdir)
    plot_fps_vs_resolution(log_cols, outdir)
    plot_pipeline_occupancy(args.timeline, outdir, max_batches=args.max_batches)

    print(f'Wrote plots to: {outdir.resolve()}')


if __name__ == '__main__':
    main()

