import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

CSV = Path(__file__).parent.parent / "assets" / "experiment_results" / "ablation_summary_unified.csv"
OUT = Path(__file__).parent

# ── configurations to display ─────────────────────────────────────────────────
CONFIGS = {
    "full_pipeline":        "Full Pipeline",
    "no_saving":            "No Saving",
    "no_roi":               "No ROI",
    "no_lane_segmentation": "No Lane/Scene-Mask",
    "no_mog2_gating":       "No MOG2",
    "tiling_enabled":       "Tiling Enabled",
}

VIDEO_LABELS = {
    "samples/test_samples/MVI_39401.mp4":  "MVI_39401",
    "samples/test_samples/fisheye.mp4":    "Fisheye",
    "samples/test_samples/highway.mp4":    "Highway",
    "samples/jetson_2_recording.mp4":      "Jetson 2",
    "samples/jetson_3_recording.mp4":      "Jetson 3",
}

VIDEO_ORDER = [
    "samples/test_samples/MVI_39401.mp4",
    "samples/test_samples/fisheye.mp4",
    "samples/test_samples/highway.mp4",
    "samples/jetson_2_recording.mp4",
    "samples/jetson_3_recording.mp4",
]

# ── load data ─────────────────────────────────────────────────────────────────
rows = []
with open(CSV, newline="") as f:
    for row in csv.DictReader(f):
        key    = row["key"]
        status = row["status"]
        video  = row["video_source"]
        fps    = row["fps"]
        total_ms = row["total_ms"]
        p95    = row["p95_latency_ms"]
        if status != "completed":
            continue
        if key not in CONFIGS:
            continue
        if video not in VIDEO_LABELS:
            continue
        try:
            fps_v   = float(fps)
            total_v = float(total_ms)
            p95_v   = float(p95)
        except (TypeError, ValueError):
            continue
        rows.append({"key": key, "video": video, "fps": fps_v,
                     "total_ms": total_v, "p95_ms": p95_v})

# index: data[video][config_key] = {fps, total_ms, p95_ms}
data = {v: {} for v in VIDEO_ORDER}
for r in rows:
    data[r["video"]][r["key"]] = r

# ── colour palette ────────────────────────────────────────────────────────────
PALETTE = ["#1565C0", "#2E7D32", "#F57F17", "#E65100", "#6A1B9A", "#B71C1C"]
CONFIG_KEYS = list(CONFIGS.keys())
COLOR = {k: PALETTE[i] for i, k in enumerate(CONFIG_KEYS)}

# ── helpers ───────────────────────────────────────────────────────────────────
STYLE = dict(figure_facecolor="white", axes_facecolor="white",
             text_color="#1a1a1a", grid_color="#cccccc", spine_color="#999999")

def apply_dark(ax):
    ax.set_facecolor(STYLE["axes_facecolor"])
    ax.tick_params(colors=STYLE["text_color"])
    ax.xaxis.label.set_color(STYLE["text_color"])
    ax.yaxis.label.set_color(STYLE["text_color"])
    ax.title.set_color(STYLE["text_color"])
    for sp in ax.spines.values():
        sp.set_edgecolor(STYLE["spine_color"])
    ax.yaxis.grid(True, color=STYLE["grid_color"], linewidth=0.6, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)

# =============================================================================
# Chart 1 – Grouped FPS bar chart
# =============================================================================
n_videos  = len(VIDEO_ORDER)
n_configs = len(CONFIGS)
bar_w     = 0.14
group_gap = 0.2
x_centers = np.arange(n_videos) * (n_configs * bar_w + group_gap)

fig, ax = plt.subplots(figsize=(16, 7), facecolor=STYLE["figure_facecolor"])
apply_dark(ax)

for ci, (ckey, clabel) in enumerate(CONFIGS.items()):
    offsets = x_centers + (ci - n_configs / 2 + 0.5) * bar_w
    vals = []
    for vkey in VIDEO_ORDER:
        vals.append(data[vkey].get(ckey, {}).get("fps", 0))
    bars = ax.bar(offsets, vals, bar_w, color=COLOR[ckey],
                  label=clabel, edgecolor="white", linewidth=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.5, color=STYLE["text_color"], rotation=90)

ax.set_xticks(x_centers)
ax.set_xticklabels([VIDEO_LABELS[v] for v in VIDEO_ORDER], fontsize=11)
ax.set_ylabel("Throughput (FPS)", fontsize=12)
ax.set_title("Ablation Study – Throughput per Configuration & Video\n(Jetson VPS · YOLOv8s TensorRT)", fontsize=13, pad=14)
ax.legend(handles=[mpatches.Patch(color=COLOR[k], label=CONFIGS[k]) for k in CONFIG_KEYS],
          facecolor="white", edgecolor="#999999", labelcolor=STYLE["text_color"],
          fontsize=10, loc="upper right")
ax.set_xlim(x_centers[0] - n_configs * bar_w, x_centers[-1] + n_configs * bar_w)

fig.tight_layout()
out1 = OUT / "ablation_fps_grouped.png"
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=STYLE["figure_facecolor"])
plt.close(fig)
print(f"Saved: {out1}")

# =============================================================================
# Chart 2 – Total latency + P95 latency per configuration, grouped by video
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=STYLE["figure_facecolor"])

for ax_idx, (metric, ylabel, title_suffix) in enumerate([
    ("total_ms",  "Latency (ms)", "Avg Frame Latency"),
    ("p95_ms",    "Latency (ms)", "P95 Frame Latency"),
]):
    ax = axes[ax_idx]
    apply_dark(ax)

    for ci, (ckey, clabel) in enumerate(CONFIGS.items()):
        offsets = x_centers + (ci - n_configs / 2 + 0.5) * bar_w
        vals = []
        for vkey in VIDEO_ORDER:
            vals.append(data[vkey].get(ckey, {}).get(metric, 0))
        bars = ax.bar(offsets, vals, bar_w, color=COLOR[ckey],
                      label=clabel, edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom",
                        fontsize=6.0, color=STYLE["text_color"], rotation=90)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([VIDEO_LABELS[v] for v in VIDEO_ORDER], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Latency Breakdown – {title_suffix}\n(Jetson VPS · YOLOv8s TensorRT)",
                 fontsize=12, pad=12)
    ax.set_xlim(x_centers[0] - n_configs * bar_w, x_centers[-1] + n_configs * bar_w)

    if ax_idx == 1:
        ax.legend(handles=[mpatches.Patch(color=COLOR[k], label=CONFIGS[k]) for k in CONFIG_KEYS],
                  facecolor="white", edgecolor="#999999", labelcolor=STYLE["text_color"],
                  fontsize=9, loc="upper right")

# add a shared note about P95 real-time stability
fig.text(0.5, 0.01,
         "P95 latency captures tail-frame cost — a low P95 indicates stable real-time processing with few spikes.",
         ha="center", fontsize=9, color="#555555")

fig.tight_layout(rect=[0, 0.04, 1, 1])
out2 = OUT / "ablation_latency_total_p95.png"
fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor=STYLE["figure_facecolor"])
plt.close(fig)
print(f"Saved: {out2}")

# =============================================================================
# Chart 3 – % FPS change from baseline per configuration
# =============================================================================

ABLATION_VIDEOS = list(VIDEO_ORDER)
ABLATION_VIDEO_LABELS = VIDEO_LABELS

# Reload raw fps per (video, key) from the CSV
fps_lookup = {}
with open(CSV, newline="") as f:
    for row in csv.DictReader(f):
        if row["status"] != "completed":
            continue
        try:
            fps_lookup[(row["video_source"], row["key"])] = float(row["fps"])
        except (TypeError, ValueError):
            pass

# Compute % change vs full_pipeline per video, then average + collect per-video points
DELTA_CONFIGS = [
    ("no_saving",           "No Saving"),
    ("no_roi",              "No ROI"),
    ("no_lane_segmentation","No Lane /\nScene-Mask"),
    ("no_mog2_gating",      "No MOG2"),
    ("tiling_enabled",      "Tiling\nEnabled"),
]

per_video_deltas = {k: [] for k, _ in DELTA_CONFIGS}
for vkey in ABLATION_VIDEOS:
    base = fps_lookup.get((vkey, "full_pipeline"))
    if base is None or base == 0:
        continue
    for ckey, _ in DELTA_CONFIGS:
        val = fps_lookup.get((vkey, ckey))
        if val is not None:
            per_video_deltas[ckey].append((val - base) / base * 100)

all_entries = [(k, lbl, per_video_deltas[k]) for k, lbl in DELTA_CONFIGS]

# Sort descending so the chart flows from gain → loss
all_entries.sort(key=lambda e: (sum(e[2]) / len(e[2])) if e[2] else 0, reverse=True)

fig, ax = plt.subplots(figsize=(11, 6), facecolor=STYLE["figure_facecolor"])
apply_dark(ax)

bar_positions = np.arange(len(all_entries))
bar_width = 0.55

for i, (ckey, clabel, deltas) in enumerate(all_entries):
    avg = sum(deltas) / len(deltas) if deltas else 0
    bar_color = "#2E7D32" if avg >= 0 else "#B71C1C"
    bar_edge  = "#1B5E20" if avg >= 0 else "#7F0000"
    ax.bar(i, avg, bar_width, color=bar_color, edgecolor=bar_edge,
           linewidth=0.8, zorder=3)

    # value label inside/above the bar
    label_y = avg + (1.5 if avg >= 0 else -3.5)
    va = "bottom" if avg >= 0 else "top"
    ax.text(i, avg + (0.8 if avg >= 0 else -0.8), f"{avg:+.1f}%",
            ha="center", va=va, fontsize=10, fontweight="bold",
            color="#1a1a1a" if avg >= 0 else "#1a1a1a")

    # scatter per-video dots (skip panorama which has only one point)
    if len(deltas) > 1:
        jitter = np.linspace(-0.12, 0.12, len(deltas))
        for j, (dv, jx) in enumerate(zip(deltas, jitter)):
            dot_color = "#66BB6A" if dv >= 0 else "#EF5350"
            ax.scatter(i + jx, dv, color=dot_color, s=55, zorder=5,
                       edgecolors="#333333", linewidths=0.5)

ax.axhline(0, color="#555555", linewidth=1.2, zorder=2)
ax.set_xticks(bar_positions)
ax.set_xticklabels([e[1] for e in all_entries], fontsize=11)
ax.set_ylabel("FPS change vs. Full Pipeline (%)", fontsize=12)
ax.set_title("Ablation Study – % FPS Change from Baseline\n(Jetson VPS · YOLOv8s TensorRT · avg across videos)",
             fontsize=13, pad=14)

# legend for the dots
dot_legend = [
    plt.scatter([], [], color="#66BB6A", edgecolors="#333333", linewidths=0.5, s=55, label="Per-video result"),
]
ax.legend(handles=dot_legend, facecolor="white", edgecolor="#999999",
          labelcolor=STYLE["text_color"], fontsize=9, loc="lower left")

# light shading to separate gain/loss zones
ymin, ymax = ax.get_ylim()
ax.axhspan(0, ymax, alpha=0.04, color="#2E7D32", zorder=1)
ax.axhspan(ymin, 0,  alpha=0.04, color="#B71C1C", zorder=1)
ax.set_ylim(ymin - 2, ymax + 4)

fig.tight_layout()
out3 = OUT / "ablation_fps_pct_change.png"
fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor=STYLE["figure_facecolor"])
plt.close(fig)
print(f"Saved: {out3}")
