import argparse 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from pathlib import Path

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # make time relative (seconds) for nicer plots
    t0 = df["t_wall"].iloc[0]
    df["t_rel_s"] = df["t_wall"] - t0
    return df

def fps_vs_motion_density(dfs, labels, outdir: Path):
    plt.figure()
    for df, lab in zip(dfs, labels):
        x = df["motion_density"].to_numpy()
        y = df["fps_sliding"].to_numpy()
        plt.scatter(x, y, s=10, alpha=0.35, label=lab)

        # binned mean trend
        bins = np.linspace(0, 1, 11)
        inds = np.digitize(x, bins) - 1
        xb, yb = [], []
        for bi in range(len(bins)-1):
            m = inds == bi
            if m.any():
                xb.append((bins[bi] + bins[bi+1]) / 2)
                yb.append(np.mean(y[m]))
        if xb:
            plt.plot(xb, yb, linewidth=2)

    plt.xlabel("Motion density (fraction of frames in batch with motion)")
    plt.ylabel("Effective FPS (sliding)")
    plt.title("FPS vs Motion Density")
    plt.legend()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "fps_vs_motion_density.png", dpi=200, bbox_inches="tight")


def inference_calls_per_sec(dfs, labels, outdir: Path):
    plt.figure()
    for df, lab in zip(dfs, labels):
        # build 1-second bins
        t = df["t_rel_s"].to_numpy()
        infer = df["inference_ran"].to_numpy()  # 1 per batch if inference ran
        max_t = float(np.max(t)) if len(t) else 0.0
        edges = np.arange(0, max_t + 1.0, 1.0)
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
        counts, _ = np.histogram(t[infer > 0], bins=edges)
        centers = edges[:-1] + 0.5
        plt.plot(centers, counts, label=lab)

    plt.xlabel("Time (s)")
    plt.ylabel("Inference calls per second")
    plt.title("Inference Calls / Second")
    plt.legend()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "inference_calls_per_sec.png", dpi=200, bbox_inches="tight")



def latency_breakdown(dfs, labels, outdir: Path):
    # Stacked bar: mean ms per frame per stage
    stages = [
        ("roi_ms_per_frame", "ROI"),
        ("mog2_ms_per_frame", "MOG2"),
        ("preprocess_ms_per_frame", "Preprocess"),
        ("inference_ms_per_frame", "Inference"),
        ("postprocess_ms_per_frame", "Postprocess"),
    ]
    means = []
    for df in dfs:
        vals = [float(df[c].mean()) for c, _ in stages]
        known = sum(vals)
        total = float(df["total_ms_per_frame"].mean())
        other = max(total - known, 0.0)
        means.append(vals + [other])

    labels_stage = [n for _, n in stages] + ["Other"]

    plt.figure()
    x = np.arange(len(dfs))
    bottom = np.zeros(len(dfs))
    for si, stage_name in enumerate(labels_stage):
        height = np.array([m[si] for m in means])
        plt.bar(x, height, bottom=bottom, label=stage_name)
        bottom += height

    plt.xticks(x, labels)
    plt.ylabel("Mean latency (ms/frame)")
    plt.title("Latency Breakdown (Stacked)")
    plt.legend()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "latency_breakdown.png", dpi=200, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="Path to perf_log.csv (can be provided multiple times).")
    ap.add_argument("--label", action="append", help="Label(s) for each CSV (same count as --csv).")
    ap.add_argument("--outdir", default="assets/perf_plots", help="Output directory for PNGs.")
    args = ap.parse_args()

    labels = args.label or [Path(p).stem for p in args.csv]
    if len(labels) != len(args.csv):
        raise SystemExit("Provide the same number of --label entries as --csv, or omit --label entirely.")

    dfs = [load_csv(p) for p in args.csv]
    outdir = Path(args.outdir)

    fps_vs_motion_density(dfs, labels, outdir)
    inference_calls_per_sec(dfs, labels, outdir)
    latency_breakdown(dfs, labels, outdir)

    print(f"Wrote plots to: {outdir.resolve()}")

if __name__ == "__main__":
    main()




















