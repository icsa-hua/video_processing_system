from obs_system.utils.benchmarking.interface.benchmark import BenchMark 

import csv
import json
import subprocess
import collections
import numpy as np 

from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any 


class ComputationalPerf(BenchMark): 
    """PlaceHolder""" 
    def __init__(self) -> None:
        self._stages: Dict[str, List[float]] = {}
        self._results: Dict[str, float] = {}

    def tick(self, stage: str, ms: float) -> None:
        self._stages.setdefault(stage, []).append(float(ms))

    def update(self, *args, **kwargs) -> None:
        # Could accept a dict of stage->ms per frame
        data: Dict[str, float] = kwargs.get("timings", {})
        for k, v in data.items():
            self.tick(k, v)

    def finalize(self) -> None:
        def pct(a: np.ndarray, q: float) -> float:
            return float(np.percentile(a, q)) if a.size else 0.0
        out = {}
        for stage, arr in self._stages.items():
            xs = np.array(arr, dtype=np.float32)
            out[f"{stage}_p50_ms"] = pct(xs, 50)
            out[f"{stage}_p95_ms"] = pct(xs, 95)
            out[f"{stage}_mean_ms"] = float(xs.mean()) if xs.size else 0.0
        self._results = out

    def results(self) -> Dict[str, float]:
        return dict(self._results)

    def reset(self) -> None:
        self._stages.clear()
        self._results = {}


# ================= Performance logging helpers =================
class PerfLogger:
    """Lightweight CSV logger for pipeline performance analysis."""
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.csv_path, 'w', newline='')
        self._writer = csv.DictWriter(self._fp, fieldnames=[
            't_wall', 'batch_idx', 'frames_in_batch',
            'res_w', 'res_h',
            'motion_density', 'avg_motion_score',
            'inference_ran', 'frames_inferred',
            'infer_calls_per_sec',
            'gpu_util', 'gpu_mem_used_mb', 'gpu_mem_total_mb',
            'cpu_util',
            'roi_ms_per_frame', 'mog2_ms_per_frame',
            'preprocess_ms_per_frame', 'inference_ms_per_frame', 'postprocess_ms_per_frame',
            'total_ms_per_frame', 'fps_sliding'

        ])
        self._writer.writeheader()

    def log(self, row: dict):
        self._writer.writerow(row)
        self._fp.flush()

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass

class SlidingCounter:
    """Counts events in a sliding time window (seconds)."""
    def __init__(self, window_s: float = 1.0):
        self.window_s = float(window_s)
        self.events = collections.deque()  # (t, value)

    def add(self, t: float, value: float = 1.0):
        self.events.append((t, float(value)))
        self._trim(t)

    def _trim(self, now: float):
        cutoff = now - self.window_s
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def sum(self, now: float) -> float:
        self._trim(now)
        return sum(v for _, v in self.events)

    def rate(self, now: float) -> float:
        # events per second over the window
        return self.sum(now) / self.window_s if self.window_s > 0 else 0.0

# ===============================================================

class FramePerfLogger:
    """Per-frame CSV logger for latency distribution plots."""
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.csv_path, 'w', newline='')
        self._writer = csv.DictWriter(self._fp, fieldnames=[
            't_wall', 'batch_idx', 'frame_id',
            'res_w', 'res_h',
            'motion_passed', 'motion_score',
            'gpu_util', 'gpu_mem_used_mb', 'cpu_util',
            'roi_ms', 'mog2_ms', 'preprocess_ms', 'inference_ms', 'postprocess_ms',
            'total_ms',
        ])
        self._writer.writeheader()

    def log(self, row: dict):
        self._writer.writerow(row)
        self._fp.flush()

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass



class GPUMonitor:
    """Best-effort GPU utilization monitor.

    Tries NVML (pynvml) first, falls back to nvidia-smi.
    Returns NaNs if GPU stats are unavailable.
    """
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = int(gpu_index)
        self._mode = None
        self._handle = None
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._mode = 'nvml'
        except Exception:
            self._pynvml = None
            self._mode = 'nvidia-smi'

    def sample(self) -> dict:
        """Return dict with keys: gpu_util, mem_used_mb, mem_total_mb."""
        if self._mode == 'nvml' and self._pynvml is not None and self._handle is not None:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return {
                    'gpu_util': float(util.gpu),
                    'mem_used_mb': float(mem.used) / (1024.0 * 1024.0),
                    'mem_total_mb': float(mem.total) / (1024.0 * 1024.0),
                }
            except Exception:
                pass

        # Fallback to nvidia-smi
        try:
            cmd = [
                'nvidia-smi',
                f'--id={self.gpu_index}',
                '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
            # Example: "12, 1024, 8192"
            parts = [p.strip() for p in out.split(',')]
            if len(parts) >= 3:
                return {
                    'gpu_util': float(parts[0]),
                    'mem_used_mb': float(parts[1]),
                    'mem_total_mb': float(parts[2]),
                }
        except Exception:
            pass

        return {'gpu_util': float('nan'), 'mem_used_mb': float('nan'), 'mem_total_mb': float('nan')}


class CPUMonitor:
    """Best-effort CPU utilization monitor."""
    def __init__(self):
        try:
            import psutil  # type: ignore
            self._psutil = psutil
            # Prime cpu_percent
            self._psutil.cpu_percent(interval=None)
        except Exception:
            self._psutil = None

    def sample(self) -> dict:
        if self._psutil is None:
            return {'cpu_util': float('nan')}
        try:
            return {'cpu_util': float(self._psutil.cpu_percent(interval=None))}
        except Exception:
            return {'cpu_util': float('nan')}


class TimelineLogger:
    """JSONL logger for CPU/GPU pipeline occupancy (Gantt)."""
    def __init__(self, jsonl_path: str):
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.jsonl_path, 'w', encoding='utf-8')

    def log_span(self, batch_idx: int, stage: str, t0: float, t1: float, extra: Optional[dict] = None):
        row = {
            'batch_idx': int(batch_idx),
            'stage': str(stage),
            't0': float(t0),
            't1': float(t1),
        }
        if extra:
            row.update(extra)
        self._fp.write(json.dumps(row) + "\n")
        self._fp.flush()

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass

# ===============================================================

