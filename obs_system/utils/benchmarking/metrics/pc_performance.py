from obs_system.utils.benchmarking.interface.benchmark import BenchMark 

import numpy as np 
from typing import Dict, Tuple, Iterable, List, Any 


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
