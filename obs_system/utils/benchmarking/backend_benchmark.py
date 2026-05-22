from __future__ import annotations

import csv
import json
import re
import subprocess
import threading
import time

import numpy as np

from pathlib import Path
from typing import Any

from obs_system.utils.logger import get_logger


logger = get_logger("obs_system." + __name__)


class JetsonSampler:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = max(0.2, float(interval_s))
        self.samples: list[dict[str, float]] = []
        self.static_context: dict[str, Any] = collect_jetson_context()
        self.available = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        try:
            from jtop import jtop
        except Exception:
            return

        try:
            with jtop() as jetson:
                self.available = True
                while jetson.ok() and not self._stop.is_set():
                    stats = dict(getattr(jetson, "stats", {}) or {})
                    self.static_context.update(_extract_jetson_context(stats))
                    sample = _extract_jetson_sample(stats)
                    if sample:
                        self.samples.append(sample)
                    time.sleep(self.interval_s)
        except Exception:
            logger.debug("Jetson sampling unavailable", exc_info=True)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _pick_value(stats: dict[str, Any], *needles: str) -> float | None:
    lowered = [needle.lower() for needle in needles]
    matches: list[float] = []
    for key, value in stats.items():
        key_lower = str(key).lower()
        if any(needle in key_lower for needle in lowered):
            parsed = _to_float(value)
            if parsed is not None:
                matches.append(parsed)
    if matches:
        return float(np.mean(matches))
    return None


def _pick_text_value(stats: dict[str, Any], *needles: str) -> str | None:
    lowered = [needle.lower() for needle in needles]
    for key, value in stats.items():
        key_lower = str(key).lower()
        if any(needle in key_lower for needle in lowered):
            text = str(value).strip()
            if text:
                return text
    return None


def _run_command_output(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.5,
        ).strip()
    except Exception:
        return None
    return out or None


def _read_sysfs_number(paths: list[str], scale: float = 1.0) -> float | None:
    for raw_path in paths:
        try:
            # Skip path.exists() – on Python <3.12 it raises PermissionError for
            # /sys/kernel/debug/* files that require root to stat().  Attempting
            # the read directly is cheaper and handles all error cases uniformly.
            value = float(Path(raw_path).read_text(encoding="utf-8").strip())
            return value / scale if scale else value
        except Exception:
            continue
    return None


def _parse_power_mode(output: str | None) -> str | None:
    if not output:
        return None
    for line in output.splitlines():
        lower = line.lower()
        if "power mode" in lower or "nvp model" in lower or "nvpmodel" in lower:
            return line.split(":", 1)[-1].strip() or line.strip()
    return output.splitlines()[0].strip() if output.splitlines() else None


def collect_jetson_context() -> dict[str, Any]:
    context: dict[str, Any] = {}

    power_mode = _parse_power_mode(_run_command_output(["nvpmodel", "-q"]))
    if power_mode:
        context["power_mode"] = power_mode

    emc_frequency_mhz = _read_sysfs_number(
        paths=[
            "/sys/kernel/debug/bpmp/debug/clk/emc/rate",
            "/sys/kernel/debug/clk/emc/rate",
        ],
        scale=1_000_000.0,
    )
    if emc_frequency_mhz is not None:
        context["emc_frequency_mhz"] = emc_frequency_mhz

    return context


def _extract_jetson_context(stats: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}

    power_mode = _pick_text_value(stats, "power mode", "nvp model", "nvpmodel")
    if power_mode:
        context["power_mode"] = power_mode

    return context


def _extract_jetson_sample(stats: dict[str, Any]) -> dict[str, float]:
    sample: dict[str, float] = {}

    cpu_util = _pick_value(stats, "cpu")
    gpu_util = _pick_value(stats, "gpu")
    ram_used = _pick_value(stats, "ram")
    temp_cpu = _pick_value(stats, "temp cpu", "cpu temp")
    temp_gpu = _pick_value(stats, "temp gpu", "gpu temp")
    temp_board = _pick_value(stats, "temp board", "board temp", "thermal")
    power_w = _pick_value(stats, "power tot", "power", "vdd")
    emc_freq_mhz = _pick_value(stats, "emc freq", "emc mhz", "emc rate", "emc clock")
    emc_util = _pick_value(stats, "emc")

    if cpu_util is not None:
        sample["jetson_cpu_util"] = cpu_util
    if gpu_util is not None:
        sample["jetson_gpu_util"] = gpu_util
    if ram_used is not None:
        sample["jetson_ram_metric"] = ram_used
    if temp_cpu is not None:
        sample["jetson_cpu_temp_c"] = temp_cpu
    if temp_gpu is not None:
        sample["jetson_gpu_temp_c"] = temp_gpu
    if temp_board is not None:
        sample["jetson_board_temp_c"] = temp_board
    if power_w is not None:
        sample["jetson_power_w"] = power_w
    if emc_freq_mhz is not None:
        sample["jetson_emc_frequency_mhz"] = emc_freq_mhz
    if emc_util is not None:
        sample["jetson_emc_metric"] = emc_util

    return sample


def count_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def read_cbor_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        import cbor2
    except ImportError:
        logger.debug("cbor2 not installed; MQTT archive at %s will be skipped", path)
        return []

    messages: list[dict[str, Any]] = []
    for raw in path.read_bytes().splitlines():
        if not raw.strip():
            continue
        try:
            payload = cbor2.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            messages.append(payload)
    return messages


def probe_model_backend(model_path: str) -> tuple[bool, str]:
    """
    Check whether *model_path* can be run in the current environment.

    Returns ``(True, "ok")`` when the model is runnable, or
    ``(False, <human-readable reason>)`` when it should be skipped.

    Reasons for skipping:
    - File does not exist
    - ``.engine`` backend requires TensorRT, which is not installed
      (common on WSL / non-Jetson hosts)
    """
    path = Path(model_path)
    if not path.exists():
        return False, f"file not found: {path}"

    if path.suffix.lower() == ".engine":
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            return False, (
                "TensorRT is not available in this environment "
                "(install tensorrt or run on a Jetson / TRT-capable host)"
            )

    return True, "ok"


def latency_summary(frame_rows: list[dict[str, str]]) -> dict[str, float]:
    latencies = [float(row["total_ms"]) for row in frame_rows if row.get("total_ms")]
    if not latencies:
        return {
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    xs = np.asarray(latencies, dtype=np.float32)
    return {
        "avg_latency_ms": float(xs.mean()),
        "p50_latency_ms": float(np.percentile(xs, 50)),
        "p95_latency_ms": float(np.percentile(xs, 95)),
    }


def mean_metric(rows: list[dict[str, str]], key: str) -> float | None:
    vals = []
    for row in rows:
        value = row.get(key)
        if value in (None, "", "nan"):
            continue
        try:
            vals.append(float(value))
        except ValueError:
            continue
    if not vals:
        return None
    return float(np.mean(vals))


def metric_distribution(rows: list[dict[str, str]], key: str) -> dict[str, float] | None:
    vals = []
    for row in rows:
        value = row.get(key)
        if value in (None, "", "nan"):
            continue
        try:
            vals.append(float(value))
        except ValueError:
            continue
    if not vals:
        return None

    xs = np.asarray(vals, dtype=np.float32)
    return {
        "mean_ms": float(xs.mean()),
        "p50_ms": float(np.percentile(xs, 50)),
        "p95_ms": float(np.percentile(xs, 95)),
    }


def summarize_stage_latency(rows: list[dict[str, str]], stage_keys: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in stage_keys:
        dist = metric_distribution(rows, key)
        if dist is not None:
            summary[key] = dist
    return summary


def summarize_jetson(samples: list[dict[str, float]], static_context: dict[str, Any] | None = None) -> dict[str, Any]:
    if not samples and not static_context:
        return {}

    out: dict[str, Any] = {}
    keys = sorted({key for sample in samples for key in sample})
    for key in keys:
        vals = [sample[key] for sample in samples if key in sample]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))

    if static_context:
        if static_context.get("power_mode"):
            out["power_mode"] = static_context["power_mode"]
        if static_context.get("emc_frequency_mhz") is not None and "jetson_emc_frequency_mhz_mean" not in out:
            out["jetson_emc_frequency_mhz_mean"] = float(static_context["emc_frequency_mhz"])

    return out


def build_hardware_summary(
    *,
    batch_rows: list[dict[str, str]] | None = None,
    jetson_samples: list[dict[str, float]] | None = None,
    jetson_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = batch_rows or []
    hardware = {
        "cpu_util_mean": mean_metric(rows, "cpu_util"),
        "gpu_util_mean": mean_metric(rows, "gpu_util"),
        "gpu_mem_used_mb_mean": mean_metric(rows, "gpu_mem_used_mb"),
        "gpu_mem_total_mb_mean": mean_metric(rows, "gpu_mem_total_mb"),
    }

    jetson_summary = summarize_jetson(jetson_samples or [], jetson_context)
    hardware.update(jetson_summary)

    if hardware["cpu_util_mean"] is None:
        hardware["cpu_util_mean"] = jetson_summary.get("jetson_cpu_util_mean")
    if hardware["gpu_util_mean"] is None:
        hardware["gpu_util_mean"] = jetson_summary.get("jetson_gpu_util_mean")

    hardware["temperature_c_mean"] = _first_present(
        jetson_summary.get("jetson_gpu_temp_c_mean"),
        jetson_summary.get("jetson_cpu_temp_c_mean"),
        jetson_summary.get("jetson_board_temp_c_mean"),
    )
    hardware["power_w_mean"] = jetson_summary.get("jetson_power_w_mean")
    hardware["emc_frequency_mhz_mean"] = jetson_summary.get("jetson_emc_frequency_mhz_mean")
    hardware["power_mode"] = jetson_summary.get("power_mode")

    return hardware


def fmt_opt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        return value
    return None
