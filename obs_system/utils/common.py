from __future__ import annotations 

import re
import subprocess
import glob
import cv2
import sys
import os
import socket
import numpy as np
import torch 

from typing import List, Dict, Iterable, Optional, Union, Literal 
from pathlib import Path
from torchvision.ops import batched_nms, nms
from typing import Dict, Any, Optional
from dataclasses import dataclass 
from ultralytics.engine.model import Results

ModelKind = Literal['pt', 'onnx', 'engine']

@dataclass(frozen=True) 
class ModelSpecification: 
    kind:ModelKind 
    name:str 
    path:Optional[Path]
    resolved_from:Literal["path", "alias", "auto"]


# check the existence of GPU 
def check_nvidia_existence(): 

    try: 
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return result.returncode == 0 
    
    except ValueError:
        return False
    

def find_available_port(start_port=8000, max_attempts=10):
    for port in range(start_port, start_port + max_attempts): 
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
            s.settimeout(1) 
            host = socket.gethostbyname("localhost")
            if s.connect_ex((host, port)) != 0: 
                return port 
            
    return None 


def check_model_name(model:Union[str,Path], *, model_dirs:Optional[Iterable[Union[str,Path]]]=None, aliases:Optional[Dict[str,str]]=None, prefer_ext:tuple[str, ...]=('.pt', '.onnx', '.engine'), must_exist:bool=False): 
        
    default_aliases = {
        'pt':"yolov8s.pt", 
        'yolo':"yolov8s.pt", 
        'YOLO':"yolov8s.pt",
        'v8':"yolov8s.pt",
        'yolov8':"yolov8s.pt",
        'onnx':"yolov8s.onnx", 
        'trt': "yolov8s.onnx", 
        'compressed': "yolov8s.onnx",
        'TRT':"yolov8s.onnx", 
        'ONNX':"yolov8s.onnx", 
        'engine': "yolov8s.onnx"
    }
        
    alias_map: Dict[str, str] = {**default_aliases, **(aliases or {})}
    raw = str(model).strip()
    p = Path(raw)
    is_path_like = bool(p.suffix) or ("/" in raw) or ("\\" in raw)
        
    def kind_from_suffix(sfx: str) -> ModelKind:
        sfx = sfx.lower()
        if sfx == ".pt":
            return "pt"
        if sfx == ".onnx":
            return "onnx"
        if sfx == ".engine":
            return "engine"
        raise ValueError(f"Unsupported model extension: {sfx!r} (expected .pt, .onnx, .engine)")

    if is_path_like:
        # Direct path case
        path = p.expanduser()
        kind = kind_from_suffix(path.suffix)
        name = path.stem
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        return ModelSpecification(kind=kind, name=name, path=path, resolved_from="path")

    # Alias case
    alias = raw.lower()

    # Resolve alias chain (alias -> alias -> filename)
    seen = set()
    target = alias
    while target in alias_map:
        if target in seen:
            raise ValueError(f"Alias loop detected while resolving {alias!r}")
        seen.add(target)
        target = alias_map[target].strip()

    target_path = Path(target)

    model_dirs_list = [Path(d).expanduser() for d in (model_dirs or [Path.cwd()])]
    
    if target_path.suffix:
        # It might be a filename or a relative path; try resolving via model_dirs if not absolute
        if target_path.is_absolute():
            resolved = target_path
        else:
            # If target includes subdirs, join directly with each model_dir
            candidates = [d / target_path for d in model_dirs_list]
            resolved = next((c for c in candidates if c.exists()), candidates[0])

        kind = kind_from_suffix(resolved.suffix)
        name = resolved.stem
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Model alias {alias!r} resolved to missing file: {resolved}")
        return ModelSpecification(kind=kind, name=name, path=resolved, resolved_from="alias")

    candidates = []
    for d in model_dirs_list:
        for ext in prefer_ext:
            candidates.append(d / f"{target}{ext}")
            candidates.append(d / f"{alias}{ext}")

    resolved = next((c for c in candidates if c.exists()), None)
   
    if resolved is None:
        # Can't resolve to an existing file; return "best guess" with the first preferred ext
        guess = model_dirs_list[0] / f"{target}{prefer_ext[-1]}"  # default to .pt as last
        # Better: if your prefer_ext has .pt last, use .pt; if not, use first.
        guess = model_dirs_list[0] / f"{target}{prefer_ext[-1] if '.pt' in prefer_ext else prefer_ext[0]}"
        kind = kind_from_suffix(guess.suffix)
        if must_exist:
            raise FileNotFoundError(
                f"Could not resolve model alias {alias!r}. Tried:\n" +
                "\n".join(str(c) for c in candidates[:12]) +
                ("\n..." if len(candidates) > 12 else "")
            )
        return ModelSpecification(kind=kind, name=target, path=guess, resolved_from="auto")

    kind = kind_from_suffix(resolved.suffix)
    name = resolved.stem
    return ModelSpecification(kind=kind, name=name, path=resolved, resolved_from="auto")


def get_frame_ids(labels: List[str], fallback_start: Optional[int] = None) -> List[int]:
    """
    Extract frame ids from Ultralytics batch metadata.

    Video labels include a frame token, for example:
        "video 1/1 (frame 12/345) /tmp/video.mp4: "

    Live streams currently return empty labels, so callers can pass
    fallback_start to generate stable sequential ids instead of failing.
    """

    next_fallback = 0 if fallback_start is None else int(fallback_start)
    frame_ids = []

    for label in labels:
        label = str(label or "")
        frame_match = re.search(r"\(frame\s+(\d+)(?:/|\))", label)
        if frame_match:
            frame_ids.append(int(frame_match.group(1)))
            continue

        tokens = label.split()
        path_token = tokens[-1].rstrip(":") if tokens else ""
        stem = Path(path_token).stem
        if stem.isdigit():
            frame_ids.append(int(stem))
            continue

        frame_ids.append(next_fallback)
        next_fallback += 1

    return frame_ids


def attr_memory_report(obj, top=10):
    """
    Print a per-attribute memory report for `obj`.
    Handles: numpy/cupy arrays, torch tensors (CPU/GPU), and Python containers.
    """
    import sys
    import numpy as np
    try:
        import torch
    except Exception:
        torch = None
    try:
        import cupy as cp
    except Exception:
        cp = None

    def _container_items(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield k; yield v
        elif isinstance(x, (list, tuple, set, frozenset)):
            for v in x:
                yield v

    def _rough_size(x, seen):
        # rough recursive size for containers (avoid deep/slow scans)
        obj_id = id(x)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        if isinstance(x, np.ndarray):
            return x.nbytes
        if torch is not None and isinstance(x, torch.Tensor):
            return x.element_size() * x.numel()
        if cp is not None and isinstance(x, cp.ndarray):
            return x.nbytes

        s = sys.getsizeof(x)
        if isinstance(x, (dict, list, tuple, set, frozenset)):
            for it in _container_items(x):
                s += sys.getsizeof(it)
        return s

    rows = []
    for name, val in vars(obj).items():
        dtype = type(val).__name__
        shape = getattr(val, "shape", None)
        where = "CPU"
        if isinstance(val, np.ndarray):
            size = val.nbytes
        elif torch is not None and isinstance(val, torch.Tensor):
            size = val.element_size() * val.numel()
            where = str(val.device)
        elif cp is not None and isinstance(val, cp.ndarray):
            size = val.nbytes
            where = "CUDA"
        else:
            size = _rough_size(val, seen=set())
        rows.append((name, size, where, dtype, shape))

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"{'attr':24s} {'MB':>8s} {'where':>10s} {'type':>18s} shape")
    for name, size, where, dtype, shape in rows[:top]:
        print(f"{name:24s} {size/1e6:8.2f} {where:>10s} {dtype:>18s} {shape}")
    return rows


def _bytes_fmt(n: float) -> str:
    if n is None: 
        return 
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def gpu_vars_report(scope: Dict[str, Any], top: int = 20, deep: bool = False):
    try:
        import torch
    except Exception:
        torch = None
    try:
        import cupy as cp
    except Exception:
        cp = None

    seen = set()
    rows = []  # (bytes, name, device, dtype, shape, obj)

    def add(name: str, obj: Any):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if torch is not None and isinstance(obj, getattr(torch, "Tensor", ())):
            if obj.is_cuda:
                nbytes = obj.element_size() * obj.numel()
                rows.append((nbytes, name, str(obj.device), str(obj.dtype), tuple(obj.shape), obj))
        elif cp is not None and isinstance(obj, getattr(cp, "ndarray", ())):
            if obj.device.id is not None:  # on a device
                nbytes = int(obj.nbytes)
                rows.append((nbytes, name, f"cuda:{obj.device.id}", str(obj.dtype), tuple(obj.shape), obj))


    def walk(prefix: str, obj: Any, level: int = 0):
        # limit depth to keep it fast
        if level > (2 if deep else 0):
            add(prefix, obj)
            return

        # base cases
        add(prefix, obj)

        # container/object descent if deep
        if not deep:
            return

        # dict-like
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}[{repr(k)[:24]}]", v, level + 1)
            return
        # sequence-like
        if isinstance(obj, (list, tuple, set, frozenset)):
            for i, v in enumerate(obj):
                walk(f"{prefix}[{i}]", v, level + 1)
            return
        # object attributes
        if hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                walk(f"{prefix}.{k}", v, level + 1)


    # walk top-level scope
    for k, v in scope.items():
        walk(k, v, level=0)

    # sort & print
    rows.sort(key=lambda r: r[0], reverse=True)
    header = f"{'name':48s} {'size':>10s} {'device':>10s} {'dtype':>12s} {'shape'}"
    print(header)
    print("-" * len(header))
    for nbytes, name, dev, dtype, shape, _ in rows[:top]:
        print(f"{name:48.48s} {_bytes_fmt(nbytes):>10s} {dev:>10s} {dtype:>12s} {shape}")
    return rows

def gpu_nvml_snapshot(device_index: int = 0):
    """
    Snapshot overall GPU memory and per-process usage via NVML.
    Requires: pip install pynvml
    """
    try:
        import pynvml as nv
    except Exception:
        print("pynvml not installed (pip install pynvml)")
        return None

    nv.nvmlInit()
    try:
        h = nv.nvmlDeviceGetHandleByIndex(device_index)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        name = nv.nvmlDeviceGetName(h).decode() if isinstance(nv.nvmlDeviceGetName(h), bytes) else nv.nvmlDeviceGetName(h)
        print(f"GPU {device_index}: {name}")
        print(f"  Memory: used {_bytes_fmt(mem.used)} / total {_bytes_fmt(mem.total)} ({100*mem.used/mem.total:.1f}%)")

        procs = []
        try:
            infos = nv.nvmlDeviceGetComputeRunningProcesses_v3(h)
        except Exception:
            infos = nv.nvmlDeviceGetComputeRunningProcesses(h)  # older API
        for p in infos:
            pid = p.pid
            used = getattr(p, "usedGpuMemory", getattr(p, "usedGpuMemory", 0))
            procs.append((used, pid))

        procs.sort(reverse=True)
        if procs:
            print("  Top processes by GPU mem:")
            for used, pid in procs[:10]:
                print(f"    pid {pid}  { _bytes_fmt(used) }")
        else:
            print("  No compute processes reported.")
        return {"name": name, "total": mem.total, "used": mem.used, "procs": procs}
    finally:
        nv.nvmlShutdown()


def ensure_dir(p: Path): 
    p.mkdir(parents=True, exist_ok=True)


def _pick_codec_and_suffix(): 

    if sys.platform == "darwin": #macOS 
        return [("mp4v", ".mp4"), ("MJPG", ".avi")]

    elif os.name == "nt": 
        return [("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")] 

    else: 
        return [("mpv4", ".mp4"), ("MJPG", ".avi")]


def open_writer(video_path: Path, fps:int, size_hw): 
    h, w = size_hw 
    candidates = _pick_codec_and_suffix() 

    ext = video_path.suffix.lower() 
    ordered = sorted(
        candidates, 
        key=lambda cs: 0 if cs[1] == ext else 1
    )
    for fourcc_str, suffix in ordered: 
        trial_path = video_path.with_suffix(suffix) 
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str) 
        vw = cv2.VideoWriter(
            filename=str(trial_path), 
            fourcc=fourcc, 
            fps=int(max(1,round(fps))), 
            frameSize=(w,h),
        )

        if vw.isOpened(): 
            return vw, trial_path, fourcc_str

    return None, None, None


def key_func(x): 
    filename = os.path.basename(x)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]
                

def read_yolo_labels(path:str, w:int, h:int): 
    data = np.loadtxt(path, ndmin=2) 

    if data.size == 0 :
        raise ValueError("Label format does not match yolo labels")

    cls = data[:,0].astype(np.int32)
    x_c, y_c, bw, bh = data[:,1:].T 
    x1 = (x_c - bw/2) * w 
    x2 = (x_c + bw/2) * w
    y1 = (y_c - bh/2) * h 
    y2 = (y_c + bh/2) * h
    boxes = np.stack([x1,y1,x2,y2], axis=1).astype(np.float32)
    return cls, boxes 

def build_gt_index(labels:list, fixed_size=None, image_dir:str=""): 
    gt_by_stem = {} 
    for f in labels: 
        stem = os.path.splitext(os.path.basename(f))[0]
        if fixed_size is not None: 
            w,h, = fixed_size 
        else: 
            for ext in (".jpg", ".jpeg", ".png"): 
                if os.path.exists(image_dir): 
                    im = cv2.imread(image_dir) 
                    h,w = im.shape[:2] 
                    break 
                else: continue

        cls, boxes = read_yolo_labels(f, w, h) 
        gt_by_stem[stem] = (cls, boxes) 

    return gt_by_stem


def _empty_dets_tensor(): 
    return (torch.zeros((0,4), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64))


def _empty_dets_numpy(): 
    return (np.zeros((0,4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64))


def _get_gt(stem: str, gt_by_stem:dict): 
    if stem in gt_by_stem: 
        return gt_by_stem[stem] 

    return (np.zeros((0,), np.float32),np.zeros((0,4), np.float32))


def _empty_results(orig_image, class_names:Optional[List[str]]=None, frame_id:int=0, device:str="cpu") -> Results: 
    empty = torch.zeros((0, 6), dtype=torch.float32, device=device)  # or device consistent with rest
   
    return Results(
            orig_img=orig_image, 
            boxes=empty, 
            names=class_names if class_names is not None else [], 
            speed={'preprocess':0.0, 'inference':0.0, 'postprocess':0.0}, 
            path=f"image_{frame_id}.jpg",
    )


def empty_image(image): 
    return np.zeros_like(image)


def return_no_motion_frames(im0s, batch_size): 
    batch_size = min(int(batch_size), len(im0s))
    results = [
        _empty_results(
            orig_image=im0s[i], 
            frame_id=i, 
            device="cpu"
        ) for i in range(batch_size)
    ]

    return results
