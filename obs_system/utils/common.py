import re
import subprocess
import cv2
import sys
import os
import socket
import numpy as np
import torch 

from typing import List
from pathlib import Path
from torchvision.ops import batched_nms, nms
from typing import Dict, Any, Iterable

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


def check_model_name(model_key:str,condition:str): 

    model_validation = {
        'yolo': ('autoshape', 'yolov8s'),
        'yolov5': ('autoshape', 'yolov5s'),
        'yolov8': ('autobackbone', 'yolov8s'),
        'yolov5s': ('autoshape', 'yolov5s'),
        'yolov8s': ('autobackbone', 'yolov8s'),
        'yolov5n': ('autoshape', 'yolov5n'),
        'yolov8n': ('autobackbone', 'yolov8n'),
        'yolo5': ('autoshape', 'yolov5n'),
        'yolo8': ('autobackbone', 'yolov8s'),
        'yolov5m': ('autoshape', 'yolov5m'),
        'yolov8m': ('autobackbone', 'yolov8m'),
        'onnx' : ('compressed', 'onnx'), 
        'compressed' : ('compressed', 'onnx'), 
        'trt':('trt', 'engine'),
        'trt-onnx':('trt', 'onnx')
    }
    model_validation_2 = {
        'yolo': ('YOLO', 'yolov8s'),
        'yolov5': ('YOLO', 'yolov5s'),
        'yolov8': ('YOLO', 'yolov8s'),
        'yolov5s': ('YOLO', 'yolov5s'),
        'yolov8s': ('YOLO', 'yolov8s'),
        'yolov5n': ('YOLO', 'yolov5n'),
        'yolov8n': ('YOLO', 'yolov8n'),
        'yolo5': ('YOLO', 'yolov5n'),
        'yolo8': ('YOLO', 'yolov8s'),
        'yolov5m': ('YOLO', 'yolov5m'),
        'yolov8m': ('YOLO', 'yolov8m'),
        'onnx' : ('compressed', 'onnx'), 
        'compressed' : ('compressed', 'onnx') ,
        'trt':('trt', 'engine'), 
        'trt-onnx':('trt', 'onnx')
    }
    
    if condition == 'tracking': 
        if model_key in model_validation_2: 
            return model_validation_2[model_key][1], model_validation_2[model_key][0]
        else: 
            raise ValueError("No valid model was provided...\nUse 'yolov8' as an example")

    else: 

        if model_key in model_validation:
                return model_validation[model_key][1], model_validation[model_key][0]
        else: 
            raise ValueError("No valid model was provided...\nUse 'yolov8' as an example")



def _make_colors(num_classes): 
    rng = np.random.default_rng(12345) 
    base = rng.integers(64,256,size=(num_classes,3),dtype=np.uint8)
    return [tuple(int(c) for c in base[i].tolist()) for i in range(num_classes)]


def draw_and_save_frames(
    orig_images, 
    frames_out, 
    class_names, 
    out_dir="assets/save_inferences/", 
    thickness=2, 
    font = cv2.FONT_HERSHEY_SIMPLEX, 
    font_scale=0.5, 
    text_thickness=1, 
    save=False, 
    show=False
):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir) 

    if not orig_images or frames_out: 
        raise ValueError("No original images or process frames")

    num_classes = max((int(c.max()) for _, (_,_,c) in frames_out.items() if len(c)), default=-1) + 1 
    colors = _make_colors(max(num_classes, len(class_names)))

    for index, f_id in enumerate(frames_out): 

        boxes, scores, classes = frames_out[f_id]
        boxes_t = torch.from_numpy(boxes)
        scores_t = torch.from_numpy(scores) 
        classes_t = torch.from_numpy(classes) 
        image = orig_images[index]
        
        if boxes is None or len(boxes) == 0: 
            cv2.imwrite(os.path.join(out_dir, f"{f_id}.jpg"), image)
            continue

        keep = batched_nms(boxes_t, scores_t, classes_t.long(), 0.2)
        keep_nms = nms(boxes_t, scores_t, 0.4) 

        keep_ind = torch.searchsorted(keep.sort().values, keep_nms.sort().values) 
        keep = keep[keep_ind]
        
        if save or show: 
            boxes = boxes[keep]
            scores = scores[keep] 
            classes = classes[keep]
            boxes = boxes.astype(np.int32, copy=False)

            for (x1, y1, x2, y2), sc, cid in zip(boxes, scores, classes): 
                color = colors[int(cid) % len(colors)] 

                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA) 
                cls_name = class_names[cid] if 0 <= cid < len(class_names) else str(cid) 
                label = f"{cls_name} {sc:.2f}" 

                (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness) 
                ty1 = max(y1 - th - 4, 0) 
                cv2.rectangle(image, (x1, ty1), (x1 + tw + 4, ty1 + th + 4), color, -1) 
                cv2.putText(image, label, (x1 + 2, ty1 + th + 2), font, font_scale, (0,0,0), thickness=text_thickness, lineType=cv2.LINE_AA) 

        if save: 
            cv2.imwrite(os.path.join(out_dir, f"{f_id}.jpg"), image)

        if show: 
            cv2.imshow("Test_image", orig_images[-1])

        frames_out[f_id] = (boxes_t[keep], scores_t[keep], classes_t[keep]) 
        
    return frames_out


def get_frame_ids(labels:List[str])->List[int]: 

    frame_ids = [value.split(' ') for value in labels]
    frame_ids = [id[3] for id in frame_ids]  
    frame_ids = [re.sub(r'[^\w]','|', id) for id in frame_ids]
    frame_ids = [id.split('|') for id in frame_ids]
    frame_ids = [int(id[0]) for id in frame_ids] 
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
