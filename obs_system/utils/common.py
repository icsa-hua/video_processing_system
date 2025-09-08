from obs_system.utils.logger import logger

import subprocess
import cv2
import os
import socket
import numpy as np
import torch 
from torchvision.ops import batched_nms, nms

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



def check_model_name(model_key:str,condition:str,condition_type:str): 

    model_validation = {
        'yolo': ('autoshape', 'y5'),
        'yolov5': ('autoshape', 'y5'),
        'yolov8': ('autobackbone', 'y8'),
        'yolov5s': ('autoshape', 'y5'),
        'yolov8s': ('autobackbone', 'y8'),
        'yolov5n': ('autoshape', 'y5'),
        'yolov8n': ('autobackbone', 'y8'),
        'yolo5': ('autoshape', 'y5'),
        'yolo8': ('autobackbone', 'y8'),
        'yolov5m': ('autoshape', 'y5'),
        'yolov8m': ('autobackbone', 'y8'),
        'onnx' : ('compressed', 'y8'), 
        'compressed' : ('compressed', 'y8') 
    }
    if model_key in model_validation and condition==condition_type:
            return model_validation[model_key][0]
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
        logger.debug("No original images or frames kept parts provided")

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
        




















