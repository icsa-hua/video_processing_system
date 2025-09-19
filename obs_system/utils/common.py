import re
import subprocess
import cv2
import os
import socket
import numpy as np
import torch 

from typing import List
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
        'compressed' : ('compressed', 'onnx') 
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
        'compressed' : ('compressed', 'onnx') 
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


 














