from abc import ABC, abstractmethod
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG,MACOS, WINDOWS,callbacks
from ultralytics.data.augment import LetterBox
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.checks import check_imshow

import platform 
import psutil 
import os 
import cv2 
import threading 
import torch 
import numpy as np
from typing import Union, List, Any
from pathlib import Path 
import pynvml

class YOLOStreamer(ABC): 

    STREAM_WARNING = """
    WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
    errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

    Example:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
    """

    @abstractmethod
    def __init__(self, cfg:DEFAULT_CFG, overrides:dict, _callbacks:Any)->None:
        
        
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        
        self.done_warmup = False
        
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        self.process_memory = psutil.Process(os.getpid())
        self.callbacks = _callbacks or callbacks.get_default_callbacks() 
        
        if self.args.conf is None: 
            self.args.conf = 0.25 
        self.done_warmup = False 
        
        if self.args.show: 
            self.args.show = check_imshow(warn=True)

        self.model = None 
        self.data = self.args.data
        self.imgsz = None 
        self.device = None 
        self.dataset = None 
        self.vid_writer = {} 
        self.plotted_img = None
        self.source_type = None 
        self.seen = 0
        self.windows = [] 
        self.batch = None 
        self.results = None
        self.transforms = None 
        self.callbacks = _callbacks or callbacks.get_default_callbacks() 
        self.txt_path = None 
        self._lock = threading.Lock() 
        self.shape = None 

        callbacks.add_integration_callbacks(self)
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)


    @abstractmethod 
    def warmup(self, imgsz:tuple)->torch.Tensor:
        pass  


    @classmethod 
    def from_numpy(self, x:np.ndarray)->torch.Tensor:
        pass 


    @abstractmethod
    def __call__(self, source:str, model:str, stream:bool, *args, **kwargs)->None:
        pass


    @abstractmethod
    def pre_transform(self, im:List[np.ndarray])->list: 
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes =  len({x.shape for x in im}) == 1
        letterbox = LetterBox(self.imgsz,auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]
    
    @abstractmethod
    def preprocess(self, im: Union[torch.Tensor, List[np.ndarray]])-> torch.Tensor | List[np.ndarray]:
        pass
    
    @abstractmethod
    def inference(self, im: torch.Tensor | List[np.ndarray], *args, **kwargs)->Any:
        pass

    @abstractmethod
    def postprocess(self, preds:Any)->Any : 
        return preds 
    
    @abstractmethod
    def predict_cli(self, source:str, model:str)->None: 
        gen = self.stream_inference(source, model)
        for _ in gen: 
            pass 
        #Sourcery skip: remove empty nested block noqa 


    @abstractmethod
    def setup_source(self, source:str)->None:
        pass


    @abstractmethod
    def setup_model(self, model:str, verbose:bool)->None:
        pass 

    @abstractmethod
    def non_max_suppression(self, opt:str, detections:Any, iou_thres:float)->Any:
        pass 

    @abstractmethod
    def translate_data(self)->None:
        pass


    @abstractmethod
    @smart_inference_mode()
    def stream_inference(self, source:str, model:str, *args, **kwargs)->None:
        pass



    @abstractmethod
    def write_results(self, i:Any, p:Any, im:Any, s:Any)->str:
        pass 

    @abstractmethod
    def save_predicted_images(self, save_path:str, frame:int)->None: 
        im = self.plotted_img 

        if self.dataset.mode in {"strean","video"}: 
            fps = self.dataset.fps if self.dataset.mode =="video" else 30
            frames_path = f'{save_path.split(".",1)[0]}_frames/'
            if save_path not in self.vid_writer: 
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = {".mp4","avc1"} if MACOS else (".avi","WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc), 
                    fps=fps, 
                    frameSize=(im.shape[1], im.shape[0])   
                )
                
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", im) 

        else:
            cv2.imwrite(save_path, im)

    @abstractmethod
    def show(self, p=str)->None:
        im = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows: 
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])
        cv2.imshow(p,im)
        cv2.waitKey(300 if self.dataset.mode == 'image' else 1)


    @classmethod
    def run_callbacks(self, event:str)->None: 
        for cb in self.callbacks.get(event, []): 
            cb(self) 

    @classmethod
    def add_callback(self, event: str, func:Any)->None: 
        self.callbacks[event].append(func)
