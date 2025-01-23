from abc import ABC, abstractmethod
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, MACOS, WINDOWS,callbacks
from ultralytics.data.augment import LetterBox
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import colors
from ultralytics import YOLO 
from shapely.geometry import Polygon
from shapely.geometry.point import Point

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
import logging


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
    def __init__(self, cfg:str, overrides:dict, _callbacks:Any)->None:
        
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        
        self.done_warmup = False
        
        # if self.args.show:
        #     self.args.show = check_imshow(warn=True)

        self.process_memory = psutil.Process(os.getpid())
        self.callbacks = _callbacks or callbacks.get_default_callbacks() 

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
        self.orig_shape = None 
        self.proc_image = None
        self.points = dict() 

        #Counting Regions
        self.regions = self.count_regions()
        self.current_region = None

        callbacks.add_integration_callbacks(self)
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        print("Initialization Completed for YOLO Streaming")
        logging.info("Initialization Completed for YOLO Streaming")


    @abstractmethod 
    def warmup(self, imgsz:tuple)->torch.Tensor:
        pass  


    @abstractmethod 
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
        pt = None 
        stride=None

        if isinstance(self.model, YOLO): 
            pt = True 
            stride = 32 
        else: 
            pt = self.model.pt 
            stride = self.model.stride

        same_shapes =  len({x.shape for x in im}) == 1 #Ensure that all images have the same shape 
        letterbox = LetterBox(self.imgsz,auto=same_shapes ^ pt, stride=stride)
        return [letterbox(image=x) for x in im]
    

    @abstractmethod
    def preprocess(self, im: Union[torch.Tensor, List[np.ndarray]])-> torch.Tensor | List[np.ndarray]:
        pass

    
    @abstractmethod
    def inference(self, im: torch.Tensor | List[np.ndarray], *args, **kwargs)->Any:
        pass


    @abstractmethod
    def postprocess(self, preds:Any, img:Any, orig_imgs:Any)->Any : 
        """Post-processes predictions for an image and returns them."""

        return preds
    
    
    @abstractmethod
    def predict_cli(self, source:str, model:str, producer_flag:Any=None, queue:Any=None)->None: 
        """
        Method used for Command Line Interface (CLI) prediction.

        This function is designed to run predictions using the CLI. It sets up the source and model, then processes
        the inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
        generator without storing results.

        Note:
            Do not modify this function or remove the generator. The generator ensures that no outputs are
            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
        """
        gen = self.stream_inference(source, model, producer_flag, queue)
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
    def stream_inference(self, source:str, model:str,producer_flag:Any=None, queue:Any=None, *args, **kwargs)->None:
        pass


    @abstractmethod
    def write_results(self, i:Any, p:Any, im:Any, s:Any)->str:
        pass 


    @abstractmethod
    def save_predicted_images(self, save_path:str, frame:int)->None: 
        
        im = self.plotted_img 
        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = f'{save_path.split(".", 1)[0]}_frames/'
            
            if save_path not in self.vid_writer:  # new video
                
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            # Save video
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", im)

        # Save images
        else:
            cv2.imwrite(save_path, im)


    @abstractmethod
    def show(self, p=str)->None:
        im = self.plotted_img
        for region in self.regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2

            cv2.rectangle(im,(text_x - 5, text_y - text_size[1] - 5),(text_x + text_size[0] + 5, text_y + 5),region_color,-1,)
            cv2.putText(im, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2)
            cv2.polylines(im, [polygon_coords], isClosed=True, color=region_color, thickness=2)
        
        for cls in self.points.keys(): 
            cv2.polylines(im, [self.points[cls]], isClosed=False, color=colors(cls, True), thickness=2)

        if platform.system() == "Linux" and p not in self.windows: 
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.setMouseCallback(p, self.mouse_callback)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        if DEFAULT_CFG.gui:
            cv2.destroyAllWindows()
            self.proc_image = im
        else:
            cv2.imshow(p,im)
            cv2.waitKey(300 if self.dataset.mode == 'image' else 1)


    @abstractmethod
    def run_callbacks(self, event:str)->None: 
        for cb in self.callbacks.get(event, []): 
            cb(self) 


    @abstractmethod
    def add_callback(self, event: str, func:Any)->None: 
        self.callbacks[event].append(func)


    @abstractmethod
    def count_regions(self) -> list: 
        return [{
            "name": "YOLOv8 Polygon Region",
            "polygon": Polygon([(100, 150), (340, 150), (340, 450), (100, 450)]),  # Polygon points
            "counts": 0,
            "dragging": False,
            "region_color": (255, 42, 4),  # BGR Value
            "text_color": (255, 255, 255),  # Region Text Color
        },
        {
            "name": "YOLOv8 Rectangle Region",
            "polygon": Polygon([(300, 150), (540, 150), (540, 450), (300, 450)]),  # Polygon points
            "counts": 0,
            "dragging": False,
            "region_color": (37, 255, 225),  # BGR Value
            "text_color": (0, 0, 0),  # Region Text Color
        }]


    @abstractmethod
    def mouse_callback(self, event:int, x:int, y:int, flags:int, param:Any)->None:
            self.regions 
            # Mouse left button down event
            if event == cv2.EVENT_LBUTTONDOWN:
                for region in self.regions:
                    if region["polygon"].contains(Point((x, y))):
                        self.current_region = region
                        self.current_region["dragging"] = True
                        self.current_region["offset_x"] = x
                        self.current_region["offset_y"] = y

            # Mouse move event
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.current_region is not None and self.current_region["dragging"]:
                    dx = x - self.current_region["offset_x"]
                    dy = y - self.current_region["offset_y"]
                    self.current_region["polygon"] = Polygon(
                        [(p[0] + dx, p[1] + dy) for p in self.current_region["polygon"].exterior.coords]
                    )
                    self.current_region["offset_x"] = x
                    self.current_region["offset_y"] = y

            # Mouse left button up event
            elif event == cv2.EVENT_LBUTTONUP:
                if self.current_region is not None and self.current_region["dragging"]:
                    self.current_region["dragging"] = False

