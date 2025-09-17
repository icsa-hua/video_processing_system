from obs_system.detection_module.interface.streaming import YOLOStreamer
from obs_system.utils.logger import get_logger

import torch 
import numpy as np
import cv2 
import warnings 
import os

from typing import Any
from pathlib import Path 
from collections import defaultdict
from ultralytics import YOLO 
from ultralytics.utils import DEFAULT_CFG
from ultralytics.engine.results import Results
from ultralytics.utils.files import increment_path
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

logger = get_logger("obs_system."+__name__)

class Yolov8Streamer(YOLOStreamer):

    # Ultralytics YOLO 🚀, AGPL-3.0 license

    def __init__(self, cfg:Any=DEFAULT_CFG, overrides=None, _callbacks=None)->None: 
        super().__init__(cfg, overrides, _callbacks)
        
        
    def warmup(self, imgsz=(1, 3, 640, 640)): 
        """Pytorch uses graph optimizations that are triggered with the first inference."""

        if self.device != 'cpu': 
            im = torch.empty(*imgsz, dtype=torch.float, device=self.device)
            y = self.model(im, show=False)
            if isinstance(y, (list,tuple)):
                return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
            else: 
                return self.from_numpy(y)
            

    def from_numpy(self, x): 
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x


    def __call__(self, source=None, model=None, logic_module=None, mqtt_broker=None, producer_flag=None, queue=None, *args, **kwargs):
        
        """Performs inference on an image, video or stream."""
        self.mqtt_interface = mqtt_broker
        self.args.stream_buffer = True
        self.logic_module = logic_module

        try: 
            self.predict_cli(source=os.path.normpath(os.path.abspath(source)) if  os.path.isfile(source)  else source,
                             model=model,
                             producer_flag=producer_flag,
                             queue=queue)
            
        except KeyboardInterrupt as ke: 
            if producer_flag is not None: 
                producer_flag.value=False
            if self.logic_module is not None and self.logic_module["DAV2"] is not None:
                self.logic_module["DAV2"].deallocate_resources()
            cv2.destroyAllWindows()
            logger.exception(f"KeyboardInterrupt: {ke}")
            return 


    def pre_transform(self, im): 
        return super().pre_transform(im)
    

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        
        not_tensor = not isinstance(im, torch.Tensor)

        
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        
        if not isinstance(self.model, YOLO): 
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        else: 
            im = im.float()  # uint8 to fp32

        if not_tensor:
            im = im.div(255.0)  # 0 - 255 to 0.0 - 1.0
        return im


    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )

        if isinstance(self.model, YOLO): 
            return self.model.track(im, augment=self.args.augment, visualize=visualize,embed=self.args.embed, conf=0.4, iou=0.5, verbose=False, show=False, persist=True, tracker="bytetrack.yaml", stream_buffer=self.args.stream_buffer)
        
        else: 
            return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)
    

    def postprocess(self, preds, img, orig_img): 
        return super().postprocess(preds, img, orig_img)
    
  
    
    def predict_cli(self, source, model, producer_flag=None, queue=None): 
        return super().predict_cli(source, model, producer_flag, queue) #sourcery skip: remove-empty-nested-block noqa


    def setup_source(self, source=""): 
        super().setup_source(source)


    def setup_model(self, model, verbose=True, opt='autobackbone'):
        
        """Initialize YOLO model with given parameters and set it to evaluation mode if needed."""
        if self.model: 
            return self.model

        device = select_device(self.args.device, verbose=verbose)

        if opt == "autobackbone": 
            self.model = AutoBackend(
                weights=model or self.args.model,
                device=device,
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
                batch=self.args.batch,
                fuse=True,
                verbose=verbose,
            )

            self.device = self.model.device  # update device
            self.args.half = self.model.fp16  # update half
            self.model.eval()

        elif opt=="tracking": 
            self.track_history = defaultdict(list)
            self.model = YOLO(model)
            self.model = self.model.to(device)
            self.device = self.model.device
            self.stride = 32 


    def non_max_suppression(self, detections,scores, iou):
        return super().non_max_suppression(detections,scores, iou)
        

    @smart_inference_mode()
    def stream_inference(self, source, model, producer_flag, queue, *args, **kwargs):
        return super().stream_inference(source, model, producer_flag, queue, *args, **kwargs)
        # ""Streams real-time inference on camera feed and saves results to file."""
    
      
    def write_results(self, i, p, im, original_images, s)->str:
        return super().write_results(i, p, im, original_images, s)
        


