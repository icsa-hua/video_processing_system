from obs_system.compressed.interface.compressed_yolo import CompressedYOLO
from obs_system.compressed.interface.tensor_yolo import TensorRTYOLO
from obs_system.utils.global_config import *
from obs_system.utils.logger import get_logger 
from obs_system.utils.tiles import * 
from obs_system.utils.common import *
from obs_system.detection_module.interface.streamer import Streamer

import cv2 
import pdb
import time
import torch 
import numpy as np
from typing import Union, List, Any
from pathlib import Path 

from memory_profiler import profile as mem_profile
from abc import abstractmethod
from torch.profiler import profile, ProfilerActivity
from ultralytics import YOLO 
from ultralytics.data.augment import LetterBox
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.engine.results import Results
from ultralytics.utils import ops, colorstr

logger = get_logger("obs_system."+__name__)

class YOLOStreamer(Streamer): 


    @abstractmethod
    def __init__(self, cfg:str, overrides:dict, _callbacks:Any)->None:
        super().__init__(cfg, overrides, _callbacks) 


    @abstractmethod 
    def warmup(self, imgsz:tuple)->torch.Tensor:
        pass  


    # @abstractmethod
    def __call__(self, source:str, model:str, logic_module=None, mqtt_broker=None, producer_flag=None, preview_queue=None, *args, **kwargs)->None:
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
        if isinstance(self.model, YOLO) or isinstance(self.model, CompressedYOLO) : 
            pt = True 
            self.stride = 16 if self.args.half else 32 
        elif isinstance(self.model, TensorRTYOLO): 
            pt = True 
            self.stride = 16 if self.args.half else 32
        else: 
            pt = self.model.pt 
            self.stride = self.model.stride

        same_shapes =  len({x.shape for x in im}) == 1 #Ensure that all images have the same shape 
        letterbox = LetterBox(self.imgsz,auto=same_shapes ^ pt, stride=self.stride)
        return [letterbox(image=x) for x in im]
    

    # @abstractmethod
    def preprocess(self, im: Union[torch.Tensor, List[np.ndarray]])-> torch.Tensor | List[np.ndarray]:
        return super().preprocess(im)

    
    @abstractmethod
    def inference(self, im: torch.Tensor | List[np.ndarray], *args, **kwargs)->Any:
        pass


    # @abstractmethod
    def postprocess(self, preds:Any, img:Any, orig_imgs:Any)->Any : 
        return super().postprocess(preds, img, orig_imgs) 
    
    
    def setup_model(self, model:str, opt:str)->None:
        pass 


    # @abstractmethod
    # def non_max_suppression(self, detections:Any, scores:Any, iou:float)->Any:
    #     if len(detections)==0:
    #         logger.warning("No Detections were applicable from the model...")
    #         return[]
    #     
    #     return operation.nms(detections, scores, iou_threshold=iou)            
        

    @smart_inference_mode()
    def stream_inference(self, source:str, model:str, producer_flag:Any, preview_queue:Any, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            logger.info("")

        with self._lock:  # for thread-safe inference
            
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)
            
            # Prepare the images (Crop & Zoom) based on ROI 
            for batch in self.dataset:
                paths, im0s, s = batch
                if self.logic_module is not None and self.logic_module["ROI"] is not None: 
                    self.logic_module["ROI"].set_regions(im0s[0])
                break
           
            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
            
            # Warmup model
            if not self.model_warmup_done and not isinstance(self.model, YOLO) :

                if model == "yolov8":
                    self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                elif model == "yolov5":
                    self.warmup(imgsz=(1 if self.model.pt else self.dataset.bs, 3, *self.imgsz))
                elif model == 'engine': 
                   pass 
                self.model_warmup_done = True

            else: 
                self.warmup(imgsz=(1, 3, *self.imgsz))
                self.model_warmup_done = True 

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )

            self.run_callbacks("on_predict_start")
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            
            for self.batch in self.dataset:
                if self.runtime_limit_reached():
                    break

                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                
                if self.logic_module is not None and self.logic_module["ROI"] is not None: 
                    im0s = self.logic_module["ROI"].crop_image(im0s)

                motion_flags = self.logic_module["SUBTRACTOR"].detect(im0s,threshold=500) 

                # Filter batch by motion
                filtered_indices = [i for i, m in enumerate(motion_flags) if m]

                if not filtered_indices:
                    continue 

                paths = [paths[i] for i in filtered_indices]
                s = [s[i] for i in filtered_indices]
                tmp_im0s = [im0s[i] for i in filtered_indices]
                
                # Preprocess
                with profilers[0]:
                    images = self.preprocess(tmp_im0s)

                # Inference
                with profilers[1]:
                    if self.seen == 0:
                        with profile(activities=activities) as prof: 
                            preds = self.inference(images, *args, **kwargs)
                        prof.export_chrome_trace(f"trace_{model}.json")
                    else: 
                        preds = self.inference(images, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, images, im0s)

                if not isinstance(self.results[0], Results):
                    self.results = self.results[0]
                    self.results = torch.reshape(self.results, (self.results.shape[0], self.results.shape[2], self.results.shape[1]))

                self.run_callbacks("on_predict_postprocess_end")

                n = len(images)

                for i in range(n):
                    self.seen += 1
                    if isinstance(self.results[i], Results):
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                    else: 
                        self.speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }

                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), images, im0s, s)
                        self.publish_preview(preview_queue, producer_flag)
                        time.sleep(0.08)

                    self.capture_object_boxes(i, im0s[i], self.results[i], cropped_dirname=self.cropped_image_dirname) 
                
                if self.args.verbose:
                    logger.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            logger.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *images.shape[2:])}" % t
            )

        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            logger.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.release_session_resources(preview_queue=preview_queue, producer_flag=producer_flag)
        self.run_callbacks("on_predict_end")

   
