from obs_system.detection_module.interface.streaming import YOLOStreamer
from obs_system.utils.logger import logger  

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
            warnings.warn(KeyboardInterrupt.__doc__)
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
            logger.info("Inference started")
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
        

    def save_predicted_images(self, save_path="", frame=0):
        return super().save_predicted_images(save_path, frame)
    

    def show(self, p=""): 
        return super().show(p)
    

    def run_callbacks(self, event): 
        return super().run_callbacks(event)
    

    def add_callback(self, event, func):
        return super().add_callback(event, func)


    def translate_data(self, frame_index, video_path, images, results, orig_images):
        
        """
        This function will only be needed in the case results are not of Type Results, 
        either use of AutoShape or AutoBackbone. 
        """
        
        names = self.model.names 
        orig_img = orig_images[frame_index]
        [height, width, _]= orig_img.shape
        shape = len(results.shape)
        try: 
            rectBoxes,class_ids = self.iteration_rows(results, frame_index, shape, height, width)
            if len(rectBoxes) == 0: 
                return []
            return Results(
                orig_img=orig_img,
                path=video_path, 
                names=names,
                boxes=rectBoxes,
                speed=self.speed,
                probs=class_ids
            )
        except KeyboardInterrupt as e: 
            exit(1)


    def calculate_padding(self, original_height, original_width, target_dimension):
        
        # Calculate the scale needed to fit the width and height into the target dimension
        # Determine which scale to use (the one that fits the image entirely within the target box)
        scale_used = min(target_dimension / original_width, target_dimension / original_height)

        # Calculate the effective width and height after scaling
        # Calculate padding by subtracting the effective dimensions from the target dimension
        effective_width = original_width * scale_used
        effective_height = original_height * scale_used
        
        return target_dimension - effective_width, target_dimension - effective_height, scale_used
    

    def box_creation(self, results, i, r, shape): 
        
        if shape == 3: 
            return [results[i, 0, r] - results[i, 2, r]/2,
                    results[i, 1, r] - results[i, 3, r]/2,
                    results[i, 0, r] + results[i, 2, r],
                    results[i, 1, r] + results[i, 3, r]]
        
        return [results[0, r] - results[2, r]/2,
                results[1, r] - results[3, r]/2,
                results[0, r] + results[2, r],
                results[1, r] + results[3, r]]


    def scale_boxes(self, boxes, pad_x, pad_y, scale): 
        boxes[:,[0,2]] -= pad_x // 2
        boxes[:,[1,3]] -= pad_y // 2 
        boxes[:, :4] /= scale
        return boxes 
    

    def data_to_tensor_filter(self, boxes, scores, class_ids): 
        boxes = torch.FloatTensor(boxes)
        scores = torch.FloatTensor(scores)
        class_ids = torch.LongTensor(class_ids)
        result_boxes = self.non_max_suppression(detections=boxes,scores=scores, iou=0.45)
        if len(result_boxes)==0 or len(boxes)==0: 
                return [],[],[]
        return boxes[result_boxes], scores[result_boxes], class_ids[result_boxes]
        

    def class_scores_creation(self, results, i, r, shape): 
        if shape == 3:
            return results[i, 4:, r]
        return results[4:, r]


    def iteration_rows(self, results, frame_index, shape, height, width, conf_thr=0.4): 
        boxes, class_ids, scores = [],[],[]

        pad_x,pad_y,scale = self.calculate_padding(height,width,640)

        for r in range(results.shape[-1]):
            classes_scores = self.class_scores_creation(results, frame_index, r, shape)
            
            (_, maxScore, _, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores.cpu().numpy()) 
            if maxScore >= conf_thr:
               boxes.append(self.box_creation(results, frame_index, r, shape))
               class_ids.append(maxClassIndex)
               scores.append(maxScore)
        try: 
            boxes, scores, class_ids = self.data_to_tensor_filter(boxes=boxes, scores=scores, class_ids=class_ids)
            boxes = self.scale_boxes(boxes, pad_x=pad_x, pad_y=pad_y, scale=scale)
            return torch.stack((boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3],scores, class_ids), axis=-1), class_ids

        except:
            boxes, scores,class_ids  = [],[],[]
            return torch.empty((0,6)),class_ids
        

    def count_regions(self):
       pass
    
    
    # def mouse_callback(self, event, x, y, flags, param) -> None:
    #     return super().mouse_callback(event, x, y, flags, param)
    

        

   
