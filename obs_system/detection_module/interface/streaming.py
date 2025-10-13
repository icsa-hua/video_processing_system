from obs_system.compressed.interface.compressed_yolo import CompressedYOLO
from obs_system.compressed.interface.tensor_yolo import TensorRTYOLO
from obs_system.logic_module.dummy_logic.region_setter import RegionSetter
from obs_system.compressed.interface.convert_to_Results import ConverterResults 
from obs_system.utils.logger import get_logger 
from obs_system.utils.tiles import * 
from obs_system.utils.appraisal import StepContext
from obs_system.utils.common import get_frame_ids, ensure_dir, open_writer

import re
import os 
import cv2 
import pdb
import time
import torch 
import threading 
import platform 
import logging
import numpy as np
import torchvision.ops as operation
from typing import Union, List, Any
from pathlib import Path 
from io import StringIO 

from abc import ABC, abstractmethod
from collections import deque, defaultdict
from torch.profiler import profile, ProfilerActivity
from ultralytics import YOLO 
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.augment import LetterBox
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import colors
from ultralytics.data.build import load_inference_source
from ultralytics.utils.checks import check_imgsz
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, MACOS, WINDOWS,callbacks, ops, colorstr

logger = get_logger("obs_system."+__name__)

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
        self.docker_flag = False 
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        
        self.done_warmup = False
        
        if self.args.show:
            if os.path.exists("/.dockerenv") or os.getenv("container") == "docker":
                self.docker_flag = True 
                log_stream = StringIO() 
                log_handler = logging.StreamHandler(log_stream)
                logger_ultra = logging.getLogger("ultralytics")
                logger_ultra.addHandler(log_handler)
                self.args.show = check_imshow(warn=True)
                log_handler.flush()
                log_contents = log_stream.getvalue()
                logger_ultra.removeHandler(log_handler)
                    
                if "WARNING ⚠️" in log_contents:
                    self.args.show = True # Probably we are on a docker, where with streamlit we can show the images. 
            else: 
                self.args.show = check_imshow(warn=True)

        self.seen = 0
        self.speed = {} 
        self.windows = [] 
        self.trackers = [] 
        self.vid_writer = {} 
        self.frame_images = {}
        self.data = self.args.data

        self.model:Any = None 
        self.stride:Any = None
        self.imgsz:Any = None 
        self.device:Any = None 
        self.dataset:Any = None 
        self.plotted_img = None
        self.source_type:Any = None 
        self.batch:Any = None 
        self.results:Any = None
        self.txt_path = None 
        self.proc_image = None
        self.logic_module:Any = None
        self.tracker_model:Any = None
        self.points = dict() 

        self._lock = threading.Lock()
        self.mqtt_interface:Any = None

        self.callbacks = _callbacks or callbacks.get_default_callbacks() 
        self.cropped_image_dirname = f'cropped_trial_{np.random.randint(44)}'
         
        #ConverterResults
        self.converter = ConverterResults() 

        callbacks.add_integration_callbacks(self)
        if self.args.verbose: 
            logger.info("Initialization Completed for Streamer")


    @abstractmethod 
    def warmup(self, imgsz:tuple)->torch.Tensor:
        pass  


    @abstractmethod 
    def from_numpy(self, x:np.ndarray)->torch.Tensor:
        pass 


    @abstractmethod
    def __call__(self, source:str, model:str, logic_module=None, mqtt_broker=None, producer_flag=None, queue=None, *args, **kwargs)->None:
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
        """Sets up source and inference mode."""

        self.imgsz = check_imgsz(self.args.imgsz, stride=self.stride,min_dim=2) 

        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer
        )
        
        self.source_type = self.dataset.source_type

        if not getattr(self,"stream", True ) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000 # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ): 
            logger.warning(YOLOStreamer.STREAM_WARNING)

        if self.args.verbose:
            logger.debug("Dataset Source Type | {}".format(self.source_type))


    @abstractmethod
    def setup_model(self, model:str, opt:str)->None:
        pass 


    @abstractmethod
    def non_max_suppression(self, detections:Any, scores:Any, iou:float)->Any:
        if len(detections)==0:
            logger.warning("No Detections were applicable from the model...")
            return[]
        
        return operation.nms(detections, scores, iou_threshold=iou)            
        

    @smart_inference_mode()
    def stream_inference(self, source:str, model:str, producer_flag:Any, queue:Any, *args, **kwargs):
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
            if not self.done_warmup and not isinstance(self.model, YOLO) :

                if model == "yolov8":
                    self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                elif model == "yolov5":
                    self.warmup(imgsz=(1 if self.model.pt else self.dataset.bs, 3, *self.imgsz))

                self.done_warmup = True

            else: 
                self.warmup(imgsz=(1, 3, *self.imgsz))
                self.done_warmup = True 

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )

            self.run_callbacks("on_predict_start")
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            
            for self.batch in self.dataset:
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

                if self.logic_module is not None and self.logic_module["DAV2"] is not None: 
                    fps = self.dataset.fps if self.dataset.mode == "video" else 30 
                    self.logic_module["DAV2"].detect(images, fps, "runs/detect/DI_results/dav2_detections" )

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
                        if producer_flag is not None: 
                            producer_flag.value = True
                        if self.proc_image is not None and queue is not None:
                            queue.put(self.proc_image)
                        elif self.proc_image is None and queue is not None: 
                            queue.put(None)    
                        time.sleep(0.08)

                    self.capture_object_boxes(i, im0s[i], self.results[i], cropped_dirname=self.cropped_image_dirname) 
                
                if self.args.verbose:
                    logger.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

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
        
        self.run_callbacks("on_predict_end")

    
    @abstractmethod
    def write_results(self, i:Any, p:Any, im:Any, original_images:Any, s:Any)->str:
        """Write inference results to a file or directory."""
        
        string = "" 

        # Ensure batch dimension
        if len(im.shape) == 3:
            im = im[None]  
        
        # Determine frame index
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined

        self.txt_path = self.save_dir / "labels" / (
            p.stem + ("" if self.dataset.mode == "image" else f"_{frame}")
        )
        string += "%gx%g " % im.shape[2:]

        #Get the batch size pictures 
        result = self.results[i] 
        if isinstance(result, torch.Tensor):
            # result = self.converter.translate_data(i, p, im, result, original_images)
            # if not result: 
                # return "(No Detection Found)"
            raise ValueError("Not using pytorch and the ultralytics.Results class")
        
        if self.mqtt_interface is not None:
            self.mqtt_interface.publish(self.mqtt_interface.topic, str(result.speed))

        result.save_dir = self.save_dir.__str__() 

        string += f"{result.verbose()}{result.speed['inference']:.1f}ms" 
        try:  
            self.points.clear()
        except: 
            pass
        self.points = self.tracker_model.update_tracker_history(result, logic_module=self.logic_module)
        
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                    line_width=self.args.line_width,
                    boxes=self.args.show_boxes,
                    conf=self.args.show_conf,
                    labels=self.args.show_labels,
            )

        # Save results
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        
        if self.args.show:
            self.show(p)
        
        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), int(frame))

        return string
            


    def save_predicted_images(self, save_path:str, frame:int)->None: 
        
        im = self.plotted_img 

        if im is None: return 

        out_path = Path(save_path).expanduser() 

        if out_path.name == "": 
            logger.error(f"Save predicted images: empty save path {save_path}")
            return 

        ensure_dir(out_path.parent) 
        
        if im.ndim==3 and im.shape[2]==3: 
            bgr = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else: 
            bgr = im 
        
        is_stream_or_video = getattr(self.dataset, "mode", None) in {"stream", "video"}
        if is_stream_or_video: 
            fps = self.dataset.fps if self.dataset.mode == "video" else 30 
            h, w = bgr.shape[:2] 

            if h <= 0 or w <= 0: 
                logger.error("Invalid frame size")
                return 

            vid_key = str(out_path.resolve())

            vw = self.vid_writer.get(vid_key) 
            if vw is None: 
                vw, opened_path, fourcc_used = open_writer(out_path, fps=fps, size_hw=(h,w))
                if vw is None: 
                    logger.error("VideoWriter failed to open for %s (fps=%s, size=%sx%s). "
                             "Check codec support in your OpenCV build.",
                             out_path, fps, w, h)
                    return
                self.vid_writer[vid_key] = vw 
                logger.info("Opened VideoWriter: %s (fourcc=%s)", opened_path, fourcc_used)
                if self.args.save_frames: 
                    frames_dir = opened_path.with_suffix("").parent / (opened_path.stem + "_frames")
                    ensure_dir(frames_dir)
                    self._frames_dir_cache = getattr(self, "_frames_dir_cache", {})
                    self._frames_dir_cache[vid_key] = frames_dir
            
            self.vid_writer[vid_key].write(bgr)

            if getattr(self.args, "save_frames", False):
                frames_dir = getattr(self, "_frames_dir_cache", {}).get(vid_key)

                if frames_dir is None:
                    frames_dir = out_path.with_suffix("").parent / (out_path.stem + "_frames")
                    ensure_dir(frames_dir)
                    self._frames_dir_cache[vid_key] = frames_dir
                img_path = frames_dir / f"{int(frame):06d}.jpg"

                ok = cv2.imwrite(str(img_path), bgr)
                if not ok:
                    logger.warning("cv2.imwrite failed: %s", img_path)

        else:
            # Save a single image
            img_path = out_path
            ok = cv2.imwrite(str(img_path), bgr)
            if not ok:
                logger.error("cv2.imwrite failed: %s", img_path)


    def show(self, p:str)->None:
        im = self.plotted_img

        if im is None: 
            return 
        
        
        if self.logic_module is not None and self.logic_module['ROI'] is not None:
            self.logic_module["ROI"]._show_regions(im)

        for cls in self.points.keys(): 
            cv2.polylines(im, [self.points[cls]], isClosed=False, color=colors(cls, True), thickness=2)

        if self.docker_flag:
            self.proc_image = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            return 

        elif platform.system() == "Linux" and p not in self.windows: 
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        if DEFAULT_CFG.gui:
            cv2.destroyAllWindows()
            self.proc_image = im
        elif self.args.show: 
            cv2.imshow(winname=p, mat=im)
            cv2.waitKey(300 if self.dataset.mode == 'image' else 1)


    def run_callbacks(self, event:str)->None: 
        for cb in self.callbacks.get(event, []): 
            cb(self) 


    def add_callback(self, event: str, func:Any)->None: 
        self.callbacks[event].append(func)

    
    def capture_object_boxes(self, image:np.ndarray|torch.Tensor,results:Any, cropped_dirname:str, save:bool=True): 

        if results is None:
            return 

        mask = np.zeros_like(image)
        orig_h, orig_w = image.shape[:2]
        
        infer_h, infer_w = results.orig_shape  

        # Calculate resize ratio and padding used in letterboxing
        scale = min(infer_w / orig_w, infer_h / orig_h)
        pad_w = (infer_w - orig_w * scale) / 2
        pad_h = (infer_h - orig_h * scale) / 2

        for _, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(float, box.xyxy[0])

        # Remove padding and rescale back to original image size
            x1 = int((x1 - pad_w) / scale)
            x2 = int((x2 - pad_w) / scale)
            y1 = int((y1 - pad_h) / scale)
            y2 = int((y2 - pad_h) / scale)

        # Clip to original image boundaries
            x1, x2 = max(0, x1), min(orig_w, x2)
            y1, y2 = max(0, y1), min(orig_h, y2)

            mask[y1:y2,x1:x2] = image[y1:y2,x1:x2]

        cropped_image_dir = os.path.join(os.getcwd(), 'assets') 
        if not os.path.exists(cropped_image_dir) : 
            os.mkdir(cropped_image_dir) 

        image_dir = os.path.join(cropped_image_dir, cropped_dirname) 
        if not os.path.exists(image_dir): 
            os.mkdir(image_dir) 

        save_cropped_img = f"{image_dir}/masked_frame_{np.random.randint(10000)}.jpg"
        if save:
            cv2.imwrite(save_cropped_img, mask) 


    def admit_frames(self, fr_queue:deque, max_f_inf:int, use_roi:bool): 

        while len(fr_queue) <= max_f_inf: 
            try:
               self.batch = next(self.dataset) 
            except StopIteration: 
                break

            im0s = self.batch[1] 
            frame_ids = get_frame_ids(labels=self.batch[2])
            if 'frame 1' in self.batch[2][0] and use_roi:
                self.logic_module['ROI'].set_regions(im0s[0]) 
                self.orig_height, self.orig_width = im0s[0].shape[:2]

            with StepContext(name="Crop & Subtraction", catch=(RuntimeError,), verbose=self.args.verbose): 

                if use_roi: 
                    im0s = self.logic_module['ROI'].crop(im0s)

                mfgs = self.logic_module["SUBTRACTOR"].detect(im0s, save_img=True)

                if not any(mfgs): 
                    continue

                mfgs = np.asarray(mfgs, np.int8).tolist()

            with StepContext(name="Fill Frame Queue", catch=(RuntimeError,), verbose=self.args.verbose): 
                for ind, (f_id, im) in enumerate(zip(frame_ids, im0s)): 
                    if not mfgs[ind]: continue 
                    fr_queue.append((f_id, im))
        
        return fr_queue


    def iter_data(self, use_roi:bool): 
        # Yields (frame_id, image) lazily from self.dataset after ROI + MOG2 gating 

        self.dataset = iter(self.dataset) 

        while True: 
            try: 
                self.batch = next(self.dataset)
            except StopIteration: 
                return 

            _, im0s, s = self.batch 
            frame_ids = get_frame_ids(labels=s) 

            # One-time ROI init on first frame if needed 
            if use_roi and ("frame_1" in s[0] or self.seen == 0): 
                self.logic_module['ROI'].set_regions(im0s[0]) 
                self.orig_height, self.orig_width = im0s[0].shape[:2] 
 
            with StepContext(name="Crop & Subtract", catch=(RuntimeError, ), verbose=self.args.verbose): 

                if use_roi: 
                    im0s = self.logic_module['ROI'].crop(im0s)
                
                # Motion gate (vectorized over the mini batch) 
                mfgs = self.logic_module["SUBTRACTOR"].detect(im0s, save_img=False) 
                
            for passed, f_id, img in zip(mfgs, frame_ids, im0s): 
                if passed: 
                    yield (int(f_id), img)
    

    def _frames_to_tiles(self, frame_iter, tile_size:int, overlap_ratio:float): 
        overlap_px = max(0, int(round(tile_size * overlap_ratio)))
        for f_id, img in frame_iter: 
            self.frame_images[f_id] = img
            for t, m in split_image_gen(img, f_id, tile_size=tile_size, overlap=overlap_px):
                yield t, m



