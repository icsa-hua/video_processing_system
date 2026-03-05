from obs_system.compressed.interface.convert_to_Results import ConverterResults 
from obs_system.logic_module.dummy_logic.obstacle_filtering import classification_obstacles
from obs_system.utils.logger import get_logger
from obs_system.utils.common import *

import os
import pdb
import cv2
import torch
import logging
import platform
import threading 
import numpy as np 
import time, json
import queue 

from pathlib import Path
from io import StringIO
from typing import Union, List, Any, Optional, final, Generator
from abc import ABC, abstractmethod
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils.plotting import colors
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils import DEFAULT_CFG, callbacks
from ultralytics.data.build import load_inference_source
from ultralytics.utils.torch_utils import smart_inference_mode


class Streamer(ABC): 
    
    logger = get_logger("obs_system"+__name__)

    STREAM_WARNING = """
    WARNING ⚠️ Infefrence results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
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
        self.done_warmup = False
        
        self.save_queue = queue.Queue(maxsize=10)
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        self.seen = 0 
        self.speed = {} 
        self.windows = [] 
        self.vid_writer = {} 
        self.frame_images = {} 
        
        self.__check_docker_env() 

        self.mp: Any = None 
        self.model: Any = None 
        self.stride: Any = None 
        self.imgsz: Any = None 
        self.cropped_imgsz:Any = None
        self.original_imgsz:Any = None 
        self.use_roi:bool = False
        self.device: Any = None 
        self.dataset: Any = None
        self.plotted_img: Any = None 
        self.lanes_final: Any = None
        self.batch: Any = None 
        self.source_type: Any = None 
        self.results: Optional[List[Any]] = []
        self.txt_path: Optional[str] = None 
        self.proc_image: Optional[bool] = None 
        self.mqtt_interface:Any = None 
        self.logic_module: Any = None 
        self.tracker_model: Any = None 
        self.points = dict() 

        self._lock = threading.Lock() 
        self.converter = ConverterResults() 
        self.callbacks = _callbacks or callbacks.get_default_callbacks() 
        self.cropped_image_dirname = f'cropped_trial_{np.random.randint(44)}'
        callbacks.add_integration_callbacks(self) 
        

    def __check_docker_env(self): 
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


    def from_numpy(self, x:np.ndarray)->torch.Tensor:
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x


    @abstractmethod
    def __call__(self, source:str, model:str, logic_module=None, mqtt_broker=None, producer_flag=None, queue_list=None, *args, **kwargs)->None:
        pass


    @abstractmethod
    def pre_transform(self, im:List[np.ndarray])->list: 
        pass  


    @abstractmethod
    def preprocess(self, im: Union[torch.Tensor, List[np.ndarray]])-> torch.Tensor | List[np.ndarray]:
        pass


    @abstractmethod
    def postprocess(self, preds:Any, orig_image:Any)->Any : 
        
        if not isinstance(preds, Results): 
            raise ValueError("Not using ultralytics.Results class in postprocess of Streamer.") 
        
        if preds.boxes is not None and preds.boxes.xyxy.numel() == 0: 
            updated_labels, orig_classes_updated = classification_obstacles(
                boxes=preds.boxes, 
                classes=preds.boxes.cls, 
                lanes_final=self.lanes_final, 
                orig_shape=self.imgsz, 
                orig_classes=self.converter.class_names
            )

            if len(self.converter.class_names) != len(orig_classes_updated) and orig_classes_updated is not None: 
                self.converter.class_names = orig_classes_updated

                preds.names.clear() 
                for i, name in enumerate(self.converter.class_names): 
                    preds.names[i] = name

            if preds.boxes.cls.numel() == updated_labels.numel(): 
                preds.boxes.cls[:] = updated_labels

        preds.save_dir = self.save_dir.__str__() 

        try:  
            self.points.clear()
        except: 
            pass

        self.points = self.tracker_model.update_tracker_history(preds, logic_module=self.logic_module)
        
        try: 
            # TODO: Don't have only the option to save the image but instead also be able to transmit them through mqtt. 
            self.capture_object_boxes(
                    image=orig_image,
                    results=preds,
                    cropped_dirname=self.cropped_image_dirname,
                    save=self.args.save
            )
        except IndexError as ie: 
            Streamer.logger.exception(ie)


        return preds


    @final
    def predict_cli(self, source:str, model:str, producer_flag:Any=None, queue_list:Any=None)->None: 
        """
        Method used for Command Line Interface (CLI) prediction.

        This function is designed to run predictions using the CLI. It sets up the source and model, then processes
        the inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
        generator without storing results.

        Note:
            Do not modify this function or remove the generator. The generator ensures that no outputs are
            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
        """
        gen = self.stream_inference(source=source, model=model, producer_flag=producer_flag, queue_list=queue_list)
        for _ in gen: 
            pass 


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
            Streamer.logger.warning(Streamer.STREAM_WARNING)

        if self.args.verbose:
            Streamer.logger.debug("Dataset Source Type | {}".format(self.source_type))


    @abstractmethod
    def setup_model(self, model:str, opt:str)-> None: 
        pass


    @abstractmethod 
    @smart_inference_mode()
    def stream_inference(self, source:str, model:str, producer_flag:Any, queue_list:Any, *args, **kwargs)->Generator[Optional[Any], None, None]: 
        raise NotImplemented


    def write_results(self, preds:Any, i: Any, p: Any, im:Any, s:Any)->str: 
        """Write inference results to a file or directory."""

        string = "" 
        
        # # Ensure batch dimension
        if isinstance(im, list): 
            im = np.array(im) 
            
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

        self.__optional_save_or_show(preds, p)

        string += f"{preds.verbose()}{preds.speed['inference']:.1f}ms" 
          
        # Save results
        if self.args.save_txt:
            preds.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        
        if self.args.save_crop:
            preds.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem if self.txt_path is not None else Path("unknown"))





        return string


    def save_predicted_images(self, save_path:str, frame:int) ->None: 
        im = self.plotted_img 
        
        if im is not None: 
            self.save_queue.put((save_path, frame, im.copy()))


    def _save_worker(self):
        while True:
            task = self.save_queue.get()
            if task is None:
                break
            self._do_save(task)


    def _do_save(self, task):
        save_path, frame, im = task
        if im is None:
            return

        out_path = Path(save_path).expanduser()

        if out_path.name == "":
            Streamer.logger.error(f"Save predicted images: empty save path {save_path}")
            return

        ensure_dir(out_path.parent)

        if im.ndim == 3 and im.shape[2] == 3:
            bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        else:
            bgr = im

        is_stream_or_video = getattr(self.dataset, "mode", None) in {"stream", "video"}

        if is_stream_or_video:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            h, w = bgr.shape[:2]

            if h <= 0 or w <= 0:
                Streamer.logger.error("Invalid frame size")
                return

            vid_key = str(out_path.resolve())

            vw = self.vid_writer.get(vid_key)

            if vw is None:
                vw, opened_path, fourcc_used = open_writer(out_path, fps=fps, size_hw=(h, w))

                if vw is None:
                    Streamer.logger.error("VideoWriter failed to open for %s (fps=%s, size=%sx%s). "
                             "Check codec support in your OpenCV build.",
                             out_path, fps, w, h)
                    return

                self.vid_writer[vid_key] = vw

                Streamer.logger.info("Opened VideoWriter: %s (fourcc=%s)", opened_path, fourcc_used)

                if self.args.save_frames and opened_path is not None:
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
                    Streamer.logger.warning("cv2.imwrite failed: %s", img_path)

        else:
            # Save a single image
            img_path = out_path
            ok = cv2.imwrite(str(img_path), bgr)
            if not ok:
                Streamer.logger.error("cv2.imwrite failed: %s", img_path)


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


    def __optional_save_or_show(self, preds:Any, p:Any)-> None: 

        if self.args.save or self.args.show: 
            self.plotted_img = preds.plot(
                    line_width=self.args.line_width,
                    boxes=self.args.show_boxes,
                    conf=self.args.show_conf,
                    labels=self.args.show_labels,
            )

        if self.args.show:
            self.show(p)     

        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), int(self.dataset.count))    


    def __generate_mqtt_message(self, preds:Any, frame_index:int)->str: 
            return json.dumps({
                "frame_id":frame_index, 
                "classes":preds.boxes.cls.tolist(), 
                "boxes": preds.boxes.xyxy.tolist(), 
                "tm_ms":time.time()*1000, 
                "track_ids": preds.boxes.id.tolist(), 
            })
    
    def __generate_mqtt_message_no_motion(self, preds:Any, frame_index:list)->str: 
        messages = [] 
        for idx, frame_id in enumerate(frame_index): 
            boxes = preds[idx].boxes
            message = {
                "frame_id":frame_id, 
                "classes":boxes.cls.tolist(), 
                "boxes": boxes.xyxy.tolist(), 
                "tm_ms":time.time()*1000, 
                "track_ids": boxes.id.tolist(), 
            }
            messages.append(message)
        return json.dumps(messages)
    

    @abstractmethod
    def _publish_mqtt_message(self, preds, frame_index)->None: 
        if self.mqtt_interface is not None: 
            message = self.__generate_mqtt_message(preds, frame_index)
            self.mqtt_interface.publish(self.mqtt_interface.topic, message)


    @abstractmethod
    def _publish_mqtt_message_no_detection(self, preds, frame_index)->None: 
        if self.mqtt_interface is not None: 
            message = self.__generate_mqtt_message_no_motion(preds, frame_index)
            self.mqtt_interface.publish(self.mqtt_interface.topic, message)
