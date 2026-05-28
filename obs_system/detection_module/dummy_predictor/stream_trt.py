from obs_system.detection_module.interface.streaming_compressed import OptimizedStreamer
from obs_system.utils.benchmarking.metrics.model_performance import ModelPerf
from obs_system.logic_module.dummy_logic.tracker_sv import TrackerHandler
from obs_system.compressed.interface.tensor_yolo import TensorRTYOLO
from obs_system.utils.logger import get_logger
from obs_system.utils.appraisal import StepContext
from obs_system.utils.global_config import CONF_THR, NMS_IOU, WARM_UP_SESSIONS, BATCH_SIZE

import time
import torch
import numpy as np

from typing import Any, Generator, Optional
from memory_profiler import profile as mem_profile
from collections import defaultdict
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


logger = get_logger("obs_system."+__name__)


class TensorRTRTXStreamer(OptimizedStreamer): 

    def __init__(self, cfg:Any=DEFAULT_CFG, overrides=None, _callbacks=None)->None: 
        super().__init__(cfg,overrides, _callbacks)
        self.source = ""
        self.__gt_labels = None if self.args.bench is None else defaultdict()


    def __call__(self, source=None, model=None, logic_module=None, mqtt_broker=None,producer_flag=None, preview_queue=None, *args, **kwargs): 
        super().__call__(source, model, logic_module, mqtt_broker, producer_flag, preview_queue , *args, **kwargs)


    def pre_transform(self, im): 
        return super().pre_transform(im) 


    def preprocess(self,im:Any): 
        """
        Prepare input image before inference. 

        Args: 
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list. 
        """

        not_tensor = not isinstance(im, torch.Tensor) 

        if not_tensor: 
            if isinstance(im, np.ndarray): im = list(im)
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im) 
            im = torch.from_numpy(im) 

        im = im.to(self.device) 

        if (not isinstance(self.model, YOLO) and not isinstance(self.model, TensorRTYOLO )): 
            im = im.half() if self.model.fp16 else im.float() 

        else: 
            im = im.float() 

        if not_tensor:
            im = im.div(255.0)  # 0 - 255 to 0.0 - 1.0

        return im


    def postprocess(self, preds, orig_image)->Any: 
        return super().postprocess(preds, orig_image) 


    def setup_model(self, model_name:str="", path_to_load:str="", opt='tracking'): 
        device = select_device(self.args.device, verbose=self.args.verbose) 
        
        self.model = TensorRTYOLO(model_name=model_name, engine_path=path_to_load, fp16=True)

        [self.height, self.width] = self.model.input_height, self.model.input_width 

        self.device = device 

        self.tracker_model = TrackerHandler(tracker_choice="byte_tracker") if opt == "tracking" else None 

        self.stride = 32 if not self.args.half else 16

        if self.args.verbose: 
            logger.info(f"[checked] Model {model_name} successfully set up")
        else: 
            logger.debug(f"[checked]  Model {model_name} successfully set up")


    @smart_inference_mode()
    def stream_inference(self, source, model, producer_flag, preview_queue, *args, **kwargs): 
        return super().stream_inference(source, model, producer_flag, preview_queue, *args, **kwargs) 


    @mem_profile
    def _stream_inference_impl_tiles(self, **kwargs) -> Generator[Optional[Any], None, None]:
        return super()._stream_inference_impl_tiles(**kwargs)        

    @mem_profile
    def _stream_inference_impl(self, **kwargs): 
        return super()._stream_inference_impl(**kwargs) 

    def _publish_mqtt_message(self, preds, mqtt_messages, frame_ids)->None: 
        super()._publish_mqtt_message(preds, mqtt_messages, frame_ids)

    def _publish_mqtt_message_no_detection(self, preds, frame_index)->None: 
        super()._publish_mqtt_message_no_detection(preds, frame_index)
















