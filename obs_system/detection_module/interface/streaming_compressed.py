from obs_system.utils.common import _get_gt, _empty_dets_numpy, _empty_results
from obs_system.compressed.interface.compressed_yolo import CompressedYOLO
from obs_system.compressed.interface.tensor_yolo import TensorRTYOLO
from obs_system.detection_module.interface.streamer import Streamer
from obs_system.utils.global_config import BATCH_SIZE
from obs_system.utils.appraisal import StepContext
from obs_system.utils.global_config import *
from obs_system.utils.common import *
from obs_system.utils.tiles import * 

import os 
import cv2 
import pdb
import time
import glob
import torch 
import numpy as np

from typing import Union, List, Any
from pathlib import Path 
from memory_profiler import profile as mem_profile
from abc import abstractmethod
from torch.profiler import  ProfilerActivity
from ultralytics.data.augment import LetterBox
from ultralytics.utils.torch_utils import smart_inference_mode

from ultralytics.utils import ops


class OptimizedStreamer(Streamer): 


    def __init__(self, cfg: str, overrides:dict, _callbacks:Any)->None: 
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        

    def __call__(self, source:str, model:str, logic_module=None, mqtt_broker=None, producer_flag=None, queue=None, *args, **kwargs)->None:
        self.mqtt_interface = mqtt_broker 
        self.args.stream_buffer = True 
        self.logic_module = logic_module 

        try: 
            self.predict_cli(source=os.path.normpath(os.path.abspath(source)) if os.path.isfile(source) else source, 
                model=model, 
                producer_flag=producer_flag, 
                queue=queue
            )

        except KeyboardInterrupt as ke: 

            if producer_flag is not None: 
                producer_flag.value=False 

            if self.logic_module is not None and self.logic_module["DAV2"] is not None: 
                self.logic_module["DAV2"].deallocate_resources() 

            cv2.destroyAllWindows() 
            Streamer.logger.exception(f"KeyboardInterrupt: {ke}")
        
        return 


    def pre_transform(self, im:List[np.ndarray])->List: 

        pt = None 

        if isinstance(self.model, CompressedYOLO) or isinstance(self.model, TensorRTYOLO): 
            pt = True 
            self.stride = 16 if self.args.half else 32 

        else: 
            raise ValueError("The type of the model parsed is incorrect")

        same_shapes = len({x.shape for x in im}) == 1 
        letterbox = LetterBox(self.imgsz,auto=same_shapes ^ pt, stride=self.stride)
        return [letterbox(image=x) for x in im]


    def preprocess(self, im: Union[torch.Tensor, List[np.ndarray]])-> torch.Tensor | List[np.ndarray]:
        pass

    
    def postprocess(self, preds:Any, img:Any, orig_imgs:Any)->Any : 
        return super().postprocess(preds, img, orig_imgs) 


    def setup_model(self, model:str, opt:str)->None:
        pass 


    @smart_inference_mode()
    def stream_inference(self, source:str, model:str, producer_flag:Any, queue:Any, *args, **kwargs):
        self.source = source 
        if self.args.verbose : Streamer.logger.info(" ")
        with self._lock: 
            self.setup_source(source if source is not None else self.args.source)
            self.seen = 0 
            self.results = [] 
            self.batch = None 
            self.mp = None 

            profilers = (
                ops.Profile(device=self.device), 
                ops.Profile(device=self.device), 
                ops.Profile(device=self.device), 
            )
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            start_time = time.perf_counter() 
            for batch in self.dataset: 
                _, im0s, _ = batch 
                self.orig_height, self.orig_width = im0s[0].shape[:2] 

            self.dataset.bs = BATCH_SIZE
            tile_flag = True if (self.orig_width // TILE_SIZE) > 2 or (self.orig_height //TILE_SIZE) >= 2 else False  
            if tile_flag : 
                Streamer.logger.info("Run Inference with Tiles")
                return self._stream_inference_impl_tiles(
                    model=model,
                    producer_flag = producer_flag, 
                    queue=queue, 
                    profilers=profilers, 
                    activities=activities, 
                    start_time=start_time
                )

            Streamer.logger.info("Run Inference without Tiles")
            return self._stream_inference_impl(
                  model=model,
                  producer_flag = producer_flag, 
                  queue=queue, 
                  profilers=profilers, 
                  activities=activities, 
                  start_time=start_time  
            )


    @abstractmethod
    @mem_profile
    def _stream_inference_impl(**kargs): 
        pass 


    @abstractmethod 
    @mem_profile
    def _stream_inference_impl_tiles(**kwargs): 
        pass


    def _frames_to_tiles(self, frame_iter, tile_size:int, overlap_ratio:float): 
        overlap_px = max(0, int(round(tile_size * overlap_ratio)))
        for f_id, img in frame_iter: 
            self.frame_images[f_id] = img
            for t, m in split_image_gen(img, f_id, tile_size=tile_size, overlap=overlap_px):
                yield t, m


    def iter_data(self, use_roi:bool): 

        """ Yields (frame_id, image) lazily from self.dataset after ROI + MOG2 gating """

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
                # self.orig_height, self.orig_width = im0s[0].shape[:2] 
 
            with StepContext(name="ROI Cropping", catch=(RuntimeError, ), verbose=self.args.verbose): 
                if use_roi: 
                    im0s = self.logic_module['ROI'].crop(im0s)

            with StepContext(name="FishEyE Processing (Defish)", catch=(RuntimeError, ), verbose=self.args.verbose):    
                # Defish FishEye camera frames to increase accuracy
                if self.logic_module["FEP"] is not None: 
                    im0s = self.logic_module["FEP"]._defish(im0s)

            with StepContext(name="BackGround Subtractor (Motion-Gating)", catch=(RuntimeError, Exception), verbose=self.args.verbose):
                # Motion gate (vectorized over the mini batch) 
                mfgs = self.logic_module["SUBTRACTOR"].detect(im0s, save_img=False) 

            if self.args.bench and self.mp is not None: 
 
                labels = [] 
                for filename in sorted(glob.glob(f'{self.args.bench_labels}/*.txt'), key=key_func):
                    labels.append(filename)

                self.__gt_labels = build_gt_index(labels, fixed_size=(FIXED_WIDTH, FIXED_HEIGHT))

                for passed, f_id, img, gt_file in zip(mfgs, frame_ids, im0s, labels): 
                    
                    stem = Path(gt_file).stem
                    gt_cls, gt_bbs = _get_gt(stem, self.__gt_labels)

                    if passed : 
                        self.__gt_labels[int(f_id)] = (gt_cls, gt_bbs)
                        yield (int(f_id), img)

                    else: 
                        empty_boxes, empty_scores, empty_cls = _empty_dets_numpy() 
                        self.mp.update(
                            boxes_xyxy=empty_boxes,
                            scores=empty_scores,
                            classes=empty_cls,
                            gt_boxes_xyxy=gt_bbs.astype(np.float32),
                            gt_classes=gt_cls.astype(np.int64),
                        )                
                        continue

            else: 
                for passed, f_id, img in zip(mfgs, frame_ids, im0s): 
                    if passed: 
                        yield (int(f_id), img)





