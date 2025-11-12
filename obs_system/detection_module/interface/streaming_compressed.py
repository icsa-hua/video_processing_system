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
from torch.profiler import profile
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
        return super().preprocess(im)

    
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
    def _stream_inference_impl(self, **kwargs): 
        model = kwargs["model"] 
        producer_flag  = kwargs["producer_flag"] 
        queue = kwargs["queue"]
        profilers=kwargs["profilers"] 
        activities=kwargs["activities"] 
        start_time = kwargs["start_time"]
        
        if self.args.bench: 
            self.mp = ModelPerf(
                class_ids=[i for i, _ in enumerate(self.converter.class_names)], 
                iou_thresholds=np.arange(0.50, 0.96, 0.05), 
                conf_threshold=CONF_THR, 
                use_101_point_interp=True
            )

        else: 
            self.mp = None 
    
        self.run_callbacks("on_predict_start") 
        with StepContext(name="Warmup Session", catch=(Exception, RuntimeError), verbose=self.args.verbose):
            if not self.done_warmup: 
                self.model.warmup(micro=BATCH_SIZE, warmup_sessions=WARM_UP_SESSIONS)
                self.done_warmup = True

        use_roi = True if self.logic_module is not None and self.logic_module["ROI"] is not None else False 
        first_batch = True

        for self.batch in self.dataset: 

            self.run_callbacks("on_predict_batch_start")
            paths, im0s, s = self.batch

            with StepContext(name="BackGround Subtractor (Motion-Gating)", catch=(RuntimeError, Exception), verbose=self.args.verbose):
                # Motion gate (vectorized over the mini batch) 
                mfgs = self.logic_module["SUBTRACTOR"].detect(im0s, save_img=False)
                Streamer.logger.info(mfgs) 

            if use_roi and ("frame_1" in s[0] or self.seen == 0): 
                self.logic_module['ROI'].set_regions(im0s[0]) 

            with StepContext(name="ROI Cropping", catch=(RuntimeError, ), verbose=self.args.verbose): 
                if use_roi: 
                    Streamer.logger.debug("ROI enabled")
                    im0s = self.logic_module['ROI'].crop(im0s)

            with StepContext(name="FishEyE Processing (Defish)", catch=(RuntimeError, ), verbose=self.args.verbose):    
                # Defish FishEye camera frames to increase accuracy
                if self.logic_module["FEP"] is not None: 
                    Streamer.logger.debug("FEP enabled")
                    im0s = self.logic_module["FEP"]._defish(im0s)  

            allowed_filter = [i for i, val in enumerate(mfgs) if val] 
            if len(allowed_filter) == 0 : 
                continue

            # paths = [paths[i] for i in allowed_filter]
            # s = [s[i] for i in allowed_filter]
            # im0s = [im0s[i] for i in allowed_filter]
            for i, al in enumerate(allowed_filter): 
                if not al: 
                    im0s[i] = empty_image(im0s[i])

            with profilers[0]: 
                images = self.preprocess(im0s) 

            with profilers[1]: 
                if first_batch: 
                    with profile(activities=activities) as prof: 
                        i_boxes, i_scores, i_classes = self.model(images, debug=self.args.verbose) 
                    prof.export_chrome_trace(f"trace_{model}.json")
                    first_batch = False
                else: 
                    i_boxes, i_scores, i_classes = self.model(images, debug=self.args.verbose) 
                
            with profilers[2]: 
                pass 

            self.run_callbacks("on_predict_postprocess_end")
            for boxes, scores, cls_, orig_img in zip(i_boxes, i_scores, i_classes, im0s): 
                if self.seen >= len(self.batch[1]): 
                    self.seen = 0
                    self.results.clear()

                if boxes is None or len(boxes) == 0: 
                    print("No detections for image : ", self.seen)
                    self.seen +=1
                    self.results.append(_empty_results(orig_img))
                    continue

                if isinstance(boxes, np.ndarray): 
                    boxes_t = torch.from_numpy(boxes)
                    scores_t = torch.from_numpy(scores) 
                    classes_t = torch.from_numpy(cls_) 
                else: 
                    boxes_t = boxes 
                    scores_t = scores 
                    classes_t = cls_

                keep_pc = batched_nms(
                        boxes_t, 
                        scores_t, 
                        classes_t.long(), 
                        iou_threshold=(1.0-NMS_IOU)
                    )
                    
                keep = keep_pc if keep_pc.numel()==0 else keep_pc[nms(boxes_t[keep_pc],scores_t[keep_pc], iou_threshold=(1-NMS_IOU))]
                boxes_t, scores_t, classes_t = boxes_t[keep], scores_t[keep], classes_t[keep] 
                inf_results = torch.stack(
                    (
                        boxes_t[:,0], 
                        boxes_t[:,1],
                        boxes_t[:,2],
                        boxes_t[:,3],
                        scores_t, 
                        classes_t
                    )
                )

                results = Results(
                    orig_img=orig_img, 
                    path=f"image_{self.seen}.jpg", 
                    names=self.converter.class_names, 
                    boxes=inf_results.T, 
                    speed={}, 
                )

                if self.tracker_model is not None:
                    results = self.tracker_model.detect(
                        predictions=results, 
                        save=False, 
                        orig_frame=orig_img, 
                        f_id=self.seen, 
                        class_names=self.converter.class_names
                    )
                
                results.speed = {
                    "preprocess": profilers[0].dt * 1e3/len(self.batch),
                    "inference": profilers[1].dt * 1e3/len(self.batch),
                    "postprocess": profilers[2].dt * 1e3 / len(self.batch),
                }

                self.results.append(results) 
                if self.mp is not None and self.args.bench: 
                    gt_cls, gt_bbs = self.__gt_labels.pop(self.seen, (np.zeros((0,), np.int64),np.zeros((0,4), np.float32)))
                    self.mp.update(
                        boxes_xyxy = boxes_t.cpu().numpy().astype(np.float32), 
                        scores = scores_t.cpu().numpy().astype(np.float32), 
                        classes = classes_t.cpu().numpy().astype(np.int32), 
                        gt_boxes_xyxy=gt_bbs.astype(np.float32), 
                        gt_classes = gt_cls.astype(np.int64)
                    ) 
                
                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    if mfgs[self.seen]: 
                        s[self.seen] += self.write_results(self.seen, Path(paths[self.seen]), images, im0s, s)
                    
                        if producer_flag is not None: 
                            producer_flag.value = True

                        if self.proc_image is not None and queue is not None:
                            queue.put(self.proc_image)

                        elif self.proc_image is None and queue is not None: 
                            queue.put(None)    

                        self.capture_object_boxes(orig_img, results, cropped_dirname=self.cropped_image_dirname)

                self.seen += 1 
                if self.seen == len(self.batch)-1 and self.args.verbose: 
                    elapsed_time=time.perf_counter() - start_time 
                    Streamer.logger.info(f"Time from capturing batch to meaningful information: {elapsed_time:.2f}")
            
            self.run_callbacks("on_predict_batch_end")
            yield from self.results

        
        if self.args.bench and self.mp is not None: 
            self.mp.finalize() 
            Streamer.logger.info(self.mp.results())
        
        for v in self.vid_writer.values(): 
            if isinstance(v, cv2.VideoWriter): 
                v.release() 

        if self.args.verbose and self.seen: 
            t = tuple(x.t / self.seen * 1e3 for x in profilers) 
            Streamer.logger.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, BATCH_SIZE)}" % t
            )

        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            Streamer.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        
        self.run_callbacks("on_predict_end")
















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





