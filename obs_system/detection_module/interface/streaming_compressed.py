from obs_system.utils.common import _get_gt, _empty_dets_numpy, _empty_results
from obs_system.compressed.interface.compressed_yolo import CompressedYOLO
from obs_system.compressed.interface.tensor_yolo import TensorRTYOLO
from obs_system.detection_module.interface.streamer import Streamer
from obs_system.utils.benchmarking.metrics.model_performance import ModelPerf
from obs_system.utils.global_config import BATCH_SIZE
from obs_system.utils.appraisal import StepContext, frame_list
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
import queue
import threading

from typing import Union, List, Any, Generator, Optional
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
        

    def __call__(self, source:str, model:str, logic_module=None, mqtt_broker=None, producer_flag=None, queue_list=None, *args, **kwargs)->None:
        self.mqtt_interface = mqtt_broker 
        self.args.stream_buffer = True 
        self.logic_module = logic_module 
        self.lanes_final = None

        try: 
            self.predict_cli(source=os.path.normpath(os.path.abspath(source)) if os.path.isfile(source) else source, 
                model=model, 
                producer_flag=producer_flag, 
                queue_list=queue_list
            )

        except KeyboardInterrupt as ke: 

            if producer_flag is not None: 
                producer_flag.value=False 

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
    def stream_inference(self, source:str, model:str, producer_flag:Any, queue_list:Any, *args, **kwargs)->Generator[Optional[Any], None, None]:

        self.source = source 

        if self.args.verbose : Streamer.logger.info(" ")

        with self._lock: 
            with StepContext(name="Set up Dataloader Process", catch=(RuntimeError, ), verbose=True):
                self.setup_source(source if source is not None else self.args.source)
                self.dataset.bs = BATCH_SIZE

            self.seen = 0 
            self.results = [] 
            self.batch = None 
            self.mp = None 
                       
            #self.use_roi = True if self.args.roi and self.logic_module["ROI"] is not None else False 
            self.use_roi = False 

            profilers = (
                ops.Profile(device=self.device), 
                ops.Profile(device=self.device),
                ops.Profile(device=self.device)
            )

            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            start_time = time.perf_counter() 

            first_batch = next(iter(self.dataset)) #Keeps the original pointer without moving it "peeking" to the first frame 
            _, im0s, _ = first_batch
            self.orig_height, self.orig_width = im0s[0].shape[:2] 

            empty_image = f"{EMPTY_IMAGE_PATH}"
            if not os.path.exists(empty_image):
                raise FileNotFoundError(f"Empty image path for background subtraction does not exist: {empty_image}")

            if self.use_roi :
                with StepContext(name="ROI Cropping", catch=(RuntimeError, ), verbose=self.args.verbose): 
                    
                    self.logic_module['ROI'].set_regions(im0s[0]) 
                    cropped_frame = self.logic_module['ROI'].crop_image(im0s[0])
                    
                    if self.args.show: 
                        self.logic_module['ROI']._show_regions(cropped_frame.copy())
                    
                    self.orig_height, self.orig_width = cropped_frame.shape[:2]
                empty_image = cv2.imread(empty_image)
                self.logic_module['SUBTRACTOR'].warm_up(empty_image, trials=TRIALS)
            
            else: 
                self.logic_module['SUBTRACTOR'].warm_up(empty_image, trials=TRIALS)
            
            tile_flag = True if (self.orig_width // TILE_SIZE) > TILE_THR or (self.orig_height //TILE_SIZE) >= TILE_THR else False  

            if tile_flag : 
                Streamer.logger.info("Run Inference with Tiles")
                return self._stream_inference_impl_tiles(
                    model=model,
                    producer_flag = producer_flag, 
                    queue_list=queue_list, 
                    profilers=profilers, 
                    activities=activities, 
                    start_time=start_time
                )

            Streamer.logger.info("Run Inference without Tiles")
            return self._stream_inference_impl(
                  model=model,
                  producer_flag = producer_flag, 
                  queue_list=queue_list, 
                  profilers=profilers, 
                  activities=activities, 
                  start_time=start_time  
            )


    @abstractmethod
    @mem_profile
    def _stream_inference_impl(self, **kwargs): 
        model = kwargs["model"] 
        producer_flag  = kwargs["producer_flag"] 
        queue_list = kwargs["queue_list"]
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

        # Asynchronous batch loading to avoid stalls
        batch_queue = queue.Queue(maxsize=2)

        def producer():
            try:
                for batch in self.dataset:
                    batch_queue.put(batch)
            except Exception as e:
                Streamer.logger.error(f"Error in batch producer: {e}")
            finally:
                batch_queue.put(None)  # sentinel

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        while True:
            self.batch = batch_queue.get()
            if self.batch is None:
                break

            self.run_callbacks("on_predict_batch_start")

            paths, im0s, s = self.batch

            #use_roi = True if self.args.roi and self.logic_module["ROI"] is not None else False 
            if self.use_roi :
                with StepContext(name="ROI Cropping", catch=(RuntimeError, ), verbose=self.args.verbose): 
                    im0s = self.logic_module['ROI'].crop_image(im0s)

            with StepContext(name="BackGround Subtractor (Motion-Gating)", catch=(RuntimeError, Exception), verbose=self.args.verbose):
                # Motion gate (vectorized over the mini batch) 
                mfgs, lanes_final = self.logic_module["SUBTRACTOR"].detect(im0s, save_img=False)
                if lanes_final is not None: 
                    self.lanes_final = lanes_final

            with StepContext(name="FishEyE Processing (Defish)", catch=(RuntimeError, ), verbose=self.args.verbose):    
                # Defish FishEye camera frames to increase accuracy
                if self.logic_module["FEP"] is not None: 
                    Streamer.logger.debug("FEP enabled")
                    im0s = self.logic_module["FEP"]._defish(im0s)  


            # Speeds up the process when no motion is detected in the incoming batch. 
            if not any(mfgs):
                yield return_no_motion_frames(
                    im0s=im0s,
                    batch_size=BATCH_SIZE 
                )

                # here normally the MQTT should update with no detections the publisher. 
                

            for i, keep in enumerate(mfgs):
                if not keep:
                    print("Empty movement detected, skipping frame.")
                    im0s[i] = empty_image(im0s[i])

            with profilers[0]: 
                images = self.preprocess(im0s) 

            with profilers[1]: 
                if self.seen == 0 and self.args.verbose: 
                    with profile(activities=activities) as prof:
                        (i_boxes, i_scores, i_classes), event = self.model(images, orig_imgs=im0s, debug=self.args.verbose)
                    prof.export_chrome_trace(f"trace_{model}.json")
                else: 
                    (i_boxes, i_scores, i_classes), event = self.model(images, orig_imgs=im0s, debug=self.args.verbose)

            if model=='engine':
                # End event and synchronize the engine after we don't need the tensors anymore 
                # Transfer them into the CPU stream 
                torch.cuda.current_stream().wait_event(event)

            for bni, (boxes, scores, cls_, orig_img) in enumerate(zip(i_boxes, i_scores, i_classes, im0s)): 

                self.seen = bni

                if boxes is None or len(boxes) == 0: 
                    inf_results = _empty_results(orig_image=orig_img)
                    if self.results is not None:
                        self.results.append(inf_results)
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
                    "preprocess": profilers[0].dt * 1e3/len(im0s),
                    "inference": profilers[1].dt * 1e3/len(im0s),
                    "postprocess": (time.perf_counter() - start_time) * 1e3/len(im0s)
                }

                if self.results is not None:
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
                
                yield results

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    filename=Path(paths[self.seen])
                    if not filename: 
                            Streamer.logger.warning("[WARNING]: filename to save image is invalid")

                    self.batch[2][self.seen] += self.write_results(
                            i = self.seen, 
                            p = filename,  
                            im= images,
                            original_images=self.batch[1], 
                            s = self.batch[2]
                        )

                if producer_flag is not None: 
                    producer_flag.value = True

                if self.proc_image is not None and queue_list is not None:
                    queue_list.put(self.proc_image)

                elif self.proc_image is None and queue_list is not None: 
                    queue_list.put(None)    

                with StepContext(name="Crop Objects to Image", catch=(RuntimeError,), verbose=self.args.verbose):
                        try: 
                            self.capture_object_boxes(
                                    image=self.batch[1][self.seen],
                                    results=self.results[self.seen],
                                    cropped_dirname=self.cropped_image_dirname,
                                    save=self.args.save
                            )
                        except IndexError as ie: 
                            Streamer.logger.exception(ie)
                
                if self.seen >= len(im0s): 
                    self.results.clear()

                if self.seen == len(im0s)-1 and self.args.verbose: 
                    elapsed_time=time.perf_counter() - start_time 
                    Streamer.logger.info(f"Time from capturing batch to meaningful information: {elapsed_time:.2f}")
            
            self.run_callbacks("on_predict_postprocess_end")
            self.run_callbacks("on_predict_batch_end")

        producer_thread.join()

        self.save_queue.put(None)
        self.save_thread.join()

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
        
        self.run_callbacks("on_predict_end")


    @abstractmethod 
    @mem_profile
    def _stream_inference_impl_tiles(self, **kwargs)->Generator[Optional[Any], None, None]: 
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
 
            with StepContext(name="ROI Cropping", catch=(RuntimeError, ), verbose=self.args.verbose): 
                if use_roi: 
                    im0s = self.logic_module['ROI'].crop_image(im0s)

            with StepContext(name="FishEyE Processing (Defish)", catch=(RuntimeError, ), verbose=self.args.verbose):    
                # Defish FishEye camera frames to increase accuracy
                if self.logic_module["FEP"] is not None: 
                    im0s = self.logic_module["FEP"]._defish(im0s)

            with StepContext(name="BackGround Subtractor (Motion-Gating)", catch=(RuntimeError, Exception), verbose=self.args.verbose):
                # Motion gate (vectorized over the mini batch) 
                mfgs, lanes_final = self.logic_module["SUBTRACTOR"].detect(im0s, save_img=False) 
                if  lanes_final is not None and len(lanes_final) != 0 : 
                    self.lanes_final = lanes_final

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





