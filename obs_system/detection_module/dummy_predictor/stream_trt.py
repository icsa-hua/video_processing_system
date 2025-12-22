from obs_system.detection_module.interface.streaming_compressed import OptimizedStreamer
from obs_system.utils.benchmarking.metrics.model_performance import ModelPerf 
from obs_system.logic_module.dummy_logic.tracker_sv import TrackerHandler
from obs_system.compressed.interface.tensor_yolo import TensorRTYOLO
from obs_system.utils.tiles import *
from obs_system.utils.logger import get_logger 
from obs_system.utils.appraisal import StepContext, frame_list
from obs_system.utils.common import _empty_dets_numpy, _empty_results, empty_image
from obs_system.utils.global_config import CONF_THR, NMS_IOU, WARM_UP_SESSIONS, BATCH_SIZE, MIN_WH, MULTIPLIER


import pdb
import time 
import torch
import cv2
import numpy as np 

from typing import Any, Generator, Optional
from memory_profiler import profile as mem_profile
from pathlib import Path 
from ultralytics import YOLO
from torch.profiler import profile
from torchvision.ops import batched_nms, nms
from collections import defaultdict
from ultralytics.utils import DEFAULT_CFG
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils import colorstr 


logger = get_logger("obs_system."+__name__)


class TensorRTRTXStreamer(OptimizedStreamer): 

    def __init__(self, cfg:Any=DEFAULT_CFG, overrides=None, _callbacks=None)->None: 
        super().__init__(cfg,overrides, _callbacks)
        self.source = ""
        self.__gt_labels = None if self.args.bench is None else defaultdict()


    def __call__(self, source=None, model=None, logic_module=None, mqtt_broker=None,producer_flag=None, queue_list=None, *args, **kwargs): 
        super().__call__(source, model, logic_module, mqtt_broker, producer_flag, queue_list , *args, **kwargs)


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


    def setup_model(self, model, opt='tracking'): 
        device = select_device(self.args.device, verbose=self.args.verbose) 

        model_path = 'obs_system/compressed/yolov8s_original.onnx'
        self.model = TensorRTYOLO(engine_path=model_path, fp16=False)

        [self.height, self.width] = self.model.input_height, self.model.input_width 

        self.device = device 

        self.tracker_model = TrackerHandler(tracker_choice="byte_tracker") if opt == "tracking" else None 

        self.stride = 32 if not self.args.half else 16

        if self.args.verbose: 
            logger.info(f"[checked] Model {model} successfully set up")


    @smart_inference_mode()
    def stream_inference(self, source, model, producer_flag, queue_list, *args, **kwargs): 
        return super().stream_inference(source, model, producer_flag, queue_list, *args, **kwargs) 


    @mem_profile
    def _stream_inference_impl_tiles(self, **kwargs)->Generator[Optional[Any], None, None]: 
        model = kwargs["model"] 
        producer_flag  = kwargs["producer_flag"] 
        queue_list = kwargs["queue_list"]
        profilers=kwargs["profilers"] 
        activities=kwargs["activities"] 
        start_time = kwargs["start_time"]

        if self.args.bench :
             self.mp = ModelPerf(
                class_ids=[i for i, _ in enumerate(self.converter.class_names)], 
                iou_thresholds=np.arange(0.50, 0.96, 0.05), 
                conf_threshold=CONF_THR, 
                use_101_point_interp=True
                )
        else: 
            self.mp = None 
        
        micro = BATCH_SIZE
        overlap_ratio = TILE_OVERLAP
        
        self.run_callbacks("on_predict_start") 

        # Warmup for Better Inference. Reduces Initial frames high inference time and is more stable. 
        with StepContext(name="Warmup Session", catch=(Exception, RuntimeError), verbose=self.args.verbose):
            if not self.done_warmup: 
                self.model.warmup(micro=micro, warmup_sessions=WARM_UP_SESSIONS)
                self.done_warmup = True

        use_roi = True if self.logic_module is not None and self.logic_module["ROI"] is not None else False 

        metas0 = [None]*micro 
        host0 = np.empty((micro, self.imgsz[0], self.imgsz[1], 3), np.uint8) 
        tbuf = torch.empty((micro, 3, self.imgsz[0], self.imgsz[1]), device=self.device, dtype=torch.float32)
        
        # Generators 
        frame_iter = self.iter_data(use_roi=use_roi)
        tile_stream = self._frames_to_tiles(frame_iter, tile_size=self.imgsz[0], overlap_ratio=overlap_ratio)

        pending = {} 
        while True: 
            t0 = time.perf_counter()  
            n0 = next_microbatch(tile_stream, micro, host0, metas0)

            if n0 == 0: break 
            
            if not self.batch or not self.batch[1]: break 

            if self.seen >= len(self.batch[1]): 
                self.seen = 0 
                if self.results is not None: 
                    self.results.clear()

            with profilers[0]:
                tb = (self.preprocess(host0[:n0])).to(self.device, non_blocking=True) 
                tbuf[:n0].copy_(tb, non_blocking=True) 

            with profilers[1]:
                if self.seen == 0 and self.args.verbose: 
                    with profile(activities=activities) as prof:
                        (i_boxes, i_scores, i_classes), event = self.model(tbuf[:n0], debug=True) 
                    prof.export_chrome_trace(f"trace_{model}.json")
                else: 
                    (i_boxes, i_scores, i_classes), event = self.model(tbuf[:n0], debug=True)

            with profilers[2]: 
                pass 

            if model=='engine':
                torch.cuda.current_stream().wait_event(event)
            
            with StepContext(name="Post Process", catch=(Exception, RuntimeError), verbose=self.args.verbose): 
                frames_out = {} 
                for det, score, cls_, meta in zip(i_boxes, i_scores, i_classes, metas0[:n0]): 

                    if meta is None:
                        continue 

                    f_id = int(meta["frame_id"])
                    s = pending.setdefault(
                        f_id, 
                        {"need":(meta['grid'][0]*meta['grid'][1]), 
                         "seen":set(), 
                         "parts":[]
                        }
                    )
                    s['seen'].add(meta['t_idx'])

                    if det is None or len(det) == 0:
                        mapped_boxes, score, cls_ = _empty_dets_numpy()

                    else: 
                        mapped_boxes = reconstruct_tiles(
                            boxes_xyxy=det, 
                            tx=meta['left_x'], 
                            ty=meta['top_y'], 
                            orig_H=meta['f_wh'][0], 
                            orig_W=meta['f_wh'][1], 
                            gain=meta['gain'], 
                            pad=(meta['pad_x'],meta['pad_y'])
                        )


                        mapped_boxes = mapped_boxes.cpu().numpy() if isinstance(mapped_boxes,torch.Tensor) else mapped_boxes
                        score = score.cpu().numpy() if isinstance(score, torch.Tensor) else score
                        cls_ = cls_.cpu().numpy() if isinstance(cls_, torch.Tensor) else cls_

                    s["parts"].append((mapped_boxes, score, cls_)) 
                    if len(s["seen"]) == s["need"]: 
                        if not s["parts"] or all(p[0].shape[0] == 0 for p in s["parts"]): 
                            B,S,C = _empty_dets_numpy()

                        else: 
                            B = np.concatenate([p[0] for p in s["parts"]], 0)
                            S = np.concatenate([p[1] for p in s["parts"]], 0) 
                            C = np.concatenate([p[2] for p in s["parts"]], 0)               

                        frames_out[f_id] = (B,S,C) 
                        del pending[f_id]

                del i_boxes
                del i_scores 
                del i_classes 
                
                for f_id,(boxes, scores, classes) in frames_out.items(): 
                    
                    if self.seen >= len(self.batch[1]): 
                        self.seen = 0

                    if isinstance(boxes, np.ndarray): 
                        boxes_t = torch.from_numpy(boxes)
                        scores_t = torch.from_numpy(scores) 
                        classes_t = torch.from_numpy(classes) 
                    else: 
                        boxes_t = boxes 
                        scores_t = scores 
                        classes_t = classes 

                    # small = ((boxes_t[:,2]-boxes[:,0] < MIN_WH) | (boxes[:,3]-boxes[:,1] < MIN_WH))
                    # scores[small] *= MULTIPLIER

                   # GLOBAL NMS
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
                        orig_img=self.frame_images[f_id], 
                        path=f"image_{f_id}.jpg", 
                        names=self.converter.class_names, 
                        boxes=inf_results.T, 
                        speed={}, 
                    )

                    if self.tracker_model is not None:
                        results = self.tracker_model.detect(
                            predictions=results, 
                            save=False, 
                            orig_frame=self.frame_images[f_id], 
                            f_id=f_id, 
                            class_names=self.converter.class_names
                        )
                    if self.results is not None: 
                        self.results.append(results)     
                                                                
                    self.frame_images.pop(f_id,None)

                    if self.mp is not None and self.args.bench: 
                        gt_cls, gt_bbs = self.__gt_labels.pop(f_id, (np.zeros((0,), np.int64),np.zeros((0,4), np.float32)))
                        self.mp.update(
                            boxes_xyxy = boxes_t.cpu().numpy().astype(np.float32), 
                            scores = scores_t.cpu().numpy().astype(np.float32), 
                            classes = classes_t.cpu().numpy().astype(np.int32), 
                            gt_boxes_xyxy=gt_bbs.astype(np.float32), 
                            gt_classes = gt_cls.astype(np.int64)
                        )

                    del inf_results 
                    yield results

                    bn = len(frames_out.keys())
                    self.results[self.seen].speed = {
                            "preprocess": profilers[0].dt * 1e3 / bn,
                            "inference": profilers[1].dt * 1e3 / bn, 
                            "postprocess": profilers[2].dt * 1e3 / bn
                    }

                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show: 
                        filename=Path(self.batch[0][self.seen])
                        if not filename: 
                            logger.warning("[WARNING]: filename to save image is invalid")
                        
                        self.batch[2][self.seen] += self.write_results(
                            i = self.seen, 
                            p = filename,  
                            im= tbuf[:n0],
                            original_images=self.batch[1], 
                            s = self.batch[2]
                        )

                    if producer_flag is not None : 
                       producer_flag.value = True 

                    if self.proc_image is not None and queue_list is not None: 
                        queue_list[0].put(self.proc_image) 
                    elif self.proc_image is None and queue_list is not None: 
                        queue_list[0].put(None) 

                    with StepContext(name="Crop Objects to Image", catch=(RuntimeError,), verbose=self.args.verbose):
                        try: 
                            self.capture_object_boxes(
                                    image=self.batch[1][self.seen],
                                    results=self.results[self.seen],
                                    cropped_dirname=self.cropped_image_dirname,
                                    save=self.args.save
                            )
                        except IndexError as ie: 
                            logger.exception(ie)

                    self.seen += 1 
                    if self.seen >= len(self.batch[1]): 
                        self.seen = 0
                        self.results.clear()

                    if self.seen == len(self.batch)-1 and self.args.verbose: 
                        elapsed_time = time.perf_counter() - start_time
                        logger.info(f"Time From Capturing batch to meaningfull inference is {elapsed_time:.2f}")

            self.run_callbacks("on_predict_postprocess_end")
            self.run_callbacks("on_predict_batch_end")
            frame_list.append((time.perf_counter() - t0) * 1000)

        self.save_queue.put(None)
        self.save_thread.join() 

        if self.args.bench and self.mp is not None: 
            self.mp.finalize() 
            logger.info(self.mp.results())
        
        for v in self.vid_writer.values(): 
            if isinstance(v, cv2.VideoWriter): 
                v.release() 

        if self.args.verbose and self.seen: 
            t = tuple(x.t / self.seen * 1e3 for x in profilers) 
            logger.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *tbuf.shape[2:])}" % t
            )

        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            logger.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        
        self.run_callbacks("on_predict_end")        

    @mem_profile
    def _stream_inference_impl(self, **kwargs): 
        return super()._stream_inference_impl(**kwargs) 

    def _publish_mqtt_message(self, preds, frame_index)->None: 
        super()._publish_mqtt_message(preds, frame_index)

    def _publish_mqtt_message_no_detection(self, preds, frame_index)->None: 
        super()._publish_mqtt_message_no_detection(preds, frame_index)






















