from obs_system.compressed.interface.compressed_yolo import CompressedYOLO 
from obs_system.detection_module.interface.streaming_compressed import OptimizedStreamer
from obs_system.logic_module.dummy_logic.tracker_sv import TrackerHandler
from obs_system.utils.tiles import *
from obs_system.utils.appraisal import StepContext, frame_list
from obs_system.utils.common import _empty_dets_numpy
from obs_system.utils.logger import get_logger 
from obs_system.utils.global_config import CONF_THR, NMS_IOU, TILE_SIZE, WARM_UP_SESSIONS, BATCH_SIZE, MIN_WH, MULTIPLIER


import os 
import gc
import pdb
import torch
import cv2
import time
import numpy as np 
 
from collections import defaultdict
from typing import Any, List
from memory_profiler import profile as mem_profile
from queue import Queue 
from pathlib import Path 
from ultralytics import YOLO
from torch.profiler import profile, ProfilerActivity 
from torchvision.ops import batched_nms, nms
from ultralytics.utils import DEFAULT_CFG, ops, callbacks
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils import colorstr 

logger = get_logger("obs_system."+__name__)

class OnnxY8Streamer(OptimizedStreamer): 

    def __init__(self, cfg:Any=DEFAULT_CFG, overrides=None, _callbacks=None)->None: 
        super().__init__(cfg,overrides, _callbacks)
        self.source = ""
        self.__gt_labels = None if self.args.bench is None else defaultdict()


    def from_numpy(self, x: np.ndarray):
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x


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

        if (not isinstance(self.model, YOLO) and not isinstance(self.model, CompressedYOLO)): 
            im = im.half() if self.model.fp16 else im.float() 

        else: 
            im = im.float() 

        if not_tensor:
            im = im.div(255.0)  # 0 - 255 to 0.0 - 1.0

        #from torchvision.utils import save_image
        #save_image(im,'debug.png')

        return im


    def postprocess(self, preds, img, orig_imgs): 
        return super().postprocess(preds, img, orig_imgs) 


    @staticmethod
    def check_onnx_model(model_path, device):
        if not os.path.exists(model_path):
            model = YOLO('yolov8s.pt') 
            model.export(
                format="onnx",
                imgsz=(TILE_SIZE), 
                dynamic=True,
                simplify=True
            )
    

    def setup_model(self, model_name, path_to_load, opt='tracking'): 

        device = select_device(self.args.device, verbose=self.args.verbose) 

        model_path = 'obs_system/compressed/yolov8s_original.onnx'

        self.check_onnx_model(model_path, device)  

        self.model = CompressedYOLO(model_path) 

        [self.height, self.width] = self.model.input_height, self.model.input_width 

        self.device = device 

        self.tracker_model = TrackerHandler(tracker_choice="byte_tracker") if opt =="tracking" else None

        self.stride = 32 if not self.args.half else 16

        if self.args.verbose: 
            logger.info(f"[checked] Model {model} successfully set up")


    @smart_inference_mode()
    def stream_inference(self, source, model, producer_flag, preview_queue, *args, **kwargs): 
        return super().stream_inference(source, model, producer_flag, preview_queue, *args, **kwargs) 


    @mem_profile
    def _stream_inference_impl_tiles(self,  **kwargs): 
        
        model = kwargs["model"] 
        producer_flag  = kwargs["producer_flag"] 
        preview_queue = kwargs["preview_queue"]
        profilers=kwargs["profilers"] 
        activities=kwargs["activities"] 
        start_time = kwargs["start_time"]

        self.frame_images = {} 
        micro = BATCH_SIZE
        overlap_ratio = TILE_OVERLAP

        self.run_callbacks("on_predict_start") 

        # Warmup for Better Inference. Reduces Initial frames high inference time and is more stable. 
        with StepContext(name="Warmup Session", catch=(Exception, RuntimeError), verbose=self.args.verbose):
            if not self.model_warmup_done: 
                self.model.warmup(micro=micro, warmup_sessions=WARM_UP_SESSIONS)
                self.model_warmup_done = True

        use_roi = True if self.logic_module is not None and self.logic_module["ROI"] is not None else False 

        host0 = np.empty((micro, self.imgsz[0], self.imgsz[1], 3), np.uint8) 
        metas0 = [None]*micro 
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
                self.results.clear()

            with profilers[0]:
                tb = (self.preprocess(host0[:n0])).to(self.device, non_blocking=True) 
                tbuf[:n0].copy_(tb, non_blocking=True) 

            with profilers[1]:
                if self.seen == 0 and self.args.verbose: 
                    with profile(activities=activities) as prof:
                        i_boxes, i_scores, i_classes = self.model(tbuf[:n0], debug=True) 
                    prof.export_chrome_trace(f"trace_{model}.json")
                else: 
                    i_boxes, i_scores, i_classes = self.model(tbuf[:n0], debug=True)

            with profilers[2]: 
                pass 
                
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

                    s["parts"].append((mapped_boxes, score, cls_)) 

                    if len(s["seen"]) == s["need"]: 
                        if not s["parts"] or all(p[0].shape[0] == 0 for p in s["parts"]): 
                            B = np.empty((0,4), np.float32) 
                            S = np.empty((0,), np.float32) 
                            C = np.empty((0,), np.float32) 

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
                    boxes_t = torch.from_numpy(boxes)
                    scores_t = torch.from_numpy(scores) 
                    classes_t = torch.from_numpy(classes) 

                    keep_pc = batched_nms(
                        boxes_t, 
                        scores_t, 
                        classes_t.long(), 
                        iou_threshold=(1.0-NMS_IOU)
                    )

                    keep = keep_pc if keep_pc.numel()==0 else keep_pc[nms(boxes_t[keep_pc],scores_t[keep_pc], iou_threshold=(1-self.args.iou))]
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
                        probs=classes
                    )

                    if self.tracker_model is not None:
                        results = self.tracker_model.detect(
                            predictions=results, 
                            save=False, 
                            orig_frame=self.frame_images[f_id], 
                            f_id=f_id, 
                            class_names=self.converter.class_names
                        )
                    
                    self.results.append(results)                                             
                    self.frame_images.pop(f_id,None)
                    del inf_results 
                    yield results

                    bn = len(frames_out.keys())
                    self.results[self.seen].speed = {
                            "preprocess": profilers[0].dt * 1e3 / bn,
                            "inference": profilers[1].dt * 1e3 / bn, 
                            "postprocess": profilers[2].dt * 1e3 / bn
                    }

                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show: 
                        self.batch[2][self.seen] += self.write_results(
                            i = self.seen, 
                            p = Path(self.batch[0][self.seen]), 
                            im= tbuf[:n0],
                            original_images=self.batch[1], 
                            s = self.batch[2]
                        )

                    self.publish_preview(preview_queue, producer_flag)

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
                        logger.info(f"Time from Capturing batch to meaningfull inference is {elapsed_time:.2f}")

            self.run_callbacks("on_predict_postprocess_end")
            self.run_callbacks("on_predict_batch_end")
            frame_list.append((time.perf_counter() -t0) * 1000)

        self.close_preview_stream(preview_queue)

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
        super()._stream_inference_impl(**kwargs) 



























